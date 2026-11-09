"""Train a set of heads on cached features, as described by an experiment YAML file.

    python -I scripts/run_heads.py --config configs/experiments/tsinghua_main.yaml
    python -I scripts/run_heads.py --config ... --only rr_rival,linear --seeds 0

Every run writes experiments/<experiment>/<tag>_s<seed>/ (curves.csv, best.pt, summary.json,
val_preds.npz) and appends one row to experiments/registry.csv. Finished runs are skipped,
interrupted runs resume from last.pt. The test split is NOT loaded here.

Config format (YAML):
    name: tsinghua_main
    dataset: tsinghua
    backbone: dinov2_vitb14_reg
    img_size: 224
    views: [orig]                # cached training views (e.g. [orig, hflip])
    seeds: [0, 1, 2]
    common: {epochs: 30, lr: 1.0e-3, ...}       # train_heads.DEFAULTS overrides
    runs:
      - {tag: linear, head: {name: linear}}
      - {tag: rr_rival, head: {name: rerank, mode: rival, k: 5}, lr: 1.0e-3}
    lr_grid: [3.0e-4, 1.0e-3, 3.0e-3]   # optional: model selection on seed 0 (val only)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

from dogbreed import paths
from dogbreed.datasets.manifest import class_names, load_manifest
from dogbreed.training.cached import load_split
from dogbreed.training.train_heads import train_head
from dogbreed.utils import auto_device, env_info, get_logger, load_json, save_json

log = get_logger()
REGISTRY_FIELDS = ["time", "experiment", "run", "tag", "seed", "dataset", "backbone", "head",
                   "lr", "epochs_run", "best_epoch", "val_top1", "val_top5", "val_nll",
                   "val_top1_global", "val_cand_recall", "val_flip_rate", "params", "train_sec",
                   "code_hash"]


def append_registry(row: dict) -> None:
    reg = paths.EXPERIMENTS / "registry.csv"
    reg.parent.mkdir(parents=True, exist_ok=True)
    new = not reg.exists()
    with open(reg, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def load_data(cfg, df, device, splits=("train", "val")):
    view_dirs = [paths.feature_dir(cfg["dataset"], cfg["backbone"], cfg.get("img_size", 224), v,
                                   cfg.get("feature_tag", "")) for v in cfg.get("views", ["orig"])]
    data = {}
    for s in splits:
        dirs = view_dirs if s == "train" else view_dirs[:1]
        t0 = time.perf_counter()
        data[s] = load_split(dirs, df, s, device, cfg.get("tokens_on_device", "auto"),
                             cfg.get("max_device_fraction", 0.6), token_dims=cfg.get("token_dims"))
        log.info("loaded %s: %d images, tokens %s on %s (%.0f s)", s, data[s].n,
                 tuple(data[s].tok[0].shape), data[s].tok[0].device, time.perf_counter() - t0)
    return data


def expand_runs(cfg, only=None, seeds=None, select_lr=False):
    """Yield (tag, seed, run config) for every requested run."""
    seeds = seeds if seeds is not None else cfg.get("seeds", [0])
    for r in cfg["runs"]:
        if only and r["tag"] not in only:
            continue
        base = {**cfg.get("common", {}), **{k: v for k, v in r.items() if k not in ("tag",)}}
        if select_lr:
            for lr in cfg["lr_grid"]:
                yield f"{r['tag']}_lr{lr:g}", 0, {**base, "lr": lr}
        else:
            for s in seeds:
                yield r["tag"], s, dict(base)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="comma list of run tags")
    ap.add_argument("--seeds", default=None, help="comma list overriding the config seeds")
    ap.add_argument("--select-lr", action="store_true",
                    help="run the lr grid on seed 0 and write the chosen lr per tag (val only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", default=[], help="override common: key=value (YAML)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.select_lr and not cfg.get("lr_grid"):
        log.info("%s has no lr_grid (its rates come from lr_selection_from): nothing to select", a.config)
        return 0
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg.setdefault("common", {})[k] = yaml.safe_load(v)
    only = set(a.only.split(",")) if a.only else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    exp_dir = paths.EXPERIMENTS / cfg["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config": cfg, "env": env_info()}, exp_dir / f"config_{time.strftime('%Y%m%d_%H%M%S')}.json")

    sel_path = exp_dir / "lr_selection.json"
    chosen = load_json(sel_path) if sel_path.exists() and not a.select_lr else {}
    if cfg.get("lr_selection_from") and not a.select_lr:
        # ablations reuse the learning rate selected (on validation) for their base head in the main
        # experiment: run key 'lr_like' names that head's tag there
        other = load_json(paths.EXPERIMENTS / cfg["lr_selection_from"] / "lr_selection.json")
        for r in cfg["runs"]:
            src = r.get("lr_like", r["tag"])
            if "lr" not in r and r["tag"] not in chosen and src in other:
                chosen[r["tag"]] = other[src]
        log.info("learning rates taken from %s: %s", cfg["lr_selection_from"], chosen)
    device = auto_device(a.device)
    df = load_manifest(cfg.get("manifest") or paths.MANIFESTS / f"{cfg['dataset']}.csv")
    if cfg.get("class_subset"):                     # e.g. known breeds of the open-set protocol
        keep = set(load_json(cfg["class_subset"])["known"])
        df = df[df["label"].isin(keep)].copy()
        remap = {c: i for i, c in enumerate(sorted(keep))}
        df["label"] = df["label"].map(remap)
    names = class_names(df)
    data = load_data(cfg, df, device)
    results = {}
    for tag, seed, rc in expand_runs(cfg, only, seeds, a.select_lr):
        base_tag = tag.split("_lr")[0]
        if not a.select_lr and base_tag in chosen and "lr" not in next(
                r for r in cfg["runs"] if r["tag"] == base_tag):
            rc["lr"] = chosen[base_tag]
        run_cfg = {**rc, "seed": seed, "dataset": cfg["dataset"], "backbone": cfg["backbone"],
                   "img_size": cfg.get("img_size", 224), "experiment": cfg["name"], "tag": base_tag}
        out = exp_dir / f"{tag}_s{seed}"
        log.info("=== %s seed %d lr %g", tag, seed, run_cfg.get("lr", 0))
        fresh = not (out / "summary.json").exists()
        grid_run = exp_dir / f"{base_tag}_lr{run_cfg.get('lr', 0):g}_s0"
        if fresh and not a.select_lr and seed == 0 and (grid_run / "summary.json").exists():
            # The seed-0 run at the chosen lr IS the winning grid run (same config, same seed):
            # reuse it instead of training it twice.
            shutil.copytree(grid_run, out, dirs_exist_ok=True)
            log.info("reusing %s for %s (identical configuration)", grid_run.name, out.name)
        s = train_head(run_cfg, data["train"], data["val"], out, device, len(names), names)
        results[(tag, seed)] = s
        if fresh:
            append_registry({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": cfg["name"],
                             "run": out.name, "tag": base_tag, "seed": seed, "dataset": cfg["dataset"],
                             "backbone": cfg["backbone"], "head": str(rc["head"]),
                             "lr": run_cfg.get("lr"), **{k: s.get(k) for k in REGISTRY_FIELDS if k in s}})
    if a.select_lr:
        best = {}
        for (tag, _), s in results.items():
            base_tag, lr = tag.split("_lr")[0], float(tag.split("_lr")[1])
            key = (s["val_top1"], -s["val_nll"])
            if base_tag not in best or key > best[base_tag][0]:
                best[base_tag] = (key, lr)
        save_json({k: v[1] for k, v in best.items()}, sel_path)
        log.info("lr selection (val top-1, seed 0): %s", {k: v[1] for k, v in best.items()})
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
