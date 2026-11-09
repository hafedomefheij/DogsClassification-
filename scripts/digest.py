"""A compact text digest of every result, small enough to copy out of the runtime and check by hash.

    python -I scripts/digest.py                       # -> results/digest/
    python -I scripts/digest.py --curves tsinghua_main/rr_rival_s0,tsinghua_main/linear_s0

Files (only those whose sources exist; numbers rounded to 5 significant digits):
  runs.csv            every training run: experiment, run, tag, seed, lr, head, val metrics, epochs, params, seconds
  lr_selection.csv    experiment, tag, chosen lr
  test.csv            final-evaluation summaries: experiment, view, tag, metric, mean, std, n, ci95
  test_runs.csv       final-evaluation rows per seed: experiment, view, tag, seed, metrics
  comparisons.csv     experiment, view, pair, seed, diff, ci95 ("low;high", paired bootstrap), p_boot_le0,
                      a_only, b_only (images only A / only B gets right), p_value (exact McNemar)
  hard_pairs.csv      experiment, view, tag, pair_errors_mean, pair_class_acc_mean (+ the pairs)
  per_class.csv       experiment, tag, class, class_name, seed-mean test accuracy
  subsets.csv         provenance subsets of the test split (scripts/subset_eval.py): top-1 per head
  subset_comparisons.csv  the paired comparisons on each subset, per seed
  ood.csv, transfer.csv, efficiency.csv, explain.csv, curves.csv (only the runs passed to --curves)
  audit.json          within- and cross-dataset audit numbers
  MANIFEST.txt        SHA-256 and size of every file above
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from dogbreed import paths
from dogbreed.datasets.manifest import load_manifest
from dogbreed.evaluation.runs import apply_class_subset, latest_config
from dogbreed.utils import load_json


def r5(v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return float(f"{v:.5g}")
    if isinstance(v, (list, tuple)):
        return ";".join(str(r5(x)) for x in v)
    return v


def class_names_of(exp_dir: Path) -> list:
    """Breed names in label order for an experiment (its class subset applied), or [] if unknown."""
    try:
        cfg = latest_config(exp_dir)
        df = load_manifest(cfg.get("manifest") or paths.MANIFESTS / f"{cfg['dataset']}.csv")
        return list(apply_class_subset(cfg, df)[1])
    except (SystemExit, FileNotFoundError, KeyError):
        return []


def write_csv(path: Path, rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r5(r.get(k)) for k in fields})
    path.write_bytes(buf.getvalue().encode())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.RESULTS / "digest"))
    ap.add_argument("--curves", default="", help="comma list of <experiment>/<run> directories")
    ap.add_argument("--per-class-tags", default="linear,rr_class,rr_rival,rr_rival_soft,rr_pairwise")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E, R = paths.EXPERIMENTS, paths.RESULTS
    written = []

    # training runs and learning-rate selection
    runs, sel = [], []
    for exp in sorted(p for p in E.iterdir() if p.is_dir()):
        if (exp / "lr_selection.json").exists():
            sel += [{"experiment": exp.name, "tag": k, "lr": v} for k, v in load_json(exp / "lr_selection.json").items()]
        for d in sorted(p for p in exp.iterdir() if p.is_dir()):
            if not (d / "summary.json").exists():
                continue
            s = load_json(d / "summary.json")
            c = s.get("config", {})
            runs.append({"experiment": exp.name, "run": d.name, "tag": c.get("tag"), "seed": c.get("seed"),
                         "lr": c.get("lr"), "head": json.dumps(c.get("head"), sort_keys=True),
                         "loss": c.get("loss"), **{k: s.get(k) for k in (
                             "val_top1", "val_top5", "val_nll", "val_top1_global", "val_cand_recall",
                             "best_epoch", "epochs_run", "params", "train_sec")}})
    if runs:
        write_csv(out / "runs.csv", runs, list(runs[0].keys()))
        written.append("runs.csv")
    if sel:
        write_csv(out / "lr_selection.csv", sel, ["experiment", "tag", "lr"])
        written.append("lr_selection.csv")

    # final evaluation
    test, test_runs, comps, hard, per_class, subsets, sub_cmp = [], [], [], [], [], [], []
    pc_tags = set(a.per_class_tags.split(","))
    for exp in sorted(p for p in R.iterdir() if p.is_dir() and (E / p.name).exists()):
        for view in sorted(p for p in exp.iterdir() if p.is_dir() and (p / "summary.json").exists()):
            for tag, ms in load_json(view / "summary.json").items():
                for m, v in ms.items():
                    test.append({"experiment": exp.name, "view": view.name, "tag": tag, "metric": m, **v})
            if (view / "runs.csv").exists():
                with open(view / "runs.csv") as f:
                    for row in csv.DictReader(f):
                        test_runs.append({"experiment": exp.name, "view": view.name, **row})
            if (view / "comparisons.json").exists():
                for pair, cmp in load_json(view / "comparisons.json").items():
                    for ps in cmp["per_seed"]:
                        comps.append({"experiment": exp.name, "view": view.name, "pair": pair, **ps})
            if (view / "subsets.json").exists():
                sj = load_json(view / "subsets.json")
                for sub, tops in sj["top1"].items():
                    for tag, v in tops.items():
                        subsets.append({"experiment": exp.name, "view": view.name, "subset": sub,
                                        "n_images": sj["n"][sub], "tag": tag, **v})
                for sub, cmp_ in sj["comparisons"].items():
                    for pair, c in cmp_.items():
                        for ps in c["per_seed"]:
                            sub_cmp.append({"experiment": exp.name, "view": view.name, "subset": sub,
                                            "pair": pair, **ps})
            if (view / "hard_pairs.json").exists():
                hp = load_json(view / "hard_pairs.json")
                for tag, v in hp.get("pair_errors_mean", {}).items():
                    hard.append({"experiment": exp.name, "view": view.name, "tag": tag, "pair_errors_mean": v,
                                 "pair_class_acc_mean": hp.get("pair_class_acc_mean", {}).get(tag),
                                 "pairs": " ; ".join(f"{p[0]}~{p[1]}:{p[2]}" for p in hp["pairs"])})
            if view.name == "orig":
                acc = {}
                for d in sorted((E / exp.name).iterdir()):
                    f = d / "test_metrics_orig.json"
                    if f.exists() and "_lr" not in d.name:
                        tm = load_json(f)
                        if tm["tag"] in pc_tags:
                            acc.setdefault(tm["tag"], []).append(tm["per_class_acc"])
                names = class_names_of(E / exp.name) if acc else []
                for tag, lst in acc.items():
                    mean = np.mean(np.asarray(lst, dtype=float), axis=0)
                    per_class += [{"experiment": exp.name, "tag": tag, "class": i,
                                   "class_name": names[i] if i < len(names) else "", "acc": v, "seeds": len(lst)}
                                  for i, v in enumerate(mean)]
    if test:
        write_csv(out / "test.csv", test, ["experiment", "view", "tag", "metric", "mean", "std", "n", "ci95"])
        written.append("test.csv")
    if test_runs:
        write_csv(out / "test_runs.csv", test_runs, list(test_runs[0].keys()))
        written.append("test_runs.csv")
    if comps:
        f = ["experiment", "view", "pair", "seed"] + [k for k in comps[0] if k not in ("experiment", "view", "pair", "seed")]
        write_csv(out / "comparisons.csv", comps, f)
        written.append("comparisons.csv")
    if hard:
        write_csv(out / "hard_pairs.csv", hard, ["experiment", "view", "tag", "pair_errors_mean", "pair_class_acc_mean", "pairs"])
        written.append("hard_pairs.csv")
    if per_class:
        write_csv(out / "per_class.csv", per_class, ["experiment", "tag", "class", "class_name", "acc", "seeds"])
        written.append("per_class.csv")
    if subsets:
        write_csv(out / "subsets.csv", subsets,
                  ["experiment", "view", "subset", "n_images", "tag", "mean", "std", "n", "ci95"])
        written.append("subsets.csv")
    if sub_cmp:
        f = ["experiment", "view", "subset", "pair", "seed"] + [k for k in sub_cmp[0]
                                                              if k not in ("experiment", "view", "subset", "pair", "seed")]
        write_csv(out / "subset_comparisons.csv", sub_cmp, f)
        written.append("subset_comparisons.csv")

    # open set, transfer, efficiency, explanations
    ood = []
    for f in sorted(R.glob("*/ood_*.json")):
        j = load_json(f)
        for tag, ms in j["summary"].items():
            for m, v in ms.items():
                ood.append({"experiment": f.parent.name, "file": f.stem, "tag": tag, "metric": m, **v})
    if ood:
        write_csv(out / "ood.csv", ood, ["experiment", "file", "tag", "metric", "mean", "std", "n", "ci95"])
        written.append("ood.csv")
    tr = []
    for f in sorted(R.glob("*/transfer_stanford.json")):
        j = load_json(f)
        for tag, ms in j["summary"].items():
            for m, v in ms.items():
                tr.append({"experiment": f.parent.name, "tag": tag, "metric": m, "n_images": j["n_images"],
                           "n_breeds": j["n_breeds"], **v})
    if tr:
        write_csv(out / "transfer.csv", tr, ["experiment", "tag", "metric", "mean", "std", "n", "ci95", "n_images", "n_breeds"])
        written.append("transfer.csv")
    eff = []
    for f in sorted((R / "efficiency").glob("*.json")) if (R / "efficiency").exists() else []:
        j = load_json(f)
        eff += [{"experiment": f.stem, "device": j["device"], **r} for r in j["rows"]]
    if eff:
        fields = ["experiment", "device", "part", "name", "batch", "params", "trainable", "gflops",
                  "median_ms", "p90_ms", "warmup", "iters", "peak_gpu_mb"]
        write_csv(out / "efficiency.csv", eff, fields)
        written.append("efficiency.csv")
    ex = []
    for f in sorted(R.glob("*/explain/box_stats.json")):
        j = load_json(f)
        for tag, st in j["stats"].items():
            ex.append({"experiment": f.parent.parent.name, "tag": tag, "top1": st["top1"],
                       "entropy_mean": st["entropy_mean"], "entropy_uniform": st["entropy_uniform"],
                       **{f"{p}_{k}": st[p][k] for p in ("body", "head") for k in st[p]}})
    if ex:
        write_csv(out / "explain.csv", ex, list(ex[0].keys()))
        written.append("explain.csv")
    curves = []
    for spec in [s for s in a.curves.split(",") if s]:
        f = E / spec / "curves.csv"
        if f.exists():
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    curves.append({"run": spec, **{k: row.get(k) for k in ("epoch", "train_loss", "train_top1",
                                                                         "val_top1", "val_nll", "gamma")}})
    if curves:
        write_csv(out / "curves.csv", curves, ["run", "epoch", "train_loss", "train_top1", "val_top1", "val_nll", "gamma"])
        written.append("curves.csv")

    # audits
    aud = {}
    for f in sorted((R / "audit").glob("*.json")) if (R / "audit").exists() else []:
        j = load_json(f)
        if f.name == "cross_tsinghua_stanford.json":
            j = {k: v for k, v in j.items() if k != "breed_correspondence"}
        if f.name == "stanford_clean_test.json":
            j = {"n_ids": len(j["ids"]), "rule": j["rule"]}
        aud[f.stem] = j
    if aud:
        (out / "audit.json").write_bytes(json.dumps(aud, indent=1, sort_keys=True, default=str).encode())
        written.append("audit.json")

    lines = []
    for name in written:
        b = (out / name).read_bytes()
        lines.append(f"{hashlib.sha256(b).hexdigest()}  {len(b):8d}  {name}")
    (out / "MANIFEST.txt").write_bytes(("\n".join(lines) + "\n").encode())
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
