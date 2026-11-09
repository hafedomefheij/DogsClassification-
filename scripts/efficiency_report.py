"""Parameters, FLOPs, latency and peak GPU memory: every head, the backbone, the full pipeline.

    python -I scripts/efficiency_report.py --config configs/experiments/tsinghua_main.yaml

* heads alone, on random inputs of the cached shapes (g: D_g, tokens: N x token_dims), fp32 as
  trained; batch 1 (one image) and batch 256 (throughput);
* the frozen backbone on a random 224 px batch, fp16 autocast as used for extraction;
* the full single-image pipeline: backbone -> token PCA (the train-fitted projection) -> head.
FLOPs: torch.utils.flop_counter (matmul / conv / attention operators; one multiply-add = 2 FLOPs).
Latency: median of timed passes after warm-up, on this runtime's device (its name is recorded).
Output: results/efficiency/<experiment>.json and .csv.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import yaml

from dogbreed import paths
from dogbreed.datasets.manifest import class_names, load_manifest
from dogbreed.evaluation import efficiency as E
from dogbreed.evaluation.runs import apply_class_subset
from dogbreed.models.backbones import Backbone
from dogbreed.models.heads import build_head
from dogbreed.utils import auto_device, autocast, env_info, get_logger, save_json

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    dev = auto_device(a.device)
    df, names = apply_class_subset(cfg, load_manifest(paths.MANIFESTS / f"{cfg['dataset']}.csv"))
    C = len(names)
    bb = Backbone(cfg["backbone"], cfg.get("img_size", 224)).eval().to(dev)
    bb.requires_grad_(False)
    pca_file = paths.feature_root(cfg["dataset"], cfg["backbone"], cfg.get("img_size", 224),
                                  cfg.get("feature_tag", "")) / "pca.npz"
    td = cfg.get("token_dims")
    if pca_file.exists():
        z = np.load(pca_file)
        td = td or int(z["components"].shape[0])
        p_mean = torch.from_numpy(z["mean"]).to(dev)
        p_comp = torch.from_numpy(z["components"][:td]).to(dev)
    else:
        td, p_mean, p_comp = bb.dim_token, None, None
    dg, n = bb.dim_global, bb.n_tokens
    log.info("backbone %s: D_g %d, N %d tokens, token dims %d, C %d", cfg["backbone"], dg, n, td, C)
    rows = []
    amp = autocast(dev, training=False) if dev.type == "cuda" else nullcontext()

    # backbone alone
    for bs in (1, 64):
        x = torch.randn(bs, 3, cfg.get("img_size", 224), cfg.get("img_size", 224), device=dev)

        def fwd():
            with amp:
                return bb(x)
        r = {"part": "backbone", "name": cfg["backbone"], "batch": bs, **E.params(bb)}
        if bs == 1:
            r["gflops"] = E.flops(fwd) / 1e9
        r.update(E.latency(fwd, dev, warmup=10, iters=a.iters))
        r.update(E.peak_memory(fwd, dev))
        rows.append(r)

    seen = set()
    heads = []
    for spec in cfg["runs"]:
        key = str(sorted(spec["head"].items()))
        if key in seen:
            continue
        seen.add(key)
        h = build_head(dict(spec["head"]), dg, td, C).eval().to(dev)
        n_params = E.params(h)
        # frozen for measuring: under no_grad a view of a trainable parameter still reports
        # requires_grad without a grad_fn, which breaks the FLOP counter's module tracker
        h.requires_grad_(False)
        heads.append((spec["tag"], h))
        for bs in (1, 256):
            g = torch.randn(bs, dg, device=dev)
            t = torch.randn(bs, n, td, device=dev)
            fwd = (lambda h=h, g=g, t=t: h(g, t))
            r = {"part": "head", "name": spec["tag"], "batch": bs, **n_params}
            if bs == 1:
                r["gflops"] = E.flops(fwd) / 1e9
            r.update(E.latency(fwd, dev, warmup=10, iters=a.iters))
            r.update(E.peak_memory(fwd, dev))
            rows.append(r)

    # full pipeline for one image: backbone -> PCA -> head
    x1 = torch.randn(1, 3, cfg.get("img_size", 224), cfg.get("img_size", 224), device=dev)
    for tag, h in heads:
        def pipe(h=h):
            with amp:
                g, tok = bb(x1)
            tok = tok.float()
            if p_comp is not None:
                tok = (tok - p_mean) @ p_comp.T
            return h(g.float(), tok)
        r = {"part": "pipeline", "name": tag, "batch": 1, "gflops": E.flops(pipe) / 1e9,
             "params": E.params(bb)["params"] + E.params(h)["params"]}
        r.update(E.latency(pipe, dev, warmup=10, iters=a.iters))
        r.update(E.peak_memory(pipe, dev))
        rows.append(r)

    out = paths.RESULTS / "efficiency"
    out.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out / f"{cfg['name']}.csv", index=False)
    info = env_info()
    save_json({"config": a.config, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               "num_classes": C, "token_dims": td, "n_tokens": n, "dim_global": dg, "env": info,
               "rows": rows}, out / f"{cfg['name']}.json")
    log.info("\n%s", df_out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
