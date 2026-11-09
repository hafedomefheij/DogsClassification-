"""Train one head on cached features: AdamW, warm-up + cosine schedule, early stopping on the
validation split, checkpoint every epoch (resumable), full training curves.

The test split is never touched here: final test evaluation is a separate step
(``scripts/final_eval.py``) run only after model selection.
"""
from __future__ import annotations

import copy
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..losses.losses import compute_loss
from ..models.heads import build_head, count_params
from ..modules.rival_attention import random_token_mask
from ..utils import autocast, code_hash, get_logger, save_json, set_seed
from .cached import SplitData

log = get_logger()

DEFAULTS = {
    "epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.05, "warmup_epochs": 1,
    "label_smoothing": 0.1, "loss": "full", "aux_global": 1.0, "token_drop": 0.1,
    "grad_clip": 1.0, "patience": 8, "random_view": True, "eval_batch_size": 512,
}


def _autocast(device: torch.device):
    # heads train and predict with the same precision: bf16 where native, otherwise fp32
    return autocast(device, training=True)


@torch.no_grad()
def predict(head: torch.nn.Module, data: SplitData, device: torch.device, batch_size: int = 512,
            return_attn: bool = False, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Logits (and stage-1 logits / candidates / attention when the head has them).

    Only the ``random`` re-ranking control draws random numbers at inference (its rival classes). With
    ``seed`` those draws come from a fixed seed in a forked RNG, so an evaluation repeats exactly and the
    global RNG is left as it was. ``seed=None`` (used during training) keeps the global RNG.
    """
    if seed is None:
        return _predict(head, data, device, batch_size, return_attn)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        return _predict(head, data, device, batch_size, return_attn)


@torch.no_grad()
def _predict(head, data, device, batch_size, return_attn):
    head.eval()
    out: Dict[str, List[np.ndarray]] = {}
    for s in range(0, data.n, batch_size):
        idx = np.arange(s, min(s + batch_size, data.n))
        g, tok, _ = data.batch(idx, device)
        with _autocast(device):
            o = head(g, tok, return_attn=return_attn)
        for k in ("logits", "logits_global", "cand", "scores", "attn"):
            if k in o and o[k] is not None:
                v = o[k].float() if o[k].is_floating_point() else o[k]
                out.setdefault(k, []).append(v.cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["ids"], res["y"] = data.ids, data.y.numpy()
    return res


def quick_metrics(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    z, y = torch.from_numpy(pred["logits"]).float(), torch.from_numpy(pred["y"])
    top5 = z.topk(min(5, z.shape[1]), dim=1).indices
    m = {"top1": float((z.argmax(1) == y).float().mean()),
         "top5": float((top5 == y[:, None]).any(1).float().mean()),
         "nll": float(F.cross_entropy(z, y))}
    if "logits_global" in pred:
        z0 = torch.from_numpy(pred["logits_global"]).float()
        m["top1_global"] = float((z0.argmax(1) == y).float().mean())
        p, p0 = z.argmax(1), z0.argmax(1)
        m["flip_rate"] = float((p != p0).float().mean())
        m["fix_rate"] = float(((p == y) & (p0 != y)).float().mean())
        m["break_rate"] = float(((p != y) & (p0 == y)).float().mean())
    if "cand" in pred:
        m["cand_recall"] = float((torch.from_numpy(pred["cand"]) == y[:, None]).any(1).float().mean())
    return m


def _param_groups(head, weight_decay):
    decay, no_decay = [], []
    for n, p in head.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def train_head(cfg: dict, train: SplitData, val: SplitData, out_dir: str | Path,
               device: torch.device, num_classes: int, class_names: Optional[list] = None) -> dict:
    """cfg: DEFAULTS plus 'head' (spec for build_head), 'seed', 'backbone', 'dataset'."""
    cfg = {**DEFAULTS, **cfg}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists():
        log.info("run already finished: %s", out_dir)
        from ..utils import load_json
        return load_json(out_dir / "summary.json")
    set_seed(cfg["seed"])
    head = build_head(cfg["head"], train.dim_global, train.dim_token, num_classes).to(device)
    opt = torch.optim.AdamW(_param_groups(head, cfg["weight_decay"]), lr=cfg["lr"])
    spe = math.ceil(train.n / cfg["batch_size"])
    total, warm = cfg["epochs"] * spe, max(1, int(cfg["warmup_epochs"] * spe))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator().manual_seed(cfg["seed"])
    start, best, best_state, history, bad = 0, None, None, [], 0
    last = out_dir / "last.pt"
    if last.exists():                                           # resume an interrupted run
        ck = torch.load(last, map_location=device, weights_only=False)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"])
        start, best, best_state, history, bad = ck["epoch"] + 1, ck["best"], ck["best_state"], ck["history"], ck["bad"]
        log.info("resumed %s at epoch %d", out_dir.name, start)
    t_run = time.perf_counter()
    for epoch in range(start, cfg["epochs"]):
        head.train()
        t0 = time.perf_counter()
        perm = torch.randperm(train.n, generator=gen).numpy()
        sums = {"loss": 0.0, "correct": 0, "n": 0, "gn": 0.0, "gn_max": 0.0, "steps": 0}
        for step in range(spe):
            idx = perm[step * cfg["batch_size"]:(step + 1) * cfg["batch_size"]]
            g, tok, y = train.batch(idx, device, gen, random_view=cfg["random_view"])
            mask = random_token_mask(len(idx), tok.shape[1], cfg["token_drop"], device)
            with _autocast(device):
                out = head(g, tok, y=y, token_mask=mask)
                loss, _ = compute_loss(out, y, cfg["loss"], cfg["aux_global"], cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["grad_clip"]))
            if not math.isfinite(gn):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch} step {step}")
            opt.step()
            sched.step()
            sums["loss"] += float(loss.detach()) * len(idx)
            sums["correct"] += int((out["logits"].argmax(1) == y).sum())
            sums["n"] += len(idx)
            sums["gn"] += gn
            sums["gn_max"] = max(sums["gn_max"], gn)
            sums["steps"] += 1
        pv = predict(head, val, device, cfg["eval_batch_size"])
        vm = quick_metrics(pv)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": sums["loss"] / sums["n"],
               "train_top1": sums["correct"] / sums["n"], "grad_norm_mean": sums["gn"] / sums["steps"],
               "grad_norm_max": sums["gn_max"], **{f"val_{k}": v for k, v in vm.items()},
               "epoch_sec": time.perf_counter() - t0}
        if device.type == "cuda":
            row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if hasattr(head, "rerank"):
            row["gamma"] = float(head.rerank.gamma)
        history.append(row)
        key = (vm["top1"], -vm["nll"])
        if best is None or key > tuple(best):
            best, best_state, bad = list(key), copy.deepcopy({k: v.cpu() for k, v in head.state_dict().items()}), 0
            best_epoch_preds = pv
            np.savez_compressed(out_dir / "val_preds.npz", **pv)
        else:
            bad += 1
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "epoch": epoch, "best": best, "best_state": best_state,
                    "history": history, "bad": bad}, last)
        log.info("%s ep %02d loss %.3f train %.3f val %.4f (global %.4f) nll %.3f %.1fs",
                 out_dir.name, epoch, row["train_loss"], row["train_top1"], vm["top1"],
                 vm.get("top1_global", float("nan")), vm["nll"], row["epoch_sec"])
        if bad >= cfg["patience"]:
            log.info("early stop at epoch %d", epoch)
            break
    with open(out_dir / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in history for k in r}))
        w.writeheader()
        w.writerows(history)
    best_epoch = int(np.argmax([r["val_top1"] + 1e-9 * -r["val_nll"] for r in history]))
    torch.save({"head_state": best_state, "config": cfg, "num_classes": num_classes,
                "class_names": class_names, "best_epoch": best_epoch,
                "dims": [train.dim_global, train.n_tokens, train.dim_token]}, out_dir / "best.pt")
    hb = history[best_epoch]
    summary = {"run": out_dir.name, "config": cfg, "params": count_params(head),
               "best_epoch": best_epoch, "epochs_run": len(history),
               "val_top1": hb["val_top1"], "val_top5": hb["val_top5"], "val_nll": hb["val_nll"],
               "val_top1_global": hb.get("val_top1_global"), "val_cand_recall": hb.get("val_cand_recall"),
               "train_sec": time.perf_counter() - t_run, "code_hash": code_hash()}
    save_json(summary, out_dir / "summary.json")
    last.unlink(missing_ok=True)                 # the finished run keeps best.pt only
    return summary


def load_head(ckpt_path: str | Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    dg, _, dt = ck["dims"]
    head = build_head(ck["config"]["head"], dg, dt, ck["num_classes"]).to(device)
    head.load_state_dict(ck["head_state"])
    head.eval()
    return head, ck
