"""Bring the checkpoints and prediction arrays of a Kaggle full archive into the work folder and the Drive mirror
when the archive is not the one whose SHA-256 the session wrote, but each of its files can be checked against that
session's own records (here: a second run of the same notebook, from the same inputs).

    python -I scripts/merge_rerun_files.py --archive kaggle_results_full.tar --work /content/work \
        --mirror /content/drive/MyDrive/DogBreed-PhD-Research/work \
        --record /content/drive/MyDrive/DogBreed-PhD-Research/kaggle_runs/<stamp> [--check-only]

Run it after scripts/ingest_kaggle.sh has merged the session's small archive, which is checked against the SHA-256
the session wrote: every run folder then holds that session's records (freeze.json, summary.json,
test_metrics_<view>.json), identical in the work folder and on the mirror. For every best.pt, val_preds.npz and
test_<view>.npz in the archive:
* best.pt must have the SHA-256 its experiment's freeze.json recorded for the run. A learning-rate grid run at the
  chosen rate must have the hash of the seed-0 run copied from it. A grid run at another rate has no recorded hash
  and is taken only if its validation predictions pass the next check;
* val_preds.npz must give the validation metrics of the run's summary.json, as the training code computes them:
  top-1, top-5, global top-1 and candidate recall exactly, NLL to 1e-6 (float32, recomputed here on any device);
* test_<view>.npz, with the run's val_preds.npz for the temperature, must give the report of
  test_metrics_<view>.json as the evaluation code computes it: every count-based metric exactly, every other
  number to 1e-9. Only the extra random draws of the random-rival control (top1_draws, top1_draw_sd) need a forward
  pass and are not checked;
* the run folder must exist in both destinations; a file already there must be identical.
Every file is checked before the first one is written; one failure and nothing is written. Nothing is overwritten
or deleted. The manifest (path, bytes, SHA-256, what it was checked against) goes to --record and to the logs of
both work folders. Nothing here trains or runs a network.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import socket
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dogbreed.evaluation import metrics as M

KINDS = re.compile(r"^(best\.pt|val_preds\.npz|test_[a-z0-9]+\.npz)$")
GRID = re.compile(r"^(?P<tag>.+)_lr(?P<lr>[0-9.e-]+)_s0$")
COUNTS = {"n", "top1", "top5", "top1_global", "top5_global", "cand_recall", "flip_rate", "fix_rate", "break_rate",
          "macro_precision", "macro_recall", "macro_f1", "mean_per_class_acc", "per_class_acc", "risk_at_90cov",
          "val_top1", "val_top5", "val_top1_global", "val_cand_recall"}
NOT_CHECKED = {"top1_draws", "top1_draw_sd"}
VAL_CORE, TEST_CORE = ("val_top1", "val_top5", "val_nll"), ("top1", "top5", "nll", "per_class_acc", "ece")
VAL_TOL, TEST_TOL = 1e-6, 1e-9


def say(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def val_metrics(p) -> dict:
    """The validation metrics of training.train_heads.quick_metrics, under summary.json's names."""
    z, y = torch.from_numpy(p["logits"]).float(), torch.from_numpy(p["y"])
    top5 = z.topk(min(5, z.shape[1]), dim=1).indices
    m = {"val_top1": float((z.argmax(1) == y).float().mean()),
         "val_top5": float((top5 == y[:, None]).any(1).float().mean()),
         "val_nll": float(F.cross_entropy(z, y))}
    if "logits_global" in p:
        m["val_top1_global"] = float((torch.from_numpy(p["logits_global"]).float().argmax(1) == y).float().mean())
    if "cand" in p:
        m["val_cand_recall"] = float((torch.from_numpy(p["cand"]) == y[:, None]).any(1).float().mean())
    return m


def test_report(p, val, num_classes: int) -> dict:
    """The report of scripts/final_eval.py for one run and view, from its saved arrays."""
    t = M.fit_temperature(val["logits"], val["y"])
    rep = M.full_report(p["logits"], p["y"], num_classes, t=t)
    if "logits_global" in p:
        rep["top1_global"] = M.topk_acc(p["logits_global"], p["y"], 1)
        rep["top5_global"] = M.topk_acc(p["logits_global"], p["y"], 5)
        rep.update(M.rerank_effect(p["logits"], p["logits_global"], p["y"]))
    if "cand" in p:
        rep["cand_recall"] = float((p["cand"] == p["y"][:, None]).any(1).mean())
    return rep


def compare(mine: dict, ref: dict, core, tol: float):
    """Largest deviation of the non-count numbers, and a list of failures (empty when the arrays reproduce ref)."""
    bad, worst = [], 0.0
    for k in core:
        if ref.get(k) is None:
            bad.append(f"record has no {k}")
    for k, v in mine.items():
        r = ref.get(k)
        if k in NOT_CHECKED or v is None or r is None or isinstance(r, str):
            continue
        a, b = np.asarray(v, dtype=float), np.asarray(r, dtype=float)
        if a.shape != b.shape:
            bad.append(f"{k}: shape {a.shape} vs {b.shape}")
            continue
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            bad.append(f"{k}: NaN pattern differs")
            continue
        d = float(np.nanmax(np.abs(a - b))) if a.size and not np.isnan(a).all() else 0.0
        if k in COUNTS:
            if d != 0.0:
                bad.append(f"{k}: {v} vs {r}")
        else:
            worst = max(worst, d)
            if d > tol:
                bad.append(f"{k}: differs by {d:.3g}")
    return worst, bad


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_record(rel: str, work: Path, mirror: Path, problems: list):
    """The record must be in both work folders, byte for byte; returns its parsed content (or None)."""
    a, b = work / rel, mirror / rel
    if not a.is_file() or not b.is_file():
        problems.append(f"{rel}: record missing in {'work' if not a.is_file() else 'mirror'}")
        return None
    if a.read_bytes() != b.read_bytes():
        problems.append(f"{rel}: record differs between work and mirror")
        return None
    return load_json(a)


def check(archive: Path, work: Path, mirror: Path):
    """Phase 1: decide every file, write nothing. Returns (rows, problems, members)."""
    rows, problems, freezes, members = [], [], {}, {}
    with tarfile.open(archive, "r:") as tar:
        by_run = {}
        for m in tar.getmembers():
            parts = m.name.split("/")
            if m.isfile() and len(parts) == 4 and parts[0] == "experiments" and KINDS.match(parts[3]):
                by_run.setdefault((parts[1], parts[2]), {})[parts[3]] = m
        for (exp, run), files in sorted(by_run.items()):
            run_rel = f"experiments/{exp}/{run}"
            if exp not in freezes:
                freezes[exp] = same_record(f"experiments/{exp}/freeze.json", work, mirror, problems)
            fz = (freezes[exp] or {}).get("runs", {})
            summ = same_record(f"{run_rel}/summary.json", work, mirror, problems)
            for dest in (work, mirror):
                if not (dest / run_rel).is_dir():
                    problems.append(f"{run_rel}: no such run folder in {dest}")
            arrays = {n: tar.extractfile(m).read() for n, m in files.items()}
            val = dict(np.load(io.BytesIO(arrays["val_preds.npz"]))) if "val_preds.npz" in arrays else None
            val_ok = None
            if val is not None and summ is not None:
                worst, bad = compare(val_metrics(val), summ, VAL_CORE, VAL_TOL)
                val_ok = not bad
                rows.append([f"{run_rel}/val_preds.npz", "summary.json", f"{worst:.2g}", bad])
            for name, data in sorted(arrays.items()):
                rel = f"{run_rel}/{name}"
                h = sha(data)
                members[rel] = (files[name], h, len(data))
                for dest in (work, mirror):
                    if (dest / rel).exists() and sha_file(dest / rel) != h:
                        problems.append(f"{rel}: a different file is already in {dest}")
                if name == "best.pt":
                    g = GRID.match(run)
                    if run in fz:
                        ok = fz[run].get("sha256_best_pt") == h
                        rows.append([rel, "freeze.json", "", [] if ok else ["SHA-256 differs from freeze.json"]])
                    elif g and f"{g['tag']}_s0" in fz and fz[f"{g['tag']}_s0"].get("sha256_best_pt") == h:
                        rows.append([rel, f"freeze.json ({g['tag']}_s0)", "", []])
                    elif g:
                        rows.append([rel, "no recorded hash", "",
                                     [] if val_ok else ["no recorded hash and validation check not passed"]])
                    else:
                        rows.append([rel, "freeze.json", "", ["a non-grid run missing from freeze.json"]])
                elif name.startswith("test_"):
                    view = name[len("test_"):-len(".npz")]
                    ref = same_record(f"{run_rel}/test_metrics_{view}.json", work, mirror, problems)
                    if ref is None or val is None or not ref.get("per_class_acc"):
                        rows.append([rel, f"test_metrics_{view}.json", "", ["record or val_preds.npz missing"]])
                        continue
                    p = dict(np.load(io.BytesIO(data)))
                    worst, bad = compare(test_report(p, val, len(ref["per_class_acc"])), ref, TEST_CORE, TEST_TOL)
                    rows.append([rel, f"test_metrics_{view}.json", f"{worst:.2g}", bad])
    problems += [f"{r[0]}: {'; '.join(r[3])}" for r in rows if r[3]]
    return rows, problems, members


def write_atomic(path: Path, data: bytes, mtime: float):
    tmp = path.with_name(path.name + ".merge_tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.utime(tmp, (mtime, mtime))
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--mirror", required=True, type=Path)
    ap.add_argument("--record", type=Path, help="folder for the manifest (required unless --check-only)")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.check_only and a.record is None:
        ap.error("--record is required unless --check-only")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archive_sha = sha_file(a.archive)
    say(f"archive {a.archive} ({a.archive.stat().st_size} bytes, sha256 {archive_sha})")
    rows, problems, members = check(a.archive, a.work, a.mirror)
    bases = {}
    for r in rows:
        key = "freeze.json (seed-0 copy)" if r[1].startswith("freeze.json (") else r[1]
        bases[key] = bases.get(key, 0) + 1
    say(f"checked {len(members)} files against: " + "; ".join(f"{k} {v}" for k, v in sorted(bases.items())))
    if problems:
        for p in problems[:50]:
            say(f"PROBLEM {p}")
        say(f"{len(problems)} problems; nothing written")
        return 1
    if a.check_only:
        say("every file passed; --check-only, nothing written")
        return 0
    basis = {r[0]: (r[1], r[2]) for r in rows}
    unchecked = sorted(set(members) - set(basis))
    if unchecked:
        say(f"PROBLEM {len(unchecked)} files have no check, e.g. {unchecked[0]}; nothing written")
        return 1
    written = present = 0
    with tarfile.open(a.archive, "r:") as tar:
        for rel, (m, h, n) in sorted(members.items()):
            data = tar.extractfile(m).read()
            if sha(data) != h:
                say(f"PROBLEM {rel} changed between the check and the copy; stopping")
                return 1
            for dest in (a.work, a.mirror):
                target = dest / rel
                if target.exists():
                    present += 1
                    continue
                write_atomic(target, data, float(m.mtime))
                if sha_file(target) != h:
                    say(f"PROBLEM {target} reads back with a different SHA-256; stopping")
                    return 1
                written += 1
    head = ["path", "bytes", "sha256", "checked against", "largest deviation"]
    lines = ["\t".join(head)] + ["\t".join([rel, str(n), h, *basis[rel]]) for rel, (m, h, n) in sorted(members.items())]
    meta = {"stamp": stamp, "host": socket.gethostname(), "archive": str(a.archive), "archive_bytes":
            a.archive.stat().st_size, "archive_sha256": archive_sha, "files": len(members),
            "written": written, "already_present": present, "checked_by": dict(sorted(bases.items())),
            "work": str(a.work), "mirror": str(a.mirror)}
    a.record.mkdir(parents=True, exist_ok=True)
    for folder in (a.record, a.work / "logs", a.mirror / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"merge_rerun_{stamp}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / f"merge_rerun_{stamp}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    say(f"written {written} file copies, {present} already present; manifest merge_rerun_{stamp}.tsv in {a.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
