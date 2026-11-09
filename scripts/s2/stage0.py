"""Study 2, stage 0 (CPU only): download, verify, list and audit CUB-200-2011 and FGVC-Aircraft.

    python -I scripts/s2/stage0.py --datasets cub aircraft --raw /kaggle/tmp/raw --out /kaggle/working/s2/stage0

Per dataset, in ``--out/<dataset>/``: the download records (``<archive>.json``: URL, bytes, SHA-256,
MD5, which published checksum matched), ``manifest_raw.csv`` (official splits, no validation split
yet), ``hashes.csv``, ``phash_pairs.csv`` (near-duplicate candidates), ``audit.json`` and montages of
the closest cross-split candidates for a visual check. Nothing is trained and no exclusion is
decided here: near duplicates are confirmed with frozen embeddings in stage 1.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from dogbreed.datasets.sources import ImageSource
from finegrained.data import audit
from finegrained.data.download import fetch
from finegrained.data.images import CroppedSource
from finegrained.data.manifests import aircraft_raw, cub_raw, manifest_sha256
from finegrained.data.specs import DATASETS

REPO = Path(__file__).resolve().parents[2]
BUILDERS = {"cub": cub_raw, "aircraft": aircraft_raw}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def safe_extract(archive: Path, dest: Path) -> None:
    marker = dest / ".extracted"
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        try:
            tf.extractall(dest, filter="data")          # refuses absolute paths, links out of dest
        except TypeError:                               # Python without extraction filters
            root = dest.resolve()
            for m in tf.getmembers():
                target = (dest / m.name).resolve()
                if root not in target.parents and target != root or m.issym() or m.islnk():
                    raise RuntimeError(f"unsafe member in {archive.name}: {m.name}")
            tf.extractall(dest)
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def montage(pairs: pd.DataFrame, df: pd.DataFrame, source, path: Path, n: int = 24, thumb: int = 150) -> int:
    sel = pairs.head(n)
    if not len(sel):
        return 0
    by_id = df.set_index("id")
    per_row, cap = 3, 30
    rows = (len(sel) + per_row - 1) // per_row
    cell_w, cell_h = 2 * thumb + 16, thumb + cap
    canvas = Image.new("RGB", (per_row * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for k, r in enumerate(sel.itertuples()):
        x0, y0 = (k % per_row) * cell_w, (k // per_row) * cell_h
        for side, iid in enumerate((r.id_a, r.id_b)):
            try:
                im = source.open(by_id.loc[iid, "key"])
                im.thumbnail((thumb, thumb))
                canvas.paste(im, (x0 + side * (thumb + 4), y0))
            except Exception as exc:  # shown in the caption instead
                draw.text((x0 + side * (thumb + 4), y0 + 4), type(exc).__name__, fill="red")
        draw.text((x0, y0 + thumb + 2), f"{r.id_a} [{r.split_a}] vs {r.id_b} [{r.split_b}]", fill="black")
        draw.text((x0, y0 + thumb + 15), f"labels {r.label_a}/{r.label_b}  pH {r.phash_d} dH {r.dhash_d}",
                  fill="black")
    canvas.save(path)
    return len(sel)


def run(name: str, raw: Path, out: Path, workers: int) -> dict:
    spec = DATASETS[name]
    d_raw, d_out = raw / name, out / name
    d_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records = []
    for a in spec.archives:
        log(f"{name}: fetching {a.url}")
        rec = fetch(a, d_raw)
        (d_out / f"{a.filename}.json").write_text(json.dumps(rec, indent=2))
        log(f"{name}: {a.filename} {rec['bytes']:,} bytes; matched {rec['matched']} (sha256 {rec['sha256'][:16]}...)")
        records.append(rec)
        safe_extract(d_raw / a.filename, d_raw / "x")
    df, image_root = BUILDERS[name](d_raw / "x")
    df.to_csv(d_out / "manifest_raw.csv", index=False)
    source = CroppedSource(ImageSource(image_root), bottom=spec.crop_bottom_px)
    log(f"{name}: {len(df)} images, {df['label'].nunique()} classes; hashing with {workers} threads")
    h = audit.hash_table(df, source, workers=workers)
    h.to_csv(d_out / "hashes.csv", index=False)
    pairs = audit.candidate_pairs(df, h, max_phash=8)
    pairs.to_csv(d_out / "phash_pairs.csv", index=False)
    rep = audit.summarise(df, h, pairs)
    cross = pairs[pairs["split_a"] != pairs["split_b"]] if len(pairs) else pairs
    test_side = cross[(cross["split_a"] == "test") | (cross["split_b"] == "test")] if len(cross) else cross
    rep["montages"] = {"cross_split_to_test": montage(test_side, df, source, d_out / "montage_to_test.png"),
                       "within_split": montage(pairs[pairs["split_a"] == pairs["split_b"]] if len(pairs) else pairs,
                                               df, source, d_out / "montage_within.png")}
    rep.update({"dataset": name, "title": spec.title, "crop_bottom_px": spec.crop_bottom_px,
                "manifest_raw_sha256": manifest_sha256(df), "downloads": records,
                "git_commit": git_commit(), "seconds": round(time.time() - t0, 1)})
    (d_out / "audit.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cub", "aircraft"], choices=sorted(BUILDERS))
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2 * (os.cpu_count() or 2))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
           "pandas": pd.__version__, "pillow": Image.__version__, "git_commit": git_commit(),
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpus": os.cpu_count()}
    (args.out / "env.json").write_text(json.dumps(env, indent=2))
    log(f"stage 0 at commit {env['git_commit']}")
    status = 0
    for name in args.datasets:
        try:
            rep = run(name, args.raw, args.out, args.workers)
            short = {k: rep[k] for k in ("images", "decodable", "per_split", "classes", "exact_file",
                                         "exact_pixels", "phash_le4_groups", "manifest_raw_sha256", "seconds")}
            log(f"{name} SUMMARY {json.dumps(short)}")
            for row in rep["phash_candidates"]:
                if row["pairs"] and row["phash_le"] in (0, 4, 8):
                    log(f"{name} pHash<={row['phash_le']} {row['splits']}: {row['pairs']} pairs "
                        f"({row['same_label']} same label, {row['also_dhash_le8']} also dHash<=8)")
        except Exception as exc:
            status = 1
            log(f"{name} FAILED: {type(exc).__name__}: {exc}")
    log("STAGE 0 DONE" if status == 0 else "STAGE 0 FAILED")
    return status


if __name__ == "__main__":
    sys.exit(main())
