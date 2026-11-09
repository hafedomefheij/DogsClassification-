"""Pack the code into one Colab cell (typed into the notebook, so no upload dialog is needed).

    python -I dist/make_bundle.py            # everything -> dist/bundle_cell.txt
    python -I dist/make_bundle.py --changed  # only files changed since the last bundle

The cell unpacks a base64 tar.xz into /content/DogBreed-PhD-Research, installs the package
(editable, no dependency changes) and runs the unit tests. dist/bundle_manifest.json records
the SHA-1 of every file sent, which is what --changed compares against.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "scripts", "configs", "tests", "pyproject.toml"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
MANIFEST = ROOT / "dist" / "bundle_manifest.json"
TEXT = {".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".md", ".txt", ".json"}


def files():
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            yield p
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_PARTS) and f.suffix != ".pyc":
                yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    old = json.loads(MANIFEST.read_text()) if (a.changed and MANIFEST.exists()) else {}
    sent, digests = [], {}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tar:
        for f in files():
            rel = f.relative_to(ROOT).as_posix()
            data = f.read_bytes()
            if f.suffix in TEXT:  # files edited on Windows may carry CRLF; bash rejects it
                data = data.replace(b"\r\n", b"\n")
            h = hashlib.sha1(data).hexdigest()
            digests[rel] = h
            if old.get(rel) == h:
                continue
            info = tar.gettarinfo(str(f), arcname=rel)
            info.size = len(data)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if f.suffix == ".sh" else 0o644
            tar.addfile(info, io.BytesIO(data))
            sent.append(rel)
    b64 = base64.b64encode(buf.getvalue()).decode()
    test = "" if a.no_tests else " && python -m pytest -q tests 2>&1 | tail -2"
    cell = ("import base64,io,tarfile,subprocess;D='/content/DogBreed-PhD-Research';"
            f"tarfile.open(fileobj=io.BytesIO(base64.b64decode('{b64}')),mode='r:xz').extractall(D,filter='data');"
            f"r=subprocess.run(f'cd {{D}} && pip install -q --no-deps -e .{test}',shell=True,capture_output=True,text=True);"
            f"print('BUNDLE OK {len(sent)} files');print(r.stdout[-800:],r.stderr[-800:])")
    out = ROOT / "dist" / "bundle_cell.txt"
    out.write_text(cell)
    MANIFEST.write_text(json.dumps(digests, indent=1))
    print(f"{len(sent)} files, {len(buf.getvalue())/1024:.1f} KiB xz, cell {len(cell)/1024:.1f} KiB -> {out}")
    if len(sent) < 15:
        print("\n".join(sent))


if __name__ == "__main__":
    main()
