"""Split dist/bundle_cell.txt into Colab cells short enough to type in one go.

    python -I dist/split_cell.py [--max 9000] [--extra "python code appended to the last cell"]

Writes dist/chunks/part_00.txt ... (each writes one slice of the base64 payload to
/content/b/q<i>) and dist/chunks/part_final.txt, which joins the slices, checks their MD5
against the value computed here, and only then unpacks and installs.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=9000)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cell = (ROOT / "dist" / "bundle_cell.txt").read_text()
    m = re.search(r"base64\.b64decode\('([A-Za-z0-9+/=]+)'\)", cell)
    payload = m.group(1)
    parts = [payload[i:i + a.max] for i in range(0, len(payload), a.max)]
    out = ROOT / "dist" / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part_*.txt"):
        old.unlink()
    for i, p in enumerate(parts):
        (out / f"part_{i:02d}.txt").write_bytes(
            f"import os;os.makedirs('/content/b',exist_ok=True);"
            f"open('/content/b/q{i:02d}','w').write('{p}');print('PART {i:02d} OK')".encode())
    md5 = hashlib.md5(payload.encode()).hexdigest()
    join = (f"''.join(open(f'/content/b/q{{i:02d}}').read() for i in range({len(parts)}))")
    final = cell.replace(m.group(0), "base64.b64decode(P)")
    final = (f"import hashlib;P={join};"
             f"assert hashlib.md5(P.encode()).hexdigest()=='{md5}','payload md5 mismatch';" + final)
    if a.extra:
        final += ";" + a.extra
    (out / "part_final.txt").write_bytes(final.encode())
    print(f"{len(parts)} parts of <= {a.max} chars + final ({len(final)} chars); md5 {md5}")


if __name__ == "__main__":
    main()
