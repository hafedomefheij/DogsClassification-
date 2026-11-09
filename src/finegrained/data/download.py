"""Download an archive from its official URL and verify it against the published checksum.

curl does the transfer (resumes after drops); Python computes SHA-256 and MD5 in one pass. A file
whose checksum does not match is deleted and never used. The returned record (url, size, both
hashes, which published value it matched, time) is written next to the file as ``<file>.json``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict

from .specs import Archive


def file_hashes(path: Path, chunk: int = 1 << 22) -> Dict[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def check(path: Path, archive: Archive) -> Dict:
    """Hash ``path`` and compare with every published value of ``archive``."""
    h = file_hashes(path)
    size = path.stat().st_size
    matched, failed = [], []
    for kind in ("sha256", "md5"):
        want = getattr(archive, kind)
        if want:
            (matched if h[kind] == want.lower() else failed).append(kind)
    if archive.size is not None:
        (matched if size == archive.size else failed).append("size")
    return {"file": archive.filename, "url": archive.url, "bytes": size, **h,
            "matched": matched, "failed": failed, "checksum_source": archive.checksum_source,
            "verified": bool(matched) and not failed}


def fetch(archive: Archive, out_dir: str | Path, attempts: int = 20, require_checksum: bool = True) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / archive.filename
    rec_path = out.with_name(out.name + ".json")
    if out.exists() and rec_path.exists():
        rec = json.loads(rec_path.read_text())
        if rec.get("verified") or not require_checksum:
            return rec
    for attempt in range(1, attempts + 1):
        rc = subprocess.call(["curl", "-fL", "--retry", "5", "--retry-delay", "10", "--connect-timeout", "30",
                              "-sS", "-C", "-", "-o", str(out), archive.url])
        if rc == 0:
            break
        if rc == 33:          # the server refused a range request: start again
            out.unlink(missing_ok=True)
        print(f"[{time.strftime('%H:%M:%S')}] {archive.filename}: attempt {attempt}/{attempts} curl exit {rc}",
              flush=True)
        time.sleep(10)
    else:
        raise RuntimeError(f"download failed: {archive.url}")
    rec = check(out, archive)
    rec["downloaded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if require_checksum and not rec["verified"]:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {archive.filename}: {rec}")
    rec_path.write_text(json.dumps(rec, indent=2))
    return rec
