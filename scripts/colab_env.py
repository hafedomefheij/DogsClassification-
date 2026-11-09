"""Exec'd inside the Colab kernel at the start of every session:

    exec(open('/content/DogBreed-PhD-Research/scripts/colab_env.py').read())

Large, re-computable things (images, feature caches) live on the fast local disk; everything
worth keeping (dataset archives, manifests, audits, runs, results) is mirrored to Google Drive
by scripts/colab_sync.sh, so a new session restores it in seconds instead of recomputing.
"""
import os

DRIVE = "/content/drive/MyDrive/DogBreed-PhD-Research"
if not os.path.isdir("/content/drive/MyDrive"):
    DRIVE = ""  # Drive not mounted: nothing is mirrored, everything stays on the local disk
WORK = "/content/work"
os.environ.update({
    "DRIVE": DRIVE,
    "WORK": WORK,
    "DOGBREED_RAW": "/content/data/raw",
    "DOGBREED_MANIFESTS": f"{WORK}/manifests",
    "DOGBREED_FEATURES": "/content/features",
    "DOGBREED_EXPERIMENTS": f"{WORK}/experiments",
    "DOGBREED_RESULTS": f"{WORK}/results",
    "DOGBREED_FIGURES": f"{WORK}/figures",
    "DOGBREED_CHECKPOINTS": f"{WORK}/checkpoints",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "PYTHONUNBUFFERED": "1",
    "TQDM_MININTERVAL": "30",
})
for _d in (WORK, f"{WORK}/logs", "/content/data/raw", "/content/features"):
    os.makedirs(_d, exist_ok=True)
if not DRIVE:
    print("note: Google Drive is not mounted; results stay on this runtime's disk only")
print("env ready:", {k: os.environ[k] for k in ("DRIVE", "WORK", "DOGBREED_RAW", "DOGBREED_FEATURES")})
