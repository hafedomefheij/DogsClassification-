"""The four benchmarks: official sources, checksums, licences, and how their splits are used.

Checksums and their sources were checked on 2026-11-09:
* CUB-200-2011: SHA-256 and size from TensorFlow Datasets' ``url_checksums/caltech_birds2011.txt``
  (the same value appears in an independent Git LFS copy); MD5 from the Caltech DATA record
  65de6-vp158 (DOI 10.22002/D1.20098).
* FGVC-Aircraft 2013b: MD5 from the official page ``archives/fgvc-aircraft-2013b.html``.
* Stanford Dogs: SHA-256 from TensorFlow Datasets (Study 1, ``scripts/download_data.sh``).
* Tsinghua Dogs: no published checksum; Study 1 recorded the SHA-256 of the copies it used
  (``configs/expected_sha256_tsinghua.txt``).
None of these datasets may be redistributed: only manifests (ids, paths, labels, hashes) are kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Archive:
    filename: str
    url: str
    sha256: Optional[str] = None
    md5: Optional[str] = None
    size: Optional[int] = None
    checksum_source: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    n_classes: int
    archives: Tuple[Archive, ...]
    homepage: str
    licence: str
    citation: str
    crop_bottom_px: int = 0          # FGVC-Aircraft: a 20-pixel copyright banner the authors ask to remove
    val_from_train: Optional[float] = None   # fraction of the official train split held out (None: official val)
    val_seed: int = 0
    notes: str = ""


CUB = DatasetSpec(
    name="cub", title="CUB-200-2011", n_classes=200,
    archives=(Archive(
        "CUB_200_2011.tgz",
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        sha256="0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081",
        md5="97eceeb196236b17998738112f37df78", size=1150585339,
        checksum_source="sha256+size: tensorflow/datasets url_checksums/caltech_birds2011.txt; "
                        "md5: Caltech DATA record 65de6-vp158"),),
    homepage="https://www.vision.caltech.edu/datasets/cub_200_2011/",
    licence="Caltech DATA lists CC-BY for the record; images are Flickr photographs (research use)",
    citation="Wah, Branson, Welinder, Perona, Belongie. The Caltech-UCSD Birds-200-2011 Dataset. "
             "Caltech CNS-TR-2011-001, 2011.",
    val_from_train=0.15, val_seed=0,
    notes="5,994 train / 5,794 test images; no official validation split, so 15% of each class's "
          "training images are held out with a fixed seed.")

AIRCRAFT = DatasetSpec(
    name="aircraft", title="FGVC-Aircraft 2013b (variant level)", n_classes=100,
    archives=(Archive(
        "fgvc-aircraft-2013b.tar.gz",
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
        md5="32eca553f897e747706144f3bb444df0",
        checksum_source="md5: official page archives/fgvc-aircraft-2013b.html"),),
    homepage="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/",
    licence="non-commercial research purposes only; photographers keep the copyright",
    citation="Maji, Kannala, Rahtu, Blaschko, Vedaldi. Fine-Grained Visual Classification of Aircraft. "
             "arXiv:1306.5151, 2013.",
    crop_bottom_px=20,
    notes="Official train / val / test (3,334 / 3,333 / 3,333) at the variant level (100 classes); "
          "family (70) and manufacturer (41) labels kept for analysis. The bottom 20 pixels (copyright "
          "banner) are removed, as the dataset page asks.")

STANFORD = DatasetSpec(
    name="stanford", title="Stanford Dogs", n_classes=120,
    archives=(
        Archive("images.tar", "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
                sha256="a443fc2f9a851aa1a7b8e4fa4af94b8459fce2c9f5a2f4dd4669cf9673a4ec58",
                checksum_source="tensorflow/datasets stanford_dogs checksums (Study 1)"),
        Archive("lists.tar", "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
                sha256="34b47cacd9a98b5d150e084f24d29391c084c55272295ec65c85651bc35f4d6c",
                checksum_source="tensorflow/datasets stanford_dogs checksums (Study 1)")),
    homepage="https://vision.stanford.edu/aditya86/ImageNetDogs/",
    licence="images from ImageNet: non-commercial research use",
    citation="Khosla, Jayadevaprakash, Yao, Fei-Fei. Novel Dataset for Fine-Grained Image Categorization. "
             "FGVC workshop, CVPR 2011.",
    val_from_train=0.1, val_seed=0,
    notes="Study 1's manifest and validation split are reused unchanged. Every image is in ILSVRC-2012 "
          "train (Study 1 audit), so supervised-ImageNet backbones have seen the test images.")

TSINGHUA = DatasetSpec(
    name="tsinghua", title="Tsinghua Dogs (low-resolution release)", n_classes=130,
    archives=(
        Archive("low-resolution.zip", "https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1",
                checksum_source="SHA-256 recorded by Study 1: configs/expected_sha256_tsinghua.txt"),
        Archive("TrainValSplit.zip", "https://cg.cs.tsinghua.edu.cn/ThuDogs/TrainValSplit.zip",
                checksum_source="SHA-256 recorded by Study 1"),
        Archive("low-annotations.zip", "https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip",
                checksum_source="SHA-256 recorded by Study 1")),
    homepage="https://cg.cs.tsinghua.edu.cn/ThuDogs/",
    licence="research use (see the dataset page)",
    citation="Zou, Wang, Chen, Hu, Chen. A new dataset of dog breed images and a benchmark for fine-grained "
             "classification. Computational Visual Media 6(4), 2020.",
    val_from_train=0.1, val_seed=0,
    notes="Study 1's manifest and validation split are reused unchanged; 3,115 of 5,200 test images are "
          "copies of Stanford Dogs photographs (Study 1 audit).")

DATASETS = {d.name: d for d in (CUB, AIRCRAFT, STANFORD, TSINGHUA)}
