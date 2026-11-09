"""Stanford Dogs images that are ILSVRC-2012 training images, matched by ImageNet file id (Stanford Dogs keeps the
ImageNet names, e.g. n02085620_5927.jpg). The ILSVRC-2012 train list is train.txt of BVLC Caffe's
caffe_ilsvrc12.tar.gz (dl.caffe.berkeleyvision.org; 1,281,167 lines). Writes results/audit/ilsvrc_overlap.json.

    mkdir -p /content/aux && cd /content/aux
    wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz && tar -xzf caffe_ilsvrc12.tar.gz train.txt synsets.txt
    python -I /path/to/scripts/ilsvrc_overlap.py /content/aux /content/work     # AUX_DIR WORK_DIR
"""
import hashlib
import json
import sys

import pandas as pd

aux, work = sys.argv[1], sys.argv[2]
lines = open(f"{aux}/train.txt").read().split("\n")
tr = {l.split()[0].split("/")[1].rsplit(".", 1)[0] for l in lines if l.strip()}
syn = {l.strip() for l in open(f"{aux}/synsets.txt") if l.strip()}
s = pd.read_csv(f"{work}/manifests/stanford.csv")
s["stem"] = s.key.str.split("/").str[-1].str.rsplit(".", n=1).str[0]
s["in_train"] = s.stem.isin(tr)
cls = s.drop_duplicates("class_key")
out = {"ilsvrc_train_list": "train.txt in caffe_ilsvrc12.tar.gz (BVLC Caffe, dl.caffe.berkeleyvision.org)",
       "ilsvrc_train_list_sha256": hashlib.sha256(open(f"{aux}/train.txt", "rb").read()).hexdigest(),
       "ilsvrc_train_images": len(tr), "stanford_classes": int(len(cls)),
       "stanford_classes_in_ilsvrc1k": int(cls.class_key.isin(syn).sum())}
for sp, g in s.groupby("official_split"):
    out[f"stanford_{sp}"] = {"images": int(len(g)), "in_ilsvrc2012_train": int(g.in_train.sum())}
for sp, g in s.groupby("split"):
    out[f"stanford_ourspl_{sp}"] = {"images": int(len(g)), "in_ilsvrc2012_train": int(g.in_train.sum())}
cp = pd.read_csv(f"{work}/results/audit/cross_pairs.csv")
conf = cp[cp.confirmed.astype(str).str.lower().isin(["true", "1"])].copy()
stem = dict(zip(s.id, s.stem))
conf["s_in"] = conf.s_id.map(lambda i: stem.get(i) in tr)
out["tsinghua_split_values"] = sorted(map(str, cp.t_split.unique()))
for sp, g in conf.groupby("t_split"):
    a = g.groupby("t_id").s_in.any()
    out[f"tsinghua_{sp}_confirmed_copy"] = {"images": int(len(a)), "copy_in_ilsvrc2012_train": int(a.sum())}
json.dump(out, open(f"{work}/results/audit/ilsvrc_overlap.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "ilsvrc_train_list_sha256"}))
