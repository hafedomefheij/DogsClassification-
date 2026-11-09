"""Paper tables, hypothesis verdicts and figure data, built from the result digest only.

    python -I scripts/make_tables.py --digest results/digest --out paper/generated

Input: the files scripts/digest.py writes (nothing else is read, so the same command gives the same tables
on the runtime and on any copy of the digest). Output, in --out:
  tab_*.tex      one booktabs tabular per table, \\input by paper/main.tex
  numbers.json   the numbers the text quotes, and H1-H7 judged by the rule fixed in docs/design.md §8
  figdata.json   the compact series scripts/make_figures.py draws
A table whose inputs are missing is skipped and named in numbers.json["missing"].

Decision rule (design.md §8): "A > B" needs the per-seed test top-1 difference to be positive in all 3 seeds
and its paired-bootstrap 95 % interval to exclude 0 in at least 2 of them. Where a hypothesis speaks of the
"best" proposed head or control, the best is chosen on VALIDATION top-1 (mean over seeds), so no test result
takes part in choosing what is compared.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["linear", "attnprobe", "classattn_all"]
CONTROLS = ["rr_class", "rr_center", "rr_random"]
PROPOSED = ["rr_rival", "rr_rival_soft", "rr_pairwise"]
ALL = BASE + CONTROLS + PROPOSED
NAME = {"linear": "Linear probe (B0)", "attnprobe": "Attentive probe (B1)",
        "classattn_all": "Class attention, all classes (B2)", "rr_class": "Re-rank: class query",
        "rr_center": "Re-rank: centred query", "rr_random": "Re-rank: random rivals",
        "rr_rival": "RCEA: rival", "rr_rival_soft": "RCEA: rival-soft", "rr_pairwise": "RCEA: pairwise"}
AUTO = ["orig", "blur", "noise", "jpeg", "lowres", "occlude"]
ANN = ["head_occlude", "bg_gray", "bg_blur", "crop_body", "crop_head"]
VIEW = {"orig": "clean", "blur": "blur", "noise": "noise", "jpeg": "JPEG", "lowres": "low-res",
        "occlude": "occlude", "head_occlude": "head occl.", "bg_gray": "bg grey", "bg_blur": "bg blur",
        "crop_body": "body crop", "crop_head": "head crop"}
BACKBONES = [("tsinghua_main", "DINOv2 ViT-B/14 (reg.)", "self-sup., LVD-142M"),
             ("tsinghua_clip_vitb16", "CLIP ViT-B/16", "image-text, WIT-400M"),
             ("tsinghua_convnext_tiny", "ConvNeXt-T", "supervised IN-1k"),
             ("tsinghua_resnet50", "ResNet-50", "supervised IN-1k")]


# ---------------------------------------------------------------------------------------------- inputs
class Digest:
    def __init__(self, d: Path):
        self.d = d
        self.missing = []
        rd = lambda n: pd.read_csv(d / n) if (d / n).exists() else None  # noqa: E731
        self.runs, self.sel, self.test = rd("runs.csv"), rd("lr_selection.csv"), rd("test.csv")
        self.cmp, self.hard, self.pc = rd("comparisons.csv"), rd("hard_pairs.csv"), rd("per_class.csv")
        self.ood, self.tr, self.eff = rd("ood.csv"), rd("transfer.csv"), rd("efficiency.csv")
        self.ex, self.curves = rd("explain.csv"), rd("curves.csv")
        self.subs, self.subcmp = rd("subsets.csv"), rd("subset_comparisons.csv")
        self.audit = json.loads((d / "audit.json").read_text()) if (d / "audit.json").exists() else {}

    def sub(self, exp, subset, tag, view="orig"):
        """(mean, std, n) of test top-1 on a provenance subset, or None; and the subset's image count."""
        if self.subs is None:
            return None
        m = self.subs[(self.subs.experiment == exp) & (self.subs.view == view) & (self.subs.subset == subset)
                      & (self.subs.tag == tag)]
        if m.empty:
            return None
        r = m.iloc[0]
        return float(r["mean"]), float(r["std"]) if not pd.isna(r["std"]) else 0.0, int(r["n"])

    def sub_n(self, exp, subset, view="orig"):
        if self.subs is None:
            return None
        m = self.subs[(self.subs.experiment == exp) & (self.subs.view == view) & (self.subs.subset == subset)]
        return int(m.iloc[0]["n_images"]) if not m.empty else None

    def t(self, exp, view, tag, metric="top1"):
        """(mean, std, n) of a test metric, or None."""
        if self.test is None:
            return None
        m = self.test[(self.test.experiment == exp) & (self.test.view == view) & (self.test.tag == tag)
                      & (self.test.metric == metric)]
        if m.empty or pd.isna(m.iloc[0]["mean"]):
            return None
        r = m.iloc[0]
        return float(r["mean"]), float(r["std"]) if not pd.isna(r["std"]) else 0.0, int(r["n"])

    def val(self, exp, tag, metric="val_top1"):
        """(mean, std, n) over the seed runs (not the lr-grid runs)."""
        if self.runs is None:
            return None
        m = self.runs[(self.runs.experiment == exp) & (self.runs.tag == tag) & ~self.runs.run.str.contains("_lr")]
        v = m[metric].dropna().astype(float)
        if v.empty:
            return None
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))

    def params(self, exp, tag):
        if self.runs is None:
            return None
        m = self.runs[(self.runs.experiment == exp) & (self.runs.tag == tag)]
        return int(m.iloc[0]["params"]) if not m.empty and not pd.isna(m.iloc[0]["params"]) else None

    def compare(self, exp, view, a, b, subset=None):
        """Per-seed rows for A - B (from 'A:B' or a negated 'B:A'): list of (seed, diff, lo, hi, p, a_only, b_only).
        With ``subset``, the comparison on that provenance subset of the test split."""
        src = self.cmp if subset is None else self.subcmp
        if src is None:
            return []
        if subset is not None:
            src = src[src.subset == subset]
        out = []
        for pair, sign in ((f"{a}:{b}", 1), (f"{b}:{a}", -1)):
            m = src[(src.experiment == exp) & (src.view == view) & (src.pair == pair)]
            for _, r in m.sort_values("seed").iterrows():
                lo, hi = (float(x) for x in str(r["ci95"]).split(";"))
                ao, bo = int(r["a_only"]), int(r["b_only"])
                if sign < 0:
                    lo, hi, ao, bo = -hi, -lo, bo, ao
                out.append((int(r["seed"]), sign * float(r["diff"]), lo, hi, float(r["p_value"]), ao, bo))
            if out:
                return out
        return out


def rule(rows, n_seeds=3):
    """'A>B', 'B>A', 'n.s.' or 'incomplete' under the pre-registered rule."""
    if len(rows) < n_seeds:
        return "incomplete"
    d = [r[1] for r in rows]
    if all(x > 0 for x in d) and sum(r[2] > 0 for r in rows) >= 2:
        return "A>B"
    if all(x < 0 for x in d) and sum(r[3] < 0 for r in rows) >= 2:
        return "B>A"
    return "n.s."


# ---------------------------------------------------------------------------------------------- formatting
def pct(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def pm(v, scale=100, nd=2):
    if v is None:
        return "--"
    m, s, n = v
    return f"{scale * m:.{nd}f}" + (r"{\scriptsize$\,\pm\,$" + f"{scale * s:.{nd}f}" + "}" if n > 1 else "")


def tex(s):
    s = re.sub(r"^\d+-n\d+-", "", str(s))
    s = s.replace("_", " ")
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def bold_best(cells, values):
    """Bold the cell(s) holding the largest value."""
    vals = [v for v in values if v is not None]
    if not vals:
        return cells
    best = max(vals)
    return [r"\textbf{" + c + "}" if v is not None and abs(v - best) < 1e-12 else c for c, v in zip(cells, values)]


def tabular(colspec, header, rows, out: Path, name: str, written: list, note: str = ""):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"] + header + [r"\midrule"]
    for r in rows:
        lines.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append("% " + note)
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(name)


# ---------------------------------------------------------------------------------------------- tables
def t_main(D, out, written, exp, name):
    rows, top = [], []
    for group, tags in (("Baselines", BASE), ("Equal-size controls", CONTROLS), ("Proposed (RCEA)", PROPOSED)):
        if not any(D.t(exp, "orig", t) for t in tags):
            continue
        rows.append(r"\multicolumn{9}{l}{\emph{" + group + r"}} \\")
        for tag in tags:
            v = D.t(exp, "orig", tag)
            if v is None:
                continue
            p = D.params(exp, tag)
            cells = [NAME[tag], f"{p:,}" if p else "--", pm(v), pm(D.t(exp, "orig", tag, "top5")),
                     pm(D.t(exp, "orig", tag, "macro_precision")), pm(D.t(exp, "orig", tag, "macro_recall")),
                     pm(D.t(exp, "orig", tag, "macro_f1")), pm(D.t(exp, "orig", tag, "nll_ts"), 1, 3),
                     pm(D.t(exp, "orig", tag, "ece_ts"))]
            rows.append(cells)
            top.append((len(rows) - 1, v[0]))
    if not top:
        D.missing.append(name)
        return
    best = max(v for _, v in top)
    for i, v in top:
        if abs(v - best) < 1e-12:
            rows[i][2] = r"\textbf{" + rows[i][2] + "}"
    hdr = [r"Head & Params & Top-1 & Top-5 & Macro-P & Macro-R & Macro-F1 & NLL$_{\mathrm{TS}}$ & ECE$_{\mathrm{TS}}$ (\%) \\"]
    tabular("lrrrrrrrr", hdr, rows, out, name, written, f"{exp}, clean test split, mean and std over seeds")


# every pair scripts/final_all.sh compares on the clean view (all fixed before any test result), in the order of the
# forest plot; the table marks the H1 pair, which is picked on Tsinghua validation (rule 1 of the research log)
COMPARE_ROWS = ([(a, b) for a in PROPOSED for b in CONTROLS]
                + [("rr_pairwise", "rr_rival"), ("rr_rival_soft", "rr_rival"), ("rr_random", "rr_center"),
                   ("rr_center", "rr_class")]
                + [(a, "classattn_all") for a in PROPOSED] + [(a, "linear") for a in PROPOSED + ["rr_class"]]
                + [("rr_rival", "attnprobe"), ("rr_pairwise", "attnprobe")])
SHORT = {"linear": "B0", "attnprobe": "B1", "classattn_all": "B2", "rr_class": "class", "rr_center": "centre",
         "rr_random": "random", "rr_rival": "rival", "rr_rival_soft": "rival-soft", "rr_pairwise": "pairwise"}


def t_compare(D, out, written, exps, name, view="orig"):
    h1 = (best_by_val(D, "tsinghua_main", PROPOSED), best_by_val(D, "tsinghua_main", CONTROLS))
    rows = []
    for exp, label in exps:
        block = []
        for a, b in COMPARE_ROWS:
            r = D.compare(exp, view, a, b)
            if not r:
                continue
            cell = [f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]" for _, d, lo, hi, *_ in r]
            cell += ["--"] * (3 - len(cell))
            ps = max(x[4] for x in r)
            v = rule(r)
            vs = {"A>B": f"{SHORT[a]} $>$ {SHORT[b]}", "B>A": f"{SHORT[b]} $>$ {SHORT[a]}", "n.s.": "no claim",
                  "incomplete": "incomplete"}[v]
            block.append([f"{SHORT[a]} $-$ {SHORT[b]}" + (r"$^\dagger$" if (a, b) == h1 else ""),
                          f"{100 * np.mean([x[1] for x in r]):+.2f}", *cell[:3], f"{ps:.2g}", vs])
        if block:
            rows.append(r"\multicolumn{7}{l}{\emph{" + label + r"}} \\")
            rows += block
    if not rows:
        D.missing.append(name)
        return
    hdr = [r"Comparison & $\bar\Delta$ & seed 0: $\Delta$ [95\% CI] & seed 1 & seed 2 & max $p_{\mathrm{McN}}$ & Rule \\"]
    tabular("lrccccl", hdr, rows, out, name, written,
            "test top-1 differences in percentage points; paired bootstrap over test images; "
            "dagger = the H1 pair (best proposed and best control on Tsinghua validation)")


def t_robust(D, out, written, exp, ann_views, name):
    views = [v for v in AUTO + ann_views if D.t(exp, v, "linear") is not None or D.t(exp, v, "rr_rival") is not None]
    if "orig" not in views:
        D.missing.append(name)
        return
    rows = []
    for tag in ALL:
        vals = [D.t(exp, v, tag) for v in views]
        if all(x is None for x in vals):
            continue
        rows.append([NAME[tag]] + [pct(x[0] if x else None, 1) for x in vals])
    rec = [D.t(exp, v, "rr_rival", "cand_recall") for v in views]
    if any(rec):
        rows.append(r"\midrule")
        rows.append([r"Shortlist recall@5 (rival)"] + [pct(x[0] if x else None, 1) for x in rec])
    na = sum(v in AUTO for v in views)
    nb = len(views) - na
    hdr = [r" & \multicolumn{" + str(na) + r"}{c}{Automatic}" +
           (r" & \multicolumn{" + str(nb) + r"}{c}{Annotation-assisted (ground-truth boxes)}" if nb else "") + r" \\",
           r"\cmidrule(lr){2-" + str(1 + na) + "}" + (r"\cmidrule(lr){" + str(2 + na) + "-" + str(1 + na + nb) + "}" if nb else ""),
           "Head & " + " & ".join(VIEW[v] for v in views) + r" \\"]
    tabular("l" + "r" * len(views), hdr, rows, out, name, written, f"{exp}: test top-1 (%), mean over seeds")


def t_ablate(D, out, written, name_k, name_o):
    ref = "tsinghua_main"
    ab = "tsinghua_ablate"
    K = [2, 3, 5, 10]
    rows = []
    for mode in ("rr_class", "rr_rival", "rr_pairwise"):
        cells = [NAME[mode]]
        for k in K:
            tag = mode if k == 5 else f"{mode}_k{k}"
            v = D.t(ref if k == 5 else ab, "orig", tag)
            if mode == "rr_pairwise" and k == 2:
                cells.append(r"$=$ rival (Fact 4)")
            else:
                cells.append(pm(v))
        rows.append(cells)
    rec = [D.t(ref if k == 5 else ab, "orig", "rr_rival" if k == 5 else f"rr_rival_k{k}", "cand_recall") for k in K]
    if any(rec):
        rows.append(r"\midrule")
        rows.append(["Shortlist recall@$K$ (test)"] + [pct(x[0] if x else None) for x in rec])
    if all(c == "--" or c.startswith("$=") for r in rows if isinstance(r, list) for c in r[1:]):
        D.missing.append(name_k)
    else:
        tabular("lrrrr", ["Head & $K=2$ & $K=3$ & $K=5$ (main) & $K=10$ \\\\"], rows, out, name_k, written,
                "test top-1 (%), mean and std over seeds; K = 5 rows come from tsinghua_main")
    rows = []
    for tag, base, what in (("rr_rival_candloss", "rr_rival", "loss over the shortlist only"),
                            ("rr_rival_nogt", "rr_rival", "no true-class insertion in training"),
                            ("rr_rival_notokdrop", "rr_rival", "no token dropout"),
                            ("rr_pairwise_t0", "rr_pairwise", "hard minimum over pairs ($t \\to 0$)"),
                            ("rr_pairwise_t3", "rr_pairwise", "softer minimum over pairs ($t = 3$)")):
        v, r0 = D.t(ab, "orig", tag), D.t(ref, "orig", base)
        if v is None:
            continue
        d = f"{100 * (v[0] - r0[0]):+.2f}" if r0 else "--"
        rows.append([f"{SHORT[base]}: {what}", pm(D.val(ab, tag)), pm(v), d])
    if rows:
        tabular("lrrr", [r"Variant & Val top-1 & Test top-1 & $\Delta$ vs main \\"], rows, out, name_o, written,
                "one change at a time from the main configuration; same learning rate as the main head")
    else:
        D.missing.append(name_o)


def t_backbones(D, out, written, name):
    rows = []
    tags = ["linear", "rr_class", "rr_rival", "rr_pairwise"]
    for exp, label, pre in BACKBONES:
        vals = [D.t(exp, "orig", t) for t in tags]
        if all(v is None for v in vals):
            continue
        r = D.compare(exp, "orig", "rr_rival", "rr_class")
        verdict = {"A>B": "rival $>$ class", "B>A": "class $>$ rival", "n.s.": "no claim",
                   "incomplete": "incomplete"}[rule(r)] if r else "--"
        dlt = f"{100 * np.mean([x[1] for x in r]):+.2f}" if r else "--"
        rows.append([label, pre] + bold_best([pm(v) for v in vals], [v[0] if v else None for v in vals]) + [dlt, verdict])
    if rows:
        tabular("llrrrrrl", [r"Backbone & Pre-training & B0 & class & rival & pairwise & $\bar\Delta$ (rival$-$class) & Rule \\"],
                rows, out, name, written, "Tsinghua Dogs test top-1 (%); every head on the same frozen features")
    else:
        D.missing.append(name)


def t_tokens(D, out, written, name):
    rows = []
    for exp, dims in (("tsinghua_tok32", 32), ("tsinghua_tok64", 64), ("tsinghua_main", 128)):
        vals = [D.t(exp, "orig", t) for t in ("attnprobe", "rr_class", "rr_rival")]
        if all(v is None for v in vals):
            continue
        rows.append([str(dims)] + [pm(v) for v in vals])
    if len(rows) > 1:
        tabular("lrrr", [r"Token PCA dims & B1 & class & rival \\"], rows, out, name, written, "Tsinghua test top-1 (%)")
    else:
        D.missing.append(name)


def t_ood(D, out, written, name):
    if D.ood is None:
        D.missing.append(name)
        return
    o = D.ood[D.ood.file == "ood_orig"]

    def g(tag, m):
        r = o[(o.tag == tag) & (o.metric == m)]
        return (float(r.iloc[0]["mean"]), float(r.iloc[0]["std"]) if not pd.isna(r.iloc[0]["std"]) else 0.0,
                int(r.iloc[0]["n"])) if not r.empty else None
    rows = []
    for tag in ["linear", "rr_class", "rr_rival", "rr_pairwise"]:
        if g(tag, "top1_known") is None:
            continue
        rows.append([NAME[tag], pm(g(tag, "top1_known")), pm(g(tag, "msp_auroc")), pm(g(tag, "msp_fpr95")),
                     pm(g(tag, "energy_auroc")), pm(g(tag, "energy_fpr95")), pm(g(tag, "maxlogit_auroc")),
                     pm(g(tag, "g_energy_auroc")) if tag != "linear" else "--"])
    if rows:
        tabular("lrrrrrrr", [r"Head & Top-1 (known) & \multicolumn{2}{c}{MSP} & \multicolumn{2}{c}{Energy} & MaxLogit & Energy, stage 1 \\",
                             r" & & AUROC & FPR95 & AUROC & FPR95 & AUROC & AUROC \\"], rows, out, name, written,
                "100 known breeds trained, 30 held-out breeds as unknowns; % ; mean and std over seeds")
    else:
        D.missing.append(name)


def t_transfer(D, out, written, name, numbers):
    if D.tr is None:
        D.missing.append(name)
        return
    rows = []

    def g(tag, m):
        r = D.tr[(D.tr.tag == tag) & (D.tr.metric == m)]
        return (float(r.iloc[0]["mean"]), float(r.iloc[0]["std"]) if not pd.isna(r.iloc[0]["std"]) else 0.0,
                int(r.iloc[0]["n"])) if not r.empty else None
    for tag in ALL:
        if g(tag, "top1") is None:
            continue
        rows.append([NAME[tag], pm(g(tag, "top1")), pm(g(tag, "top5")), pm(g(tag, "top1_mapped_only")),
                     pm(g(tag, "top1_global")) if tag.startswith("rr_") else "--"])
    if rows:
        n_img, n_br = int(D.tr.iloc[0]["n_images"]), int(D.tr.iloc[0]["n_breeds"])
        numbers["transfer"] = {"n_images": n_img, "n_breeds": n_br}
        tabular("lrrrr", [r"Head (trained on Tsinghua Dogs) & Top-1 & Top-5 & Top-1, mapped breeds only & Stage-1 top-1 \\"],
                rows, out, name, written, f"{n_img} de-duplicated Stanford test images of {n_br} breeds")
    else:
        D.missing.append(name)


def t_eff(D, out, written, name):
    if D.eff is None:
        D.missing.append(name)
        return
    e = D.eff[D.eff.experiment == "tsinghua_main"]
    if e.empty:
        D.missing.append(name)
        return
    rows = []
    bb1 = e[(e.part == "backbone") & (e.batch == 1)]
    bb64 = e[(e.part == "backbone") & (e.batch == 64)]
    if not bb1.empty:
        r = bb1.iloc[0]
        rows.append([r"Backbone (frozen, fp16)", f"{int(r['params']):,}", f"{r['gflops']:.1f} G", f"{r['median_ms']:.1f}",
                     f"{bb64.iloc[0]['median_ms']:.0f} (b=64)" if not bb64.empty else "--",
                     f"{bb64.iloc[0]['peak_gpu_mb']:.0f}" if not bb64.empty and not pd.isna(bb64.iloc[0]['peak_gpu_mb']) else "--", "--"])
        rows.append(r"\midrule")
    for tag in ALL:
        h1 = e[(e.part == "head") & (e.name == tag) & (e.batch == 1)]
        if h1.empty:
            continue
        h256 = e[(e.part == "head") & (e.name == tag) & (e.batch == 256)]
        pp = e[(e.part == "pipeline") & (e.name == tag)]
        r = h1.iloc[0]
        rows.append([NAME[tag], f"{int(r['params']):,}", f"{1000 * r['gflops']:.1f} M", f"{r['median_ms']:.2f}",
                     f"{h256.iloc[0]['median_ms']:.1f}" if not h256.empty else "--",
                     f"{h256.iloc[0]['peak_gpu_mb']:.0f}" if not h256.empty and not pd.isna(h256.iloc[0]['peak_gpu_mb']) else "--",
                     f"{pp.iloc[0]['median_ms']:.1f}" if not pp.empty else "--"])
    dev = e.iloc[0]["device"]
    # the repeat counts are recorded by efficiency_report.py; older reports did not record them and used its
    # defaults then (10 warm-up passes, 50 timed passes)
    rep = {k: (sorted({int(v) for v in e[k].dropna()}) if k in e else []) for k in ("warmup", "iters")}
    warm, iters = (rep["warmup"] or [10]), (rep["iters"] or [50])
    timing = (f"median of {iters[0]} timed passes after {warm[0]} warm-up passes" if len(warm) == len(iters) == 1
              else f"median of {'/'.join(map(str, iters))} timed passes after {'/'.join(map(str, warm))} warm-up passes")
    tabular("lrrrrrr", [r"Part & Params & FLOPs / image & ms (b=1) & ms (b=256) & Peak MB (b=256) & Pipeline ms (b=1) \\"],
            rows, out, name, written, f"device {dev}; FLOPs from torch.utils.flop_counter (a multiply-add = 2 FLOPs; "
                                      f"G = 1e9, M = 1e6); {timing}")


SUBSETS = [("all", "All test images"), ("stanford_origin", "Stanford-origin"), ("independent", "Independent")]
SUB_PAIRS = [("rr_rival", "rr_class"), ("rr_rival", "rr_center"), ("rr_rival", "rr_random"),
             ("rr_rival_soft", "rr_class"), ("rr_pairwise", "rr_class"), ("rr_pairwise", "rr_rival"),
             ("rr_rival", "classattn_all"), ("rr_rival", "linear"), ("rr_class", "linear")]
SIGN = {"A>B": "$>$", "B>A": "$<$", "n.s.": "n.s.", "incomplete": "inc."}


def t_provenance(D, out, written, exp, name, numbers):
    """Top-1 and the main paired differences on the provenance subsets of the test split (subset_eval.py)."""
    ns = {k: D.sub_n(exp, k) for k, _ in SUBSETS}
    if not any(ns.values()):
        D.missing.append(name)
        return
    rows, rec = [], {"n": ns, "top1": {}, "comparisons": {}}
    for tag in ALL:
        vals = [D.sub(exp, k, tag) for k, _ in SUBSETS]
        if all(v is None for v in vals):
            continue
        rec["top1"][tag] = {k: (v[0] if v else None) for (k, _), v in zip(SUBSETS, vals)}
        rows.append([NAME[tag]] + [pm(v) for v in vals])
    block = []
    # the H1 pair and the other rule-1 comparisons of its proposed head come first
    bp, bc = best_by_val(D, "tsinghua_main", PROPOSED), best_by_val(D, "tsinghua_main", CONTROLS)
    first = [(bp, bc), (bp, "classattn_all"), (bp, "linear")] if bp and bc else []
    for a, b in first + [p for p in SUB_PAIRS if p not in first]:
        cells, verdicts = [], {}
        for k, _ in SUBSETS:
            r = D.compare(exp, "orig", a, b, subset=k)
            if r:
                verdicts[k] = {"rule": rule(r), "mean_diff": float(np.mean([x[1] for x in r]))}
                cells.append(f"{100 * verdicts[k]['mean_diff']:+.2f} ({SIGN[verdicts[k]['rule']]})")
            else:
                cells.append("--")
        if verdicts:
            rec["comparisons"][f"{a}:{b}"] = verdicts
            block.append([f"{SHORT[a]} $-$ {SHORT[b]}" + (r"$^\dagger$" if (a, b) == (bp, bc) else "")] + cells)
    if block:
        rows += [r"\midrule", r"\multicolumn{4}{l}{\emph{Paired differences in pp, mean over seeds (rule of \S8)}} \\"]
        rows += block
    numbers.setdefault("provenance", {})[exp] = rec
    hdr = ["Head & " + " & ".join(f"{lab} ({ns[k]:,})" if ns[k] else lab for k, lab in SUBSETS) + r" \\"]
    tabular("lrrr", hdr, rows, out, name, written,
            "test top-1 (%); Stanford-origin = confirmed Stanford Dogs copy (in ImageNet-1k train); "
            "independent = no candidate match at all")


def t_rerank(D, out, written, name):
    """What the second stage does: its own stage-1 accuracy, the shortlist recall, and the share of test images
    whose top-1 it changes (fixes: wrong -> right; breaks: right -> wrong), all as % of the test images."""
    rows = []
    for exp, label in (("tsinghua_main", "Tsinghua Dogs"), ("stanford_main", "Stanford Dogs")):
        block = []
        for tag in CONTROLS + PROPOSED:
            fin = D.t(exp, "orig", tag)
            if fin is None:
                continue
            g = D.t(exp, "orig", tag, "top1_global")
            block.append([NAME[tag], pm(g), pm(fin), f"{100 * (fin[0] - g[0]):+.2f}" if g else "--",
                          pm(D.t(exp, "orig", tag, "cand_recall")), pm(D.t(exp, "orig", tag, "flip_rate")),
                          pm(D.t(exp, "orig", tag, "fix_rate")), pm(D.t(exp, "orig", tag, "break_rate"))])
        if block:
            rows.append(r"\multicolumn{8}{l}{\emph{" + label + r"}} \\")
            rows += block
    if rows:
        tabular("lrrrrrrr", [r"Head & Stage-1 top-1 & Final top-1 & Gain & Recall@5 & Changed & Fixed & Broken \\"],
                rows, out, name, written, "test split, % of test images, mean and std over seeds")
    else:
        D.missing.append(name)


def t_explain(D, out, written, name):
    if D.ex is None:
        D.missing.append(name)
        return
    rows = []
    for tag in ALL:
        r = D.ex[D.ex.tag == tag]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append([NAME[tag], pct(r["top1"], 1), pct(r["body_mass_mean"], 1), pct(r["body_area_mean"], 1),
                     f"{r['body_lift_mean']:.2f}", pct(r["head_mass_mean"], 1), pct(r["head_area_mean"], 1),
                     f"{r['head_lift_mean']:.2f}", f"{r['entropy_mean'] / r['entropy_uniform']:.3f}"])
    if rows:
        tabular("lrrrrrrrr", [r" & & \multicolumn{3}{c}{Body box} & \multicolumn{3}{c}{Head box} & \\",
                              r"Head & Top-1 & mass & area & lift & mass & area & lift & $H/H_{\mathrm{unif}}$ \\"],
                rows, out, name, written, "attention mass inside the ground-truth boxes (evaluation only), seed 0, test")
    else:
        D.missing.append(name)


def t_hard(D, out, written, exp, name, numbers):
    if D.hard is None:
        D.missing.append(name)
        return
    h = D.hard[(D.hard.experiment == exp) & (D.hard.view == "orig")]
    if h.empty:
        D.missing.append(name)
        return
    lin_pair = h[h.tag == "linear"]["pair_class_acc_mean"]
    lin_pair = float(lin_pair.iloc[0]) if len(lin_pair) and not pd.isna(lin_pair.iloc[0]) else None
    lin_all = D.t(exp, "orig", "linear")
    rows, gains = [], {}
    for tag in ALL:
        r = h[h.tag == tag]
        if r.empty:
            continue
        r = r.iloc[0]
        top = D.t(exp, "orig", tag)
        pa = float(r["pair_class_acc_mean"]) if not pd.isna(r["pair_class_acc_mean"]) else None
        g_pair = (pa - lin_pair) if pa is not None and lin_pair is not None else None
        g_all = (top[0] - lin_all[0]) if top and lin_all else None
        gains[tag] = {"pair_gain": g_pair, "overall_gain": g_all}
        rows.append([NAME[tag], f"{r['pair_errors_mean']:.1f}", pct(pa), pct(top[0] if top else None),
                     f"{100 * g_pair:+.2f}" if g_pair is not None else "--", f"{100 * g_all:+.2f}" if g_all is not None else "--"])
    numbers.setdefault("hard_pairs", {})[exp] = {"pairs": str(h.iloc[0]["pairs"]), "gains": gains}
    tabular("lrrrrr", [r"Head & Pair errors & Acc. on pair breeds & Top-1 & Gain on pairs & Gain overall \\"],
            rows, out, name, written, "the 20 breed pairs most confused by the linear probe on VALIDATION; gains vs B0 in pp")


def t_lr(D, out, written, name):
    if D.sel is None:
        D.missing.append(name)
        return
    exps = [e for e in ["tsinghua_main", "stanford_main", "tsinghua_resnet50", "tsinghua_convnext_tiny",
                        "tsinghua_clip_vitb16"] if (D.sel.experiment == e).any()]
    rows = []
    for tag in ALL:
        cells = []
        for e in exps:
            m = D.sel[(D.sel.experiment == e) & (D.sel.tag == tag)]
            cells.append(f"{float(m.iloc[0]['lr']):g}" if not m.empty else "--")
        if any(c != "--" for c in cells):
            rows.append([NAME[tag]] + cells)
    tabular("l" + "r" * len(exps), ["Head & " + " & ".join(tex(e) for e in exps) + r" \\"], rows, out, name, written,
            "learning rate chosen on validation top-1, seed 0, from {3e-4, 1e-3, 3e-3}")


def t_data(D, out, written, name, numbers):
    a = D.audit
    if not a:
        D.missing.append(name)
        return
    rows = []
    for ds, label in (("tsinghua", "Tsinghua Dogs"), ("stanford", "Stanford Dogs")):
        j = a.get(ds)
        if not j:
            continue
        s = j.get("splits", {})
        kinds = j.get("phash_pairs_by_kind", {})
        tt = sum(v for k, v in kinds.items() if set(k.split("-")) == {"train", "test"})
        rows.append([label, f"{j.get('images_listed', 0):,}", f"{s.get('train', 0):,}", f"{s.get('val', 0):,}",
                     f"{s.get('test', 0):,}", f"{s.get('excluded', 0):,}",
                     f"{j.get('train_excluded_as_test_duplicates', 0):,}", str(j.get("undecodable", "--")),
                     str(j.get("exact_duplicate_groups", "--")),
                     f"{j.get('phash_near_pairs', 0):,} ({tt})", str(j.get("phash_pairs_label_conflicts", "--"))])
        numbers.setdefault("data", {})[ds] = {k: j.get(k) for k in ("images_listed", "splits", "exact_duplicate_groups",
                                                                   "phash_near_pairs", "phash_pairs_label_conflicts",
                                                                   "train_excluded_as_test_duplicates", "undecodable")}
    if rows:
        tabular("lrrrrrrrrrr", [r" & & \multicolumn{4}{c}{Split used} & \multicolumn{2}{c}{Removed} & \multicolumn{3}{c}{Duplicates} \\",
                                r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
                                r"Dataset & Images & Train & Val & Test & Excl. & Train dup. of test & Undecodable & Exact groups & Near pairs (train--test) & Label conflicts \\"],
                rows, out, name, written, "within-dataset audit: SHA-1 exact, pHash Hamming <= 4 near")
    c = a.get("cross_tsinghua_stanford")
    if c:
        numbers["cross"] = {k: c.get(k) for k in ("candidate_pairs", "confirmed_pairs", "confirmed_tsinghua_images",
                                                  "confirmed_stanford_images", "confirmed_by_split",
                                                  "confirmed_same_breed_name", "stanford_test_in_tsinghua_trainval",
                                                  "stanford_test_clean", "rule", "with_descriptor")}
        by_split = []
        for k, v in sorted((c.get("confirmed_by_split") or {}).items()):
            ts, ss = (k.split("|") + [""])[:2]
            by_split.append([rf"\quad Tsinghua {tex(ts)} / Stanford {tex(ss)}", f"{v:,}"])
        rule_ = c.get("rule") or {}
        rule_txt = (f"confirmed = pHash Hamming <= {rule_.get('phash_dist')} and cosine >= {rule_.get('cos_pair')}, "
                    f"or cosine >= {rule_.get('cos_alone')} (DINOv2 global descriptors)") if isinstance(rule_, dict) else ""
        rows = [["Candidate pairs", f"{c.get('candidate_pairs', 0):,}"], ["Confirmed pairs", f"{c.get('confirmed_pairs', 0):,}"],
                ["Tsinghua images with a confirmed Stanford copy", f"{c.get('confirmed_tsinghua_images', 0):,}"],
                ["Stanford images with a confirmed Tsinghua copy", f"{c.get('confirmed_stanford_images', 0):,}"],
                ["Confirmed pairs with the same breed name", pct(c.get("confirmed_same_breed_name"), 1) + r"\%"],
                ["Stanford test images confirmed in Tsinghua train/val", f"{c.get('stanford_test_in_tsinghua_trainval', 0):,}"],
                ["Clean Stanford test images (no candidate in Tsinghua train/val)", f"{c.get('stanford_test_clean', 0):,}"],
                [r"Confirmed pairs by split:", ""]] + by_split
        tabular("lr", [r"Cross-dataset audit (Tsinghua vs Stanford) & Count \\"], rows, out, f"{name}_cross", written,
                rule_txt)


# ---------------------------------------------------------------------------------------------- hypotheses
def best_by_val(D, exp, tags):
    """The tag with the highest mean validation top-1 over its seed runs; ties go to the lower mean validation
    NLL (the same order as the learning-rate selection). Validation data only: fixed before any test result."""
    v = {t: (D.val(exp, t), D.val(exp, t, "val_nll")) for t in tags}
    v = {t: (a[0], -(b[0] if b else 0.0)) for t, (a, b) in v.items() if a}
    return max(v, key=lambda t: v[t]) if v else None


def hypotheses(D, numbers, out, written):
    H = {}
    # H1 compares the best proposed mode with the best control, both picked on Tsinghua validation data. Stanford
    # repeats that same pair ("same sign on Stanford", design.md §8); its own validation picks are reported beside.
    ref = {"bp": best_by_val(D, "tsinghua_main", PROPOSED), "bc": best_by_val(D, "tsinghua_main", CONTROLS)}
    for exp in ("tsinghua_main", "stanford_main"):
        own_bp, own_bc = best_by_val(D, exp, PROPOSED), best_by_val(D, exp, CONTROLS)
        if not own_bp or not own_bc:
            continue
        bp, bc = (ref["bp"], ref["bc"]) if ref["bp"] and ref["bc"] else (own_bp, own_bc)
        r = D.compare(exp, "orig", bp, bc)
        r_own = D.compare(exp, "orig", own_bp, own_bc)
        H.setdefault("H1", {})[exp] = {"best_proposed_by_val": bp, "best_control_by_val": bc,
                                       "pair_chosen_on": "tsinghua_main validation" if ref["bp"] and ref["bc"] else f"{exp} validation",
                                       "own_val_pair": f"{own_bp}:{own_bc}", "own_val_pair_rule": rule(r_own) if r_own else "missing",
                                       "rule": rule(r) if r else "missing",
                                       "per_seed": [x[1] for x in r] if r else [],
                                       "all_proposed_vs_controls": {f"{a}:{b}": rule(D.compare(exp, "orig", a, b)) or "missing"
                                                                    for a in PROPOSED for b in CONTROLS
                                                                    if D.compare(exp, "orig", a, b)},
                                       "by_subset": {k: rule(D.compare(exp, "orig", bp, bc, subset=k))
                                                     for k, _ in SUBSETS if D.compare(exp, "orig", bp, bc, subset=k)}}
        # H2: fixes vs breaks, and the gain on the hard pairs vs overall
        fx, br = D.t(exp, "orig", bp, "fix_rate"), D.t(exp, "orig", bp, "break_rate")
        hp = numbers.get("hard_pairs", {}).get(exp, {}).get("gains", {}).get(bp, {})
        if fx and br:
            ok = fx[0] > br[0] and hp.get("pair_gain") is not None and hp.get("overall_gain") is not None \
                and hp["pair_gain"] > hp["overall_gain"]
            H.setdefault("H2", {})[exp] = {"head": bp, "fix_rate": fx[0], "break_rate": br[0], **hp,
                                           "verdict": "supported" if ok else "refuted"}
        # H3: random <= centre; refuted if random within the seed spread of rival
        vr, vce, vri = D.t(exp, "orig", "rr_random"), D.t(exp, "orig", "rr_center"), D.t(exp, "orig", "rr_rival")
        if vr and vce and vri:
            spread = max(vr[1], vri[1])
            refuted = abs(vr[0] - vri[0]) <= spread
            H.setdefault("H3", {})[exp] = {"random": vr[0], "center": vce[0], "rival": vri[0], "seed_spread": spread,
                                           "random_le_center": vr[0] <= vce[0],
                                           "verdict": "refuted" if refuted else ("supported" if vr[0] <= vce[0] else "partly")}
        # H4: pairwise vs rival (exploratory)
        r4 = D.compare(exp, "orig", "rr_pairwise", "rr_rival")
        if r4:
            H.setdefault("H4", {})[exp] = {"rule": rule(r4), "mean_diff": float(np.mean([x[1] for x in r4]))}
        # H5: the best proposed head vs all-class attention
        r5 = D.compare(exp, "orig", bp, "classattn_all")
        if r5:
            v5 = rule(r5)
            H.setdefault("H5", {})[exp] = {"head": bp, "rule": v5, "mean_diff": float(np.mean([x[1] for x in r5])),
                                           "verdict": {"B>A": "refuted", "A>B": "supported (re-ranker better)",
                                                       "n.s.": "supported (no difference)"}.get(v5, v5)}
        # H7: robustness gain of the best proposed head over B0, per automatic corruption
        g = {}
        for v in AUTO:
            a_, b_ = D.t(exp, v, bp), D.t(exp, v, "linear")
            if a_ and b_:
                g[v] = a_[0] - b_[0]
        if "orig" in g and len(g) > 1:
            larger = [v for v in g if v != "orig" and g[v] > g["orig"]]
            H.setdefault("H7", {})[exp] = {"head": bp, "gain_over_B0": g, "views_gain_above_clean": larger,
                                           "verdict": "exploratory"}
    # H6: K sweep on Tsinghua
    acc, rec = {}, {}
    for k in (2, 3, 5, 10):
        exp, tag = ("tsinghua_main", "rr_rival") if k == 5 else ("tsinghua_ablate", f"rr_rival_k{k}")
        v, c = D.t(exp, "orig", tag), D.t(exp, "orig", tag, "cand_recall")
        if v:
            acc[k] = v[0]
        if c:
            rec[k] = c[0]
    if len(acc) == 4:
        a = [acc[k] for k in (2, 3, 5, 10)]
        H["H6"] = {"top1": acc, "recall": rec, "monotone_fall_from_2": all(x > y for x, y in zip(a, a[1:])),
                   "recall_rises": all(x <= y for x, y in zip([rec.get(k, 0) for k in (2, 3, 5, 10)],
                                                              [rec.get(k, 0) for k in (3, 5, 10)])) if len(rec) == 4 else None}
        H["H6"]["verdict"] = "refuted" if H["H6"]["monotone_fall_from_2"] else "supported"
    # one verdict per hypothesis; Tsinghua Dogs is the primary dataset (design.md §8)
    T, St = "tsinghua_main", "stanford_main"
    if "H1" in H and T in H["H1"]:
        t1 = H["H1"][T]
        s_sign = (np.mean(H["H1"][St]["per_seed"]) if St in H["H1"] and H["H1"][St]["per_seed"] else None)
        H["H1"]["verdict"] = ("incomplete" if t1["rule"] in ("incomplete", "missing") else
                              "refuted" if t1["rule"] != "A>B" else
                              "supported" if s_sign is not None and s_sign > 0 else
                              "partly (Stanford sign differs)" if s_sign is not None else "supported on Tsinghua only")
    for h in ("H2", "H3", "H5"):
        if h in H and T in H[h]:
            H[h]["verdict"] = H[h][T]["verdict"]
    if "H4" in H and T in H["H4"]:
        H["H4"]["verdict"] = "exploratory: " + {"A>B": "pairwise $>$ rival", "B>A": "rival $>$ pairwise",
                                                 "n.s.": "no difference"}.get(H["H4"][T]["rule"], H["H4"][T]["rule"])
    if "H7" in H:
        H["H7"]["verdict"] = "exploratory"
    numbers["hypotheses"] = H
    rows = []
    for h, label in (("H1", "Shortlist conditioning beats equal-size controls"), ("H2", "It acts where it should"),
                     ("H3", "It is the confusion set, not contrast"), ("H4", "Pairwise adds beyond centring"),
                     ("H5", "The shortlist costs nothing vs all classes"), ("H6", "K behaves like a shortlist"),
                     ("H7", "Robustness under corruptions")):
        v = H.get(h)
        if not v:
            rows.append([h, label, "not run", "--"])
            continue
        if h == "H6":
            ev = "; ".join(f"K={k}: {100 * x:.2f}" for k, x in sorted(v["top1"].items()))
            if v.get("recall"):
                ev += " (recall " + ", ".join(f"{100 * x:.1f}" for _, x in sorted(v["recall"].items())) + ")"
            rows.append([h, label, v["verdict"], ev])
            continue
        parts = []
        for exp in (T, St):
            if exp not in v:
                continue
            e = v[exp]
            short = "Tsinghua" if exp == T else "Stanford"
            if h == "H1":
                own = e["own_val_pair"].split(":")
                parts.append(f"{short}: {SHORT[e['best_proposed_by_val']]} vs {SHORT[e['best_control_by_val']]} "
                             f"(chosen on {'Tsinghua ' if e['pair_chosen_on'].startswith('tsinghua') else ''}val): {e['rule']}"
                             + (", per seed " + "/".join(f"{100 * d:+.2f}" for d in e["per_seed"]) + " pp"
                                if e["per_seed"] else "")
                             + ("; Stanford-origin " + e["by_subset"]["stanford_origin"] if "stanford_origin" in
                                e.get("by_subset", {}) else "")
                             + ("; independent " + e["by_subset"]["independent"] if "independent" in
                                e.get("by_subset", {}) else "")
                             + (f"; own val pick {SHORT[own[0]]} vs {SHORT[own[1]]}: {e['own_val_pair_rule']}"
                                if own != [e["best_proposed_by_val"], e["best_control_by_val"]] else ""))
            elif h == "H2":
                parts.append(f"{short}: fix {100 * e['fix_rate']:.2f}\\% vs break {100 * e['break_rate']:.2f}\\%; "
                             + (f"gain on pairs {100 * e['pair_gain']:+.2f} vs overall {100 * e['overall_gain']:+.2f} pp"
                                if e.get("pair_gain") is not None and e.get("overall_gain") is not None else "pairs --"))
            elif h == "H3":
                parts.append(f"{short}: random {100 * e['random']:.2f}, centre {100 * e['center']:.2f}, "
                             f"rival {100 * e['rival']:.2f} (spread {100 * e['seed_spread']:.2f})")
            elif h in ("H4", "H5"):
                parts.append(f"{short}: {e['rule']} ({100 * e['mean_diff']:+.2f} pp)")
            elif h == "H7":
                parts.append(f"{short} ({SHORT[e['head']]} $-$ B0, pp): "
                             + ", ".join(f"{VIEW[k]} {100 * x:+.1f}" for k, x in e["gain_over_B0"].items()))
        rows.append([h, label, v.get("verdict", "--"), "; ".join(parts)])
    tabular("lp{4.2cm}p{2.6cm}p{8.4cm}", [r"& Hypothesis & Verdict & Evidence (test split; decision rule of \S8) \\"],
            rows, out, "tab_hypotheses", written, "verdicts computed by scripts/make_tables.py from the digest")


# ---------------------------------------------------------------------------------------------- figure data
FOREST = [p for p in COMPARE_ROWS if p[1] != "attnprobe"]          # the comparison table's rows without B1


def figdata(D, numbers):
    F = {}
    for exp in ("tsinghua_main", "stanford_main"):
        # per-seed paired differences A - B on the clean test view: rows [seed, diff, lo, hi, McNemar p], and the
        # verdict of the pre-registered rule
        cmp = {f"{a}:{b}": D.compare(exp, "orig", a, b) for a, b in FOREST}
        cmp = {k: {"rule": rule(v), "rows": [list(r[:5]) for r in v]} for k, v in cmp.items() if v}
        if cmp:
            F.setdefault("compare", {})[exp] = cmp
        rob = {v: {t: D.t(exp, v, t)[0] for t in ALL if D.t(exp, v, t)} for v in AUTO + ANN}
        rob = {v: x for v, x in rob.items() if x}
        rec = {v: D.t(exp, v, "rr_rival", "cand_recall")[0] for v in rob if D.t(exp, v, "rr_rival", "cand_recall")}
        if rob:
            F.setdefault("robustness", {})[exp] = {"top1": rob, "recall_rival": rec}
        if D.pc is not None:
            p = D.pc[D.pc.experiment == exp]
            if not p.empty:
                p = p.assign(class_name=p["class_name"].fillna("").astype(str))
                piv = p.pivot_table(index=["class", "class_name"], columns="tag", values="acc").reset_index()
                piv = piv.sort_values("class")
                F.setdefault("per_class", {})[exp] = {
                    "class_name": piv["class_name"].tolist(),
                    **{c: [None if pd.isna(x) else round(float(x), 4) for x in piv[c]]
                       for c in piv.columns if c not in ("class", "class_name")}}
        main = {t: {"val": D.val(exp, t), "test": D.t(exp, "orig", t), "params": D.params(exp, t)} for t in ALL}
        F.setdefault("main", {})[exp] = {t: v for t, v in main.items() if v["test"] or v["val"]}
    if D.eff is not None:
        e = D.eff[(D.eff.experiment == "tsinghua_main") & (D.eff.part == "head") & (D.eff.batch == 1)]
        F["efficiency"] = {r["name"]: {"mflops": 1000 * float(r["gflops"]), "params": int(r["params"]),
                                       "top1": (D.t("tsinghua_main", "orig", r["name"]) or [None])[0]}
                           for _, r in e.iterrows()}
    if D.curves is not None:
        F["curves"] = {run: g[["epoch", "train_loss", "val_top1", "val_nll"]].astype(float).round(5).values.tolist()
                       for run, g in D.curves.groupby("run")}
    if "H6" in numbers.get("hypotheses", {}):
        F["k_sweep"] = numbers["hypotheses"]["H6"]
    ks = {}
    for k in (2, 3, 5, 10):
        exp, tag = ("tsinghua_main", "rr_rival") if k == 5 else ("tsinghua_ablate", f"rr_rival_k{k}")
        t1, rc = D.t(exp, "orig", tag), D.t(exp, "orig", tag, "cand_recall")
        if t1:
            ks[k] = {"top1": list(t1[:3]), "recall": list(rc[:3]) if rc else None}
    if ks:
        F["k_sweep_detail"] = ks          # (mean, std, n) over seeds
    F["hypotheses"] = {h: {e: {k: v for k, v in x.items() if k in ("best_proposed_by_val", "best_control_by_val")}
                           for e, x in numbers.get("hypotheses", {}).get(h, {}).items() if isinstance(x, dict)}
                       for h in ("H1",)}
    return F


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", default="results/digest")
    ap.add_argument("--out", default="paper/generated")
    a = ap.parse_args(argv)
    D = Digest(Path(a.digest))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written, numbers = [], {}
    t_data(D, out, written, "tab_data", numbers)
    t_main(D, out, written, "tsinghua_main", "tab_main_tsinghua")
    t_main(D, out, written, "stanford_main", "tab_main_stanford")
    t_compare(D, out, written, [("tsinghua_main", "Tsinghua Dogs"), ("stanford_main", "Stanford Dogs")], "tab_compare")
    t_rerank(D, out, written, "tab_rerank")
    t_provenance(D, out, written, "tsinghua_main", "tab_provenance", numbers)
    t_robust(D, out, written, "tsinghua_main", ANN, "tab_robust_tsinghua")
    t_robust(D, out, written, "stanford_main", ["bg_gray", "bg_blur", "crop_body"], "tab_robust_stanford")
    t_ablate(D, out, written, "tab_ablate_k", "tab_ablate_other")
    t_backbones(D, out, written, "tab_backbones")
    t_tokens(D, out, written, "tab_tokens")
    t_ood(D, out, written, "tab_openset")
    t_transfer(D, out, written, "tab_transfer", numbers)
    t_eff(D, out, written, "tab_efficiency")
    t_explain(D, out, written, "tab_explain")
    t_hard(D, out, written, "tsinghua_main", "tab_hardpairs_tsinghua", numbers)
    t_hard(D, out, written, "stanford_main", "tab_hardpairs_stanford", numbers)
    t_lr(D, out, written, "tab_lr")
    hypotheses(D, numbers, out, written)
    numbers["missing"] = D.missing
    numbers["written"] = written
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")
    (out / "figdata.json").write_text(json.dumps(figdata(D, numbers), separators=(",", ":"), default=str),
                                      encoding="utf-8")
    print(f"{len(written)} tables -> {out}; missing: {', '.join(D.missing) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
