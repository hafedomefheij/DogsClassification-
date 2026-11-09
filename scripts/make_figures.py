"""The paper's figures, drawn from the generated data only (no test-set access, nothing is trained).

    python -I scripts/make_figures.py --gen paper/generated --out paper/figures \
        --explain results/tsinghua_main/explain/shown_maps.json --images data/raw/tsinghua_dogs/low-resolution.zip

Inputs: ``figdata.json`` from ``scripts/make_tables.py``; for the attention figure, the quantised maps that
``scripts/explain.py`` saved and the original images (re-cropped exactly as the backbone saw them).
Outputs: vector PDFs (``--png`` adds previews). A figure whose data is missing is skipped and listed.

Colour carries the head's family only (baselines / re-rank controls / proposed), from a palette checked for
colour-vision deficiency; every head also has its own marker and its name on the axis or in the legend, so no
identity rests on colour. Each figure has a table with the same numbers in the paper.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

FAMILY = {"linear": "base", "attnprobe": "base", "classattn_all": "base",
          "rr_class": "ctrl", "rr_center": "ctrl", "rr_random": "ctrl",
          "rr_rival": "prop", "rr_rival_soft": "prop", "rr_pairwise": "prop"}
ORDER = list(FAMILY)
COLOR = {"base": "#2a78d6", "ctrl": "#eb6834", "prop": "#1baf7a"}     # validated: CVD all-pairs dE >= 9.2
FAMNAME = {"base": "baselines", "ctrl": "re-rank controls", "prop": "RCEA (proposed)"}
MARKER = {"linear": "o", "attnprobe": "h", "classattn_all": "p", "rr_class": "s", "rr_center": "P",
          "rr_random": "X", "rr_rival": "D", "rr_rival_soft": "v", "rr_pairwise": "^"}
LABEL = {"linear": "B0 linear probe", "attnprobe": "B1 attentive probe", "classattn_all": "B2 class attention",
         "rr_class": "class query", "rr_center": "centred query", "rr_random": "random rivals",
         "rr_rival": "RCEA rival", "rr_rival_soft": "RCEA rival-soft", "rr_pairwise": "RCEA pairwise"}
SHORT = {"linear": "B0", "attnprobe": "B1", "classattn_all": "B2", "rr_class": "class", "rr_center": "centred",
         "rr_random": "random", "rr_rival": "rival", "rr_rival_soft": "rival-soft", "rr_pairwise": "pairwise"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS, NEG = "#2a78d6", "#e34948"                                       # diverging poles, validated
EXP = {"tsinghua_main": "Tsinghua Dogs", "stanford_main": "Stanford Dogs"}
AUTO = ["orig", "blur", "noise", "jpeg", "lowres", "occlude"]
ANN = ["head_occlude", "bg_gray", "bg_blur", "crop_body", "crop_head"]
VIEW = {"orig": "clean", "blur": "blur", "noise": "noise", "jpeg": "JPEG", "lowres": "low resolution",
        "occlude": "random occlusion", "head_occlude": "head occluded", "bg_gray": "background greyed",
        "bg_blur": "background blurred", "crop_body": "body crop", "crop_head": "head crop"}
KIND = {"fixed": "re-rank fixed\nthe stage-1 error", "broken": "re-rank broke a\ncorrect stage 1",
        "both right": "both right"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
FULL, HALF = 6.3, 3.1                                                 # inches: the paper's text width, half


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titleweight": "bold",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})


def ci(mean_std_n):
    """Half-width of the t-based 95 % interval over seeds."""
    m, s, n = mean_std_n[:3]
    return T975.get(int(n) - 1, 1.96) * s / math.sqrt(n) if n and n > 1 else 0.0


def head_handles(tags):
    return [Line2D([], [], ls="", marker=MARKER[t], ms=5, mfc=COLOR[FAMILY[t]], mec="white", mew=0.6,
                   label=LABEL[t]) for t in tags]


def family_handles(fams=("base", "ctrl", "prop")):
    return [Line2D([], [], ls="", marker="s", ms=5, mfc=COLOR[f], mec="white", mew=0.6, label=FAMNAME[f])
            for f in fams]


def save(fig, out, name, png, written):
    fig.savefig(out / f"{name}.pdf")
    if png:
        fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    written.append(name)


# ------------------------------------------------------------------------------------------------- figures
def fig_main(F, out, png, written, missing):
    """Test top-1 of every head, mean and 95 % interval over seeds, one panel per dataset."""
    exps = [e for e in EXP if e in F.get("main", {}) and any(v.get("test") for v in F["main"][e].values())]
    if not exps:
        missing.append("fig_main")
        return
    fig, axes = plt.subplots(1, len(exps), figsize=(FULL, 2.55), squeeze=False)
    for ax, exp in zip(axes[0], exps):
        rows = [t for t in ORDER if F["main"][exp].get(t, {}).get("test")]
        y = np.arange(len(rows))[::-1]
        lo, hi = [], []
        for yi, t in zip(y, rows):
            ms = F["main"][exp][t]["test"]
            m, h = 100 * ms[0], 100 * ci(ms)
            ax.errorbar(m, yi, xerr=h, fmt="none", ecolor=COLOR[FAMILY[t]], elinewidth=1.4, capsize=0, zorder=2)
            ax.plot(m, yi, ls="", marker=MARKER[t], ms=5.5, mfc=COLOR[FAMILY[t]], mec="white", mew=0.8, zorder=3)
            lo.append(m - h)
            hi.append(m + h)
        ax.set_yticks(y, [LABEL[t] for t in rows])
        for yb in (y[2] - 0.5, y[5] - 0.5) if len(rows) == 9 else ():
            ax.axhline(yb, color=GRID, lw=0.5, zorder=0)
        pad = max(0.15, 0.08 * (max(hi) - min(lo)))
        ax.set_xlim(min(lo) - pad, max(hi) + pad)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlabel("test top-1 (%), mean ± 95 % CI")
        ax.set_title(EXP[exp], loc="left")
        ax.tick_params(axis="y", length=0)
    fig.legend(handles=family_handles(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig_main", png, written)


def fig_compare(F, out, png, written, missing):
    """Forest plot: per-seed paired difference A - B with its bootstrap 95 % interval, and the rule's verdict."""
    exps = [e for e in EXP if e in F.get("compare", {})]
    if not exps:
        missing.append("fig_compare")
        return
    pairs = [p for p in F["compare"][exps[0]]]
    fig, axes = plt.subplots(1, len(exps), figsize=(FULL, 0.24 * len(pairs) + 0.9), squeeze=False, sharey=True)
    seed_marker = ["o", "s", "D"]
    for ax, exp in zip(axes[0], exps):
        C = F["compare"][exp]
        y = np.arange(len(pairs))[::-1]
        xmax = 0.0
        for yi, p in zip(y, pairs):
            if p not in C:
                continue
            for j, (seed, d, lo, hi, _) in enumerate(C[p]["rows"]):
                yy = yi + (j - 1) * 0.22
                ax.plot([100 * lo, 100 * hi], [yy, yy], color=COLOR["base"], lw=1.0, zorder=2)
                ax.plot(100 * d, yy, ls="", marker=seed_marker[j % 3], ms=3.6, mfc=COLOR["base"], mec="white",
                        mew=0.5, zorder=3)
                xmax = max(xmax, abs(100 * lo), abs(100 * hi))
        lim = max(0.5, 1.12 * xmax)
        for yi, p in zip(y, pairs):
            if p in C:
                ax.text(lim * 1.04, yi, C[p]["rule"], va="center", ha="left", fontsize=6.5, color=INK2)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.7, len(pairs) - 0.3)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlabel("paired difference in test top-1 (pp), A $-$ B")
        ax.set_title(EXP[exp], loc="left")
        ax.tick_params(axis="y", length=0)
    a, b = zip(*(p.split(":") for p in pairs))
    axes[0][0].set_yticks(np.arange(len(pairs))[::-1], [f"{SHORT[x]} $-$ {SHORT[z]}" for x, z in zip(a, b)])
    handles = [Line2D([], [], ls="-", lw=1.0, color=COLOR["base"], marker=seed_marker[j], ms=3.6,
                      mfc=COLOR["base"], mec="white", label=f"seed {j}") for j in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04 - 0.3 / len(pairs)))
    fig.tight_layout(w_pad=3.0)
    save(fig, out, "fig_compare", png, written)


def fig_robust(F, out, png, written, missing):
    """Gain in test top-1 over B0 under each view: automatic corruptions, then the annotation-assisted views."""
    exps = [e for e in EXP if e in F.get("robustness", {})]
    if not exps:
        missing.append("fig_robust")
        return
    pick = F.get("hypotheses", {}).get("H1", {}).get("tsinghua_main", {})
    tags = []
    for t in (pick.get("best_proposed_by_val"), pick.get("best_control_by_val"), "rr_class", "classattn_all"):
        if t and t not in tags:
            tags.append(t)
    # one row per view, grouped under two header rows (no data on those), so every name reads horizontally
    views0 = [v for v in AUTO + ANN if all(v in F["robustness"][e]["top1"] and "linear" in
                                           F["robustness"][e]["top1"][v] for e in exps)]
    rows = [("automatic views", None)] + [(VIEW[v], v) for v in views0 if v in AUTO]
    if any(v in ANN for v in views0):
        rows += [("annotation-assisted", None)] + [(VIEW[v], v) for v in views0 if v in ANN]
    y = np.arange(len(rows))[::-1].astype(float)
    fig, axes = plt.subplots(1, len(exps), figsize=(FULL, 0.19 * len(rows) + 0.95), squeeze=False, sharey=True)
    for ax, exp in zip(axes[0], exps):
        R = F["robustness"][exp]["top1"]
        for j, t in enumerate(tags):
            pts = [(100 * (R[v][t] - R[v]["linear"]), yi) for (_, v), yi in zip(rows, y) if v and t in R[v]]
            if pts:
                gx, gy = zip(*pts)
                ax.plot(gx, np.asarray(gy) - (j - (len(tags) - 1) / 2) * 0.17, ls="", marker=MARKER[t], ms=4.5,
                        mfc=COLOR[FAMILY[t]], mec="white", mew=0.6, zorder=3)
        for (lab, v), yi in zip(rows, y):
            if v is None and yi < y[0]:
                ax.axhline(yi + 0.5, color=AXIS, lw=0.6, zorder=1)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlabel("gain in test top-1 over B0 (pp)")
        ax.set_title(EXP[exp], loc="left")
        ax.tick_params(axis="y", length=0)
    axes[0][0].set_yticks(y, [lab for lab, _ in rows])
    for tl, (_, v) in zip(axes[0][0].get_yticklabels(), rows):
        if v is None:
            tl.set_fontweight("bold")
            tl.set_color(INK)
    fig.legend(handles=head_handles(tags), loc="lower center", ncol=len(tags),
               bbox_to_anchor=(0.5, -0.02 - 0.25 / fig.get_figheight()))
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig_robust", png, written)


def fig_ksweep(F, out, png, written, missing):
    """Top-1 and shortlist recall of RCEA (rival) against the shortlist size K, on Tsinghua Dogs."""
    K = F.get("k_sweep_detail")
    if not K or len(K) < 2:
        missing.append("fig_ksweep")
        return
    ks = sorted(int(k) for k in K)
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.75, 1.9))
    for ax, key, title in ((axes[0], "top1", "test top-1 (%)"), (axes[1], "recall", "shortlist recall (%)")):
        pts = [(k, K[str(k)][key]) for k in ks if K[str(k)].get(key)]
        if not pts:
            ax.set_visible(False)
            continue
        xs = [k for k, _ in pts]
        m = [100 * v[0] for _, v in pts]
        h = [100 * ci(v) for _, v in pts]
        ax.plot(xs, m, color=COLOR["prop"], lw=1.4, zorder=2)
        ax.errorbar(xs, m, yerr=h, fmt="none", ecolor=COLOR["prop"], elinewidth=1.0, zorder=2)
        ax.plot(xs, m, ls="", marker=MARKER["rr_rival"], ms=5, mfc=COLOR["prop"], mec="white", mew=0.7, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("shortlist size K")
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig_ksweep", png, written)


def fig_perclass(F, out, png, written, missing, n_label=4):
    """Per-breed change in test accuracy, best proposed head minus B0 (mean over seeds), sorted."""
    P = F.get("per_class", {})
    pick = F.get("hypotheses", {}).get("H1", {}).get("tsinghua_main", {}).get("best_proposed_by_val", "rr_rival")
    exps = [e for e in EXP if e in P and pick in P[e] and "linear" in P[e]]
    if not exps:
        missing.append("fig_perclass")
        return
    fig, axes = plt.subplots(len(exps), 1, figsize=(FULL, 2.15 * len(exps) + 0.2), squeeze=False)
    for ax, exp in zip(axes[:, 0], exps):
        names = P[exp]["class_name"]
        d = np.array([np.nan if a is None or b is None else 100 * (a - b)
                      for a, b in zip(P[exp][pick], P[exp]["linear"])], dtype=float)
        ok = ~np.isnan(d)
        order = np.argsort(-d[ok], kind="stable")
        dv = d[ok][order]
        nv = [n for n, k in zip(names, ok) if k]
        nv = [nv[i] for i in order]
        x = np.arange(len(dv))
        ax.bar(x, dv, width=0.62, color=[POS if v > 0 else NEG if v < 0 else AXIS for v in dv], linewidth=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        # the extremes are named under the panel, not on the bars (130 bars leave no room for a label each);
        # equal changes keep the class-index order of the stable sort
        lab = [nv[i].replace("_", " ") + " " + f"{dv[i]:+.1f}".replace("-", "−") for i in range(len(dv))]
        ups = [lab[i] for i in range(len(dv)) if dv[i] > 0][:n_label]
        dns = [lab[i] for i in range(len(dv) - 1, -1, -1) if dv[i] < 0][:n_label]
        lines = ([f"largest gains (pp): {', '.join(ups)}"] if ups else []) + \
                ([f"largest losses (pp): {', '.join(dns)}"] if dns else [])
        if lines:
            ax.text(0, -0.05, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=6,
                    color=INK2, linespacing=1.6)
        span = max(1.0, np.nanmax(np.abs(dv)))
        ax.set_ylim(-1.15 * span, 1.15 * span)
        ax.set_xlim(-1, len(dv))
        ax.set_xticks([])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylabel(f"{SHORT[pick]} $-$ B0 (pp)")
        n_up, n_dn = int((dv > 0).sum()), int((dv < 0).sum())
        ax.set_title(f"{EXP[exp]}: {len(dv)} breeds sorted by change; {n_up} gain, {n_dn} lose, "
                     f"{len(dv) - n_up - n_dn} unchanged", loc="left")
    fig.tight_layout(h_pad=1.4)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=5, mfc=POS, mec="white", label="gain"),
                        Line2D([], [], ls="", marker="s", ms=5, mfc=NEG, mec="white", label="loss")],
               loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0 + 0.12 / fig.get_figheight()))
    save(fig, out, "fig_perclass", png, written)


def fig_curves(F, out, png, written, missing):
    """Validation top-1 per epoch for seed 0 of B0, the class query and the best proposed head (Tsinghua)."""
    Cv = F.get("curves")
    if not Cv:
        missing.append("fig_curves")
        return
    pick = F.get("hypotheses", {}).get("H1", {}).get("tsinghua_main", {}).get("best_proposed_by_val", "rr_rival")
    runs = []
    for t in ("linear", "rr_class", pick):
        k = next((r for r in Cv if r.endswith(f"/{t}_s0") or r == f"{t}_s0"), None)
        if k and t not in [x for x, _ in runs]:
            runs.append((t, k))
    if not runs:
        missing.append("fig_curves")
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.8, 2.0))
    for t, k in runs:
        a = np.asarray(Cv[k], dtype=float)
        for ax, col in ((axes[0], 2), (axes[1], 1)):
            ax.plot(a[:, 0] + 1, 100 * a[:, col] if col == 2 else a[:, col], color=COLOR[FAMILY[t]], lw=1.2,
                    ls="-" if FAMILY[t] != "ctrl" else (0, (4, 1.5)))
            ax.plot(a[-1, 0] + 1, 100 * a[-1, col] if col == 2 else a[-1, col], ls="", marker=MARKER[t], ms=5,
                    mfc=COLOR[FAMILY[t]], mec="white", mew=0.7)
    axes[0].set_title("validation top-1 (%)", loc="left")
    axes[1].set_title("training loss", loc="left")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.legend(handles=head_handles([t for t, _ in runs]), loc="lower center", ncol=len(runs),
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig_curves", png, written)


def fig_explain(explain, images, out, png, written, missing, max_rows=6):
    """Attention of the two strongest final candidates, re-plotted from explain.py's saved maps."""
    if not explain or not images or not Path(explain).exists() or not Path(images).exists():
        missing.append("fig_explain")
        return
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    from dogbreed.datasets.sources import ImageSource
    from dogbreed.datasets.transforms import default_resize
    J = json.loads(Path(explain).read_text(encoding="utf-8"))
    grid, size = int(J["grid"]), int(J["img_size"])
    crop = T.Compose([T.Resize(default_resize(size), interpolation=InterpolationMode.BICUBIC), T.CenterCrop(size)])
    src = ImageSource(images)
    # the same number of rows of each kind, in the order explain.py chose them (first by id): no hand-picking
    kinds = []
    for e in J["images"]:
        if e["kind"] not in kinds:
            kinds.append(e["kind"])
    per = max(1, max_rows // max(1, len(kinds)))
    shown = [e for k in kinds for e in [x for x in J["images"] if x["kind"] == k][:per]]
    tags = list(shown[0]["tags"]) if shown else []
    ncol = 1 + 2 * len(tags)
    name = lambda s: textwrap.fill(s.replace("_", " "), 22)                     # whole names, wrapped
    rank = ("1st", "2nd")
    fig, axes = plt.subplots(len(shown), ncol, figsize=(FULL, FULL / ncol * 1.15 * len(shown) + 0.3),
                             squeeze=False)
    im = None
    for r, e in enumerate(shown):
        img = np.asarray(crop(src.open(e["key"]).convert("RGB")))
        axes[r, 0].imshow(img)
        head = "input, true breed\n" if r == 0 else ""                          # column headers on row 1 only
        axes[r, 0].set_title(head + name(e["true"]), fontsize=5.8, fontweight="normal", loc="left")
        axes[r, 0].set_ylabel(KIND.get(e["kind"], e["kind"]), fontsize=6, color=INK2, labelpad=3)
        for j, tag in enumerate(tags):
            T_ = e["tags"][tag]
            logits = np.asarray(T_["final_logit"], dtype=float)
            best = np.argsort(-logits)[:2]
            for c, b in enumerate(best):
                ax = axes[r, 1 + 2 * j + c]
                m = np.array([int(ch, 16) for ch in T_["maps"][b]], dtype=float).reshape(grid, grid) / 15.0
                ax.imshow(spotlight(img, m))
                im = True
                mark = "✓" if T_["cand"][b] == e["true"] else "✗"
                head = f"{LABEL.get(tag, tag)}: {rank[c]}\n" if r == 0 else ""
                ax.set_title(f"{head}{name(T_['cand'][b])} {mark}", fontsize=5.8, fontweight="normal", loc="left")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    if im is not None:                     # its own axes, added after the layout, so the grid keeps its size
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        f = SPOT_FLOOR
        key = LinearSegmentedColormap.from_list("spot", [(f, f, f), (1, 1, 1)])
        bottom = min(ax.get_position().y0 for ax in axes[-1])
        cax = fig.add_axes([0.3, bottom - 0.16 / fig.get_figheight(), 0.4, 0.07 / fig.get_figheight()])
        cb = fig.colorbar(ScalarMappable(Normalize(0, 1), key), cax=cax, orientation="horizontal")
        cb.set_label("attention sets the brightness (each map scaled to its own maximum); "
                     "✓ the true breed, ✗ another breed", fontsize=6, color=INK2)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_edgecolor(AXIS)
        cb.outline.set_linewidth(0.5)
    save(fig, out, "fig_explain", png, written)


SPOT_FLOOR = 0.3                          # low attention dims the image to 30 %, so the dog stays recognisable


def spotlight(img, m, floor=SPOT_FLOOR):
    """The image dimmed where attention is low: brightness = floor + (1 - floor) * map, the map upsampled
    bilinearly from the token grid to the image. Keeps the attended evidence in its own colours."""
    from PIL import Image
    a = np.asarray(Image.fromarray(np.uint8(np.round(255 * m))).resize(img.shape[1::-1], Image.BILINEAR),
                   dtype=float) / 255.0
    return (img.astype(float) * (floor + (1 - floor) * a[..., None])).clip(0, 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="paper/generated", help="folder with figdata.json")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--explain", default="", help="shown_maps.json written by scripts/explain.py")
    ap.add_argument("--images", default="", help="the image folder or zip the maps were made from")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)
    F = json.loads((Path(a.gen) / "figdata.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    written, missing = [], []
    fig_main(F, out, a.png, written, missing)
    fig_compare(F, out, a.png, written, missing)
    fig_robust(F, out, a.png, written, missing)
    fig_ksweep(F, out, a.png, written, missing)
    fig_perclass(F, out, a.png, written, missing)
    fig_curves(F, out, a.png, written, missing)
    fig_explain(a.explain, a.images, out, a.png, written, missing)
    print(f"{len(written)} figures -> {out}: {', '.join(written)}; missing: {', '.join(missing) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
