# Core literature dossier — fine-grained dog breed classification

Started 2026-11-08. Written incrementally; every bibliographic fact checked against a primary source
(Crossref API, Semantic Scholar API, arXiv abs page, CVF open access, publisher page). Anything not confirmed is
marked **UNVERIFIED**. For every number: where it came from (table / section) and whether it was read in the
full text or only in the abstract.

Legend for "read in": FT = full text read; ABS = abstract / metadata only.

---

## Part 1 — the four named papers

### 1.1 Head-Body Guided Deep Learning Framework for Dog Breed Recognition (CMC 2025)

| Field | Verified value | Source |
|---|---|---|
| Title | Head-Body Guided Deep Learning Framework for Dog Breed Recognition | Crossref, publisher PDF |
| Authors | Noman Khan, Afnan, Mi Young Lee, Jakyoung Min (4 authors). **Crossref lists only 3** (it drops the mononymous author "Afnan"); the PDF and the publisher's citation meta tags list 4. Cite with 4. | PDF p. 2935; techscience.com meta tags; Crossref |
| Venue | *Computers, Materials & Continua* (CMC), Tech Science Press, vol. 85, no. 2, pp. 2935–2958, 2025 (published 23 Sep 2025; received 13 Jun, accepted 5 Aug 2025). ISSN 1546-2218 / 1546-2226. CC BY 4.0. | Crossref + PDF |
| DOI | 10.32604/cmc.2025.069058 — **resolves** (doi.org 302 → https://www.techscience.com/cmc/v85n2/63849) | curl, 2026-11-08 |
| Read in | **FT** (publisher PDF, all 24 pages) | |

**Architecture (FT, Sec. 3.2, 4.1).** Two stages. (1) YOLOv8n, COCO-pretrained, fine-tuned for two classes (head, body)
at 640×640, 50 epochs, batch 16, early stopping (patience 10); detector data split 80/20 train/val. (2) Two ViT-B/16
backbones (224×224, 16×16 patches, 12 layers, 12 heads, 768-d; "each pre-trained on ImageNet", variant
(1k vs 21k) **not stated**), heads replaced by identity; head-crop CLS ⊕ body-crop CLS = 1536-d → FC layers with
BN/ReLU/dropout → softmax over 130 breeds. Adam, lr 3e-5, 50 epochs, early stopping, cross-entropy.

**Claimed contribution.** "Modular, lightweight head-body detection pipeline" + "region-aware ViT" + an
augmentation/training scheme (HSV, flips, rotation, noise, random erasing; mixup/mosaic for the detector).

**Protocol — what is and is not stated (FT).**
- **Ground-truth boxes at test time? Not stated either way.** The Fig. 1 caption describes inference as image →
  YOLOv8n → head and body crops → two-stream ViT, so the *intended* pipeline uses a detector, not GT boxes. But the
  classifier is trained on "the dog's body and head crops" without saying whether those crops are GT or detected, and
  no sentence says which crops the reported accuracy was computed on. The ablations on "full-body" and "head" crops
  (Tables 4–5) are equally unspecified. Treat the 90.04 % as **possibly GT-crop based (unverifiable)**.
- **Split not stated.** The official Tsinghua split (65,228 train / 5,200 val) is never mentioned; no "test set" is
  named; the only split given is the detector's 80/20. Sec. 4.6.6 mentions a 10-fold cross-validation for paired
  t-tests (all p < 0.05) without fold details. The detector's 80/20 split may overlap the classifier's evaluation images.
- Input 224×224 per stream; ImageNet normalisation; no TTA mentioned. No code; "Data available on request from the
  authors".
- Oxford-IIIT Pet (Sec. 4.6.5): 7,349 images, 37 breeds (25 dog, 12 cat); there they use the head + the *full image
  with background* (not the body). Split not stated.

**Reported numbers (FT).**

| Setting (Tsinghua Dogs, 130 breeds) | Backbone | Acc. | Where |
|---|---|---|---|
| Two-stream head + body (proposed) | 2× ViT-B/16 @224 | **90.04 %** (P 0.8973, R 0.8350, F1 0.8627) | Sec. 4.4.2, Fig. 5a, Table 6 |
| Two-stream head + body | Xception | 88.16 % | Table 6 |
| Two-stream head + body | EfficientNet-B0 | 85.13 % | Table 6 |
| Single stream, original image | ViT | 82.54 % | Table 3 |
| Single stream, body crop | ViT | 86.63 % | Table 4 |
| Single stream, head crop | ViT | 86.74 % | Table 5 |
| "SOTA" baselines (copied from Zou et al. 2020 Table 3, not re-run) | Inception-v3 / ResNet-50 | 77.66 / 83.52 (PMG) / 83.70 (TBMSL-Net) / 86.40 (WS-DAN) | Sec. 4.5 |
| Oxford-IIIT Pet, head + full image (proposed) | 2× ViT | **96.85 %** (P 0.9681, R 0.9285, F1 0.9470) | Table 7, abstract |
| Detector AP50 head / body / overall | YOLOv8n @640 | 0.9723 / 0.9884 / 0.9804 (AP50-95 0.8164 / 0.8978 / 0.8571) | Table 2 |

(The pdftotext extraction of Tables 3–7 shifts the row labels by one row; the values above were re-aligned using the
paper's own prose in Sec. 4.6.1–4.6.5, which names each model's numbers.)

**Model size / FLOPs.** Not reported. (For scale only: one ViT-B/16 has 86 M parameters per Dosovitskiy et al.,
Table 1, so the classifier is roughly 2 × 86 M plus the detector — our arithmetic, not the paper's.)

**Weaknesses.** (i) GT-vs-detector crops at evaluation and the evaluation split are both unstated, so 90.04 % is not
comparable to anything; (ii) the "SOTA" comparison copies 2020 numbers obtained with ResNet-50/Inception-v3 on full
images and sets them against an ImageNet-pretrained ViT-B on crops — backbone, pretraining and input all differ;
(iii) no Stanford Dogs / CUB result; (iv) macro recall 0.835 vs accuracy 0.900 suggests the tail breeds are much worse
(the long tail of Tsinghua Dogs is not analysed); (v) no code, no variance over seeds, no FLOPs.
**Relevance to us: high (it is the dog-specific SOTA claim a reviewer will cite).** Reproduce a head+body two-crop
baseline under the official split, with **detector** crops as the real result and **GT-box** crops reported separately
as an oracle — and state that their 90.04 % is not reproducible as published.

---

### 1.4 Tsinghua Dogs — Zou, Zhang, Mu, Zhang, CVM 2020

| Field | Verified value | Source |
|---|---|---|
| Title | "A new dataset of dog breed images and a benchmark for fine-grained classification" (PDF title; Crossref stores "finegrained" without the hyphen — a line-break artefact; use the hyphen) | PDF; Crossref |
| Authors | Ding-Nan Zou, Song-Hai Zhang (corresponding), Tai-Jiang Mu, Min Zhang (the dataset web page misspells "Ming Zhang") | PDF; Crossref; web page |
| Venue | *Computational Visual Media* 6(4): 477–487, December 2020 (received 18 May 2020, accepted 14 Jun 2020, first online 1 Oct 2020). Tsinghua University Press (with Springer at the time). Article CC BY 4.0. | Crossref; PDF |
| DOI | 10.1007/s41095-020-0184-6 — **resolves** (doi.org 302 → IEEE Xplore document 10897337; the journal is now hosted on IEEE Xplore; the Springer PDF link still works) | curl, 2026-11-08 |
| Read in | **FT** (both the author PDF at cg.cs.tsinghua.edu.cn and the Springer PDF; same text, different layout) | |

**Size.** 70,428 images, 130 breeds, one dog per image, ≥ 200 images per breed, max 7,449 (Teddy); "over 65 %"
from everyday life (Sec. 1). Images have no fixed resolution; minimum side 60 px; at least half are higher-resolution
than Stanford Dogs (Sec. 3.3). The site offers low-res (2.5 GB) and high-res (38.8 GB) versions.

**Sources of the images (Sec. 3.1, 3.3) — the "about 18,000 from Stanford Dogs" claim is CONFIRMED.**
Sec. 3.1: "We integrated 18,000 pictures with only one dog from Stanford Dogs into our dataset", plus > 6,000 crawled
via Baidu image search. Sec. 3.3: about 46,000 images taken in Chinese cities, 18,000 from Stanford Dogs, 6,000 from
Baidu, Google and other search engines. De-duplication by SSIM; breed labels by expert review; blurred, deformed,
occluded or too-small dogs removed.
*Our inference (arithmetic, not stated by the authors):* Stanford Dogs' training split has only 12,000 images, so
18,000 Stanford images **must include ≥ 6,000 Stanford test images**. The paper never says which Stanford split they
came from.

**Official split (Sec. 4.1, FT).** Training and validation labels are provided, the validation set "randomly
selecting 40 images for each breed", giving **65,228 train and 5,200 validation** images. 130 × 40 = 5,200;
65,228 + 5,200 = 70,428 ✓. There is **no separate test set**: the 5,200-image "validation" split is the benchmark's
test set. Split files: `TrainValSplit.zip` (0.2 MB) on https://cg.cs.tsinghua.edu.cn/ThuDogs/ (not downloaded here;
count its lines when the data are fetched).

**How the boxes were made (Sec. 3.2, FT).** Tight boxes for the whole body and the head ("including the ears").
Active-learning loop: (1) 2,000 random images labelled by hand; (2) train RetinaNet ("RetianNet" in the paper) and
auto-annotate 2,000 new images; (3) correct the low-confidence ones with a keyboard-only adjustment tool, add them to
the training data, repeat until all are labelled. Throughput 1,500–2,500 images/h vs 300–500 by hand. So the boxes
are **semi-automatic (RetinaNet proposals + human correction of low-confidence cases)**; high-confidence boxes were
not necessarily checked by a person.

**Benchmark (Table 3, FT) — Tsinghua Dogs, official split; no method uses the GT boxes; pretraining not stated.**

| Model | Backbone | Input | Batch | Epochs | Top-1 |
|---|---|---|---|---|---|
| Inception V3 | — | 224×224 | 64 | 200 | 77.66 % |
| WS-DAN | Inception (v3 per web page) | 512×448 | 12 | 80 | **86.404 %** |
| PMG | ResNet-50 | 448 centre crop of 550 | 16 | 200 | 83.52 % |
| TBMSL-Net | ResNet-50 | 448 (object, raw) + 224 (parts) | 6 | 200 | 83.7 % |

Table 2 (PMG, ResNet-50): CUB 88.609 % single / 89.454 % combined (PMG paper reported 88.9 / 89.6); Stanford Dogs
84.674 % / 86.515 %; Tsinghua Dogs 81.98 % / 83.52 %. **No "with boxes" vs "without boxes" experiment is reported**,
despite the boxes being the dataset's selling point.

**Cross-dataset test and the only overlap statement (Sec. 4.2.2, FT).** WS-DAN trained on Stanford Dogs vs on
Tsinghua Dogs, tested on a 657-image subset of the Tsinghua validation split: the 20 largest breeds after images
"appeared in the training set of Stanford Dogs are excluded". Tsinghua-trained 82.65 %, Stanford-trained 58.14 %.
This sentence implies the **official validation split contains images from the Stanford Dogs training set**.
Nothing is said about ImageNet overlap. Sec. 2.2 states ImageNet-1K holds about 116,000 pictures of 117 dog breeds.

**Terms.** Abstract: "available for academic purposes". The web page gives no licence text beyond a citation request.
The CC BY 4.0 licence covers the *article*, not the images.

**Consequences for our protocol.** (1) Tsinghua val ⊃ Stanford images ⊂ ImageNet ⇒ an ImageNet-pretrained backbone
may have seen part of the Tsinghua *test* split; report accuracy separately on Stanford-origin and non-Stanford val
images (needs a provenance match, e.g. perceptual hashing against Stanford Dogs). (2) Never train on Tsinghua and test
on Stanford (or vice versa) without de-duplication: ≥ 6,000 Stanford *test* images sit inside Tsinghua. (3) The val
split is balanced (40/breed) while train is long-tailed (≥ 160 … ~7,400 per breed): report macro / per-class accuracy.

---

### 1.2 Context-Semantic Quality Awareness Network for Fine-Grained Visual Categorization (CSQA-Net, Pattern Recognition 2026)

| Field | Verified value | Source |
|---|---|---|
| Title | Context-Semantic Quality Awareness Network for fine-grained visual categorization | Crossref |
| Authors (journal) | Qin Xu, Sitong Li, Jiahui Wang, Bo Jiang, Bin Luo, Jinhui Tang (6). arXiv v1 lists 5 (no Bin Luo). Affiliations: Anhui University; Tang at Nanjing Univ. of Science and Technology (OpenAlex) | Crossref; Semantic Scholar; OpenAlex; arXiv API |
| Venue | *Pattern Recognition* vol. 170, article 112033, issue dated Feb 2026 (online 2 Jul 2025 per OpenAlex). Elsevier, paywalled. | Crossref; S2 |
| DOI | 10.1016/j.patcog.2025.112033 — **resolves** (302 → linkinghub.elsevier.com/retrieve/pii/S0031320325006934) | curl |
| Preprint | arXiv:2403.10298 v1, 15 Mar 2024 (only version) | arXiv API |
| Read in | **FT of arXiv v1 only.** The journal full text could not be read (ScienceDirect returns 403 to automated access; no abstract in Crossref/S2/OpenAlex). Any change between arXiv v1 and the journal version (e.g. added datasets) is **UNVERIFIED**. | |

**Architecture (FT arXiv, Sec. III).** Backbone ResNet-50 or Swin-B, "pre-trained on ImageNet" (Swin-B 1k vs 22k not
stated). Image branch: Multi-Level Semantic Quality Evaluation (MLSQE) — ECA channel attention on the last A = 3
stages, conv projection to C = 1024, GMP, a concatenated multi-scale vector; each vector goes to a "quality probing"
(QP) classifier = main classifier with depth-increasing label smoothing (0.7 … 1.0) + a stop-gradient linear probe
re-initialised every τ epochs, whose confidence gap re-weights the per-sample loss. Part branch (training only):
Part Navigator (NTS-Net-style anchors {48, 96, 192}, 3 scales, "scale-aware enhancement" blocks, NMS, top N = 4 crops
resized to half resolution, fed through the shared backbone) → MPMSCA cross-attention: queries = global ⊕ part
tokens, keys/values = multi-scale part tokens, top-v (v = 3) sparse mask, 16 heads. **At test time the part branch
and the auxiliary probes are removed**; prediction = sum of the main classifiers of the last stages + fusion vector.

**Claimed contribution.** (i) assessing feature "quality" in real time with stop-gradient probes; (ii) MPMSCA
multi-part multi-scale cross-attention; (iii) part navigator for scale confusion.

**Protocol (FT arXiv, Sec. IV-A/B).** CUB-200-2011 (5,994 / 5,794), Stanford Cars (8,144 / 8,041), FGVC-Aircraft
(6,667 / 3,333), NABirds (23,929 / 24,633), standard splits; 448×448; random crop + flip, centre crop at test; SGD,
lr 0.0075 (0.01 Cars), cosine, 50 epochs incl. 5 warm-up, batch 16; image labels only — **no boxes or part
annotations at training or test time. No dog dataset (neither Stanford Dogs nor Tsinghua Dogs).**

**Reported numbers (FT arXiv, Tables II–V, top-1).**

| Dataset | CSQA-Net ResNet-50 | CSQA-Net Swin-B | R50 baseline (Table VI) |
|---|---|---|---|
| CUB-200-2011 | 90.5 | 92.6 | 85.4 |
| Stanford Cars | 95.6 | 95.6 | 92.7 |
| FGVC-Aircraft | 94.5 | 94.7 | 90.3 |
| NABirds | 90.0 | 92.3 | 84.6 |

(Values taken from `pdftotext -table` output, which aligns the rows correctly, and cross-checked against the prose of
Sec. IV-C.) **Model size / FLOPs: not reported**; Sec. III-D only says inference cost is "slightly higher" than the
backbone. No code link in the preprint.

**Weaknesses.** Gains over strong 2023 methods are 0.0–0.6 points at a fixed backbone, without variance over seeds;
many hand-set hyper-parameters (label-smoothing ladder, τ, ε, N, v); Swin-B pretraining source unstated; no dogs.
**Relevance to us: medium-low.** A recent strong weakly supervised FGVC baseline at 448 px. Its cross-attention is
class-agnostic (queries come from global/part features, never from class embeddings or a candidate set) — **not a
novelty threat** to rival-conditioned attention. Worth one row in the related-work table and possibly as a baseline
if its code becomes available.

---

### 1.3 Enhancing Fine-Grained Image Classification Through Attentive Batch Training (RBI, ICT Express 2026)

| Field | Verified value | Source |
|---|---|---|
| Title | Enhancing fine-grained image classification through attentive batch training | Crossref; DOAJ |
| Authors | Duy M. Le, Bao Q. Bui, Anh Tran, Cong Tran, Cuong Pham (corresponding). Journal affiliations: Post & Telecommunication Institute of Technology (Hanoi), VinUniversity, Qualcomm AI Research | Crossref; DOAJ |
| Venue | *ICT Express* (KICS / Elsevier) vol. 12, no. 2, pp. 319–323, April 2026 (online Oct 2025). Gold OA, CC BY 4.0. A 5-page letter. | Crossref; DOAJ; S2 |
| DOI | 10.1016/j.icte.2025.10.002 — **resolves** (302 → linkinghub.elsevier.com/retrieve/pii/S2405959525001560) | curl |
| Preprint | arXiv:2412.19606 v1, 27 Dec 2024 (only version) | arXiv API |
| Read in | **FT of arXiv v1; the journal version only as abstract (DOAJ, OpenAlex)**. ScienceDirect returns 403 to automated access, so the journal's tables were not read. | |

**Architecture (FT arXiv).** "Relationship Batch Integration" (RBI): backbone embeddings N ∈ R^{B×D} for the whole
batch; a "Relationship Position Encoding" S ∈ R^{B×B} = normalised **PSNR between raw images** of the batch; a
"Residual Relationship Attention" (RRA) where every image attends to every other image in the batch (Q, K, V from
the batch embeddings, S added to K and V), gated residual with the image's own feature, BatchNorm, FC classifier.

**Protocol (FT arXiv).** 224×224; ImageNet-1K-pretrained backbones (MobileNetV3, DenseNet, Swin-T/B, ConvNeXt-B/L,
HERBS, TransFG, P2P-Net); 50 epochs, batch 32, RAdam, lr 1e-5; "top-1 classification error on the shuffled
validation set". **Inference is transductive**: the RRA mixes features of the *other test images in the batch*
(Fig. 3 varies the test batch size 1 … 120 and reports small effects). No boxes. Stanford Dogs split not stated
explicitly (presumably the standard 12,000 / 8,580); CUB, NABirds; Tiny-ImageNet.

**Reported numbers.**

| Source | Stanford Dogs | Others |
|---|---|---|
| arXiv v1 Table 1 (FT) | best **95.79 %** = ConvNeXt-Large-RBI (231.8 M params; ConvNeXt-Large baseline 93.71 %); Swin-B-RBI 93.06 %; TransFG-RBI 89.76 %; HERB-SwinT-RBI 88.9 % | CUB best 91.07 % (P2P-Net-RBI); NABirds best 90.61 % (HERB-SwinT-RBI); Tiny-ImageNet 93.71 % (abstract) |
| Journal abstract (ABS, DOAJ/OpenAlex) | **97.79 %** "on the Stanford Dog dataset" | Tiny-ImageNet 93.71 % |

**The two Stanford Dogs figures disagree by exactly 2 points (95.79 vs 97.79) — UNVERIFIED which is right** without
the journal's table. The arXiv paper is also internally inconsistent: the abstract claims average gains of +2.78 %
(CUB) and +3.83 % (Dogs), the text says +2.78 / +3.83 / +3.29 for Dogs / CUB / NABirds, and Table 1's
"Avg. Improvement" row gives +2.51 (Dogs) / +3.46 (CUB) / +3.04 (NABirds). Parameters and inference times per model
are in Table 1 (e.g. ConvNeXt-Large-RBI 231.8 M, 1.23 s); FLOPs not given.

**Weaknesses.** (i) Transductive batch inference — a test prediction depends on which other test images share its
batch; not comparable to per-image classifiers. (ii) ImageNet-1K pretraining with Stanford Dogs test images inside
ImageNet-1K train (Part 2) makes 93–98 % on Stanford Dogs a contaminated number; the paper does not discuss it.
(iii) Unequal training recipe for all baselines (they note their CUB/NABirds numbers "diverge from those in the
literature"). (iv) Inconsistent averages; 95.79 vs 97.79. (v) PSNR between raw pixels as a "relationship" signal is
ad hoc. **Relevance: medium as a cautionary example** (a headline Stanford Dogs number produced under the protocol we
will argue against) and as related work on cross-image attention; **not a novelty threat** (it conditions on the
batch, not on candidate classes).

---

## Part 2 — dataset facts

### 2.1 Stanford Dogs (Khosla, Jayadevaprakash, Yao, Fei-Fei, FGVC workshop @ CVPR 2011)

**Exact citation (dataset page + the authors' own BibTeX, FT).** Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng
Yao, Li Fei-Fei. "Novel Dataset for Fine-Grained Image Categorization." *First Workshop on Fine-Grained Visual
Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Colorado Springs, CO,
June 2011. The PDF's title line adds ": Stanford Dogs". **No DOI** (not in Crossref; it is a 2-page workshop
abstract). The dataset page asks for this as the *primary* citation and **ImageNet (Deng et al., CVPR 2009)** as the
*secondary* one. (vision.stanford.edu was unreachable on 2026-11-08; page read via the Wayback snapshot of
12 Oct 2026; paper read at people.csail.mit.edu/khosla/papers/fgvc2011.pdf.)

| Fact | Value | Source (read in) |
|---|---|---|
| Classes | 120 | dataset page; paper (FT) |
| Images | 20,580 (page). The paper's intro says "over 22,000" — the page and the paper's own Table 1 give 20,580; use 20,580. | page; paper Table 1 |
| Split | 100 images per class for training, the rest (≥ 50 per class) for testing ⇒ **12,000 train / 8,580 test**; fixed split files `train_list.mat` / `test_list.mat` | paper Sec. 3; README; the 12,000 / 8,580 counts are also printed in Zou et al. 2020 Sec. 4.1, Cui et al. 2018 Table, Sun et al. 2018 Table 1 |
| Provenance | "The images and bounding boxes were downloaded from ImageNet"; classes = leaf nodes under 'Canis familiaris'; images ≥ 200×200; duplicates within/between classes removed | paper Sec. 2 (FT); page: "built using images and annotation from ImageNet" |
| Annotations | class label + one bounding box per dog (ImageNet boxes, verified on AMT) | paper Sec. 2 |
| Terms of use | **No licence of its own**; it inherits ImageNet's terms (below) | page (no licence text) |

**Citable sources that Stanford Dogs (test split included) is inside the ImageNet-1k training set** — all FT:

| Source | Exact wording (short quote) | Where |
|---|---|---|
| **Krause, Sapp, Howard, Zhou, Toshev, Duerig, Philbin, Fei-Fei. "The Unreasonable Effectiveness of Noisy Data for Fine-Grained Recognition." ECCV 2016, LNCS, pp. 301–320. DOI 10.1007/978-3-319-46487-9_19** | "since Stanford Dogs is a subset of ILSVRC training data" — they therefore pre-train on ILSVRC "with dog data removed" | Sec. 5.1 (Experimental setup), arXiv 1511.06789 |
| Murabito, Spampinato, Palazzo, Giordano, Pogorelov, Riegler. "Top-down saliency detection driven by visual classification." CVIU 172: 67–76, 2018. DOI 10.1016/j.cviu.2018.03.005 | "since the whole Stanford Dogs is included in ImageNet", public VGG-19 / Inception models "would have the advantage of having been trained on images included in" their (Stanford-Dogs-derived) test set; they pre-train on ImageNet minus the 120 classes | Sec. 5 (experimental setup), arXiv 1709.05307 |
| Zhang, Tang, Jia. "Fine-Grained Visual Categorization Using Meta-learning Optimization with Sample Selection of Auxiliary Data." ECCV 2018, LNCS, pp. 241–256. DOI 10.1007/978-3-030-01237-3_15 | using ImageNet for Stanford Dogs "is inappropriate because the dataset of Stanford Dogs is a subset of ImageNet"; ResNet-34 pre-trained on ImageNet without the 120 dog classes gives 69.3 % (their method 73.2 %) | Sec. 5.4, arXiv 1807.10916 |

**Recommended citation for the paper: Krause et al. 2016** (it says "ILSVRC training data" explicitly, and its author list
includes Fei-Fei Li, a co-author of both Stanford Dogs and ImageNet). None of the three gives an image count of the
overlap; the Stanford Dogs file names keep their ImageNet ids (`n02085620_XXXX.jpg`), so the exact overlap with the
ILSVRC-2012 train list can be counted by file name — **do this and report the number** (not done here; no data
downloaded). *Magnitude of the effect, for motivation:* MetaFGNet's 69.3 % (ResNet-34, dog classes removed from
pre-training) vs the 90–98 % routinely reported with full-ImageNet pre-training — confounded by backbone and
resolution, so cite it as an indication only.

### 2.2 ImageNet terms of access (governs Stanford Dogs)

From image-net.org/download.php (Wayback snapshot 3 Oct 2026; live site unreachable 2026-11-08): the researcher
"shall use the Database only for non-commercial research and educational purposes"; Princeton and Stanford give no
warranties (incl. non-infringement); the researcher indemnifies them against claims, "including … use of any copies of
copyrighted images"; access may be shared only with colleagues who accept the same terms; an employer that is a
commercial entity is also bound; New Jersey law. ⇒ Stanford Dogs: non-commercial research only, images remain
copyrighted by their owners, no redistribution.

### 2.3 Tsinghua Dogs terms

Paper abstract: "available for academic purposes"; the web page adds only a citation request. No licence file was
found. The CC BY 4.0 notice in the PDF covers the article, not the images. ⇒ academic use; ask the authors before
redistributing anything (e.g. crops or a de-duplicated subset list is fine to publish as *file names*, not images).

### 2.4 Candidate independent test sets

| Dataset | Size / breeds | Licence | Overlap risk with ImageNet / Stanford Dogs | Source (read in) |
|---|---|---|---|---|
| **Oxford-IIIT Pet** (Parkhi, Vedaldi, Zisserman, Jawahar, CVPR 2012, pp. 3498–3505, DOI 10.1109/CVPR.2012.6248092) | 7,349 images, 37 breeds (**25 dog**, 12 cat); ~200/breed split 50 train / 50 val / ~100 test; totals 1,846 / 1,834 / **3,669** test; head box + trimap per image | **CC BY-SA 4.0**, "commercial/research purposes"; image copyright stays with owners | Images from Catster, Dogster, Flickr groups, Google Images (not from ImageNet), but **near-duplicates exist: 58 Pets test images have a near-duplicate in ILSVRC-2012 train, 80 in ImageNet-21k, 600 in JFT** (BiT, Kolesnikov et al., ECCV 2020, Appendix C, Table 7). Most of its 25 dog breeds are also Stanford Dogs / ImageNet classes; Havanese, Shiba Inu, American Bulldog and American Pit Bull Terrier have no Stanford Dogs class (our reading of the two class lists — re-check when mapping). | paper Sec. 2, Table 1 (FT); VGG page (FT); BiT App. C (FT) |
| **ImageNetV2** (Recht, Roelofs, Schmidt, Shankar, ICML 2019, PMLR 97: 5389–5400) | 10 new images per ImageNet class per variant (10,000 total) ⇒ ~1,200 images for the 120 Stanford Dogs synsets, ×3 variants (MatchedFrequency, Threshold0.7, TopImages) | images from Flickr; check the release's terms before use (**UNVERIFIED**) | Collected from Flickr uploads after the original ImageNet period and **near-duplicates removed against all of the original ImageNet** (App. C.1.1) ⇒ the cleanest *ImageNet-free* test set for the 120 Stanford breeds; labels follow the ImageNet MTurk process (noisier); only 10/class. | paper App. C (FT) |
| **Tsinghua Dogs, non-Stanford part** | ~52,000 images (46k user photos + 6k web) of 130 breeds; the val split holds 40/breed of mixed origin | academic use | User photos from Chinese cities are not from ImageNet; the 18k Stanford images are. Needs a provenance match (perceptual hash vs Stanford Dogs) before use as a clean test set. | Zou et al. 2020 (FT) |
| Dogs-in-the-Wild (Sun, Yuan, Zhou, Ding, "Multi-Attention Multi-Class Constraint…", ECCV 2018, pp. 834–850, DOI 10.1007/978-3-030-01270-0_49) | 299,458 images, 362 categories; 258,474 train / 40,984 test | "will be released upon acceptance"; a third-party mirror lists a non-commercial licence — official release **UNVERIFIED** | Built *without* ImageNet: "we exclude any image of ImageNet from our dataset" (Sec. 4). Overlap with web-crawled Tsinghua/Oxford images not checked. | paper Sec. 4, Table 1 (FT) |
| Columbia Dogs (Liu, Kanazawa, Jacobs, Belhumeur, ECCV 2012, pp. 172–185, DOI 10.1007/978-3-642-33718-5_13) | 133 breeds, 8,351 images, 8 part labels per image (ABS) | not found | A secondary source (Lai et al., arXiv 2007.11986) says the images come from Google, **ImageNet** and Flickr — **high overlap risk; UNVERIFIED in the original** | abstract via Springer/LNCS page (ABS) |
| Kaggle "Dog Breed Identification" (2017) | 120 breeds | competition rules | It is Stanford Dogs re-split ⇒ **not independent** (UNVERIFIED here; widely stated) | — |

**Verdict for Part 2.** Use ImageNetV2's 120 dog classes (and, after de-duplication, the dog breeds of Oxford-IIIT Pet)
as ImageNet-free test sets for Stanford-breed models; use the non-Stanford part of Tsinghua Dogs' val split as the
cleaner Tsinghua test.

---


## Part 3 — novelty search for "rival-conditioned evidence attention" (RCEA)

**The mechanism under test.** Backbone patch tokens; a global classifier proposes top-K candidates; for each candidate c
the query is q_c = e_c − ē_{rivals(c)} (mean or probability-weighted embedding of the other K − 1 candidates); q_c
cross-attends over the patch tokens; pooled evidence z_c; score s_c = ⟨q_c, z_c⟩ added to the global logit of c.
Controls planned: q_c = e_c (class-specific), q_c = e_c − mean of all classes (globally centred), random rivals.

Closeness scale: **high** = same pipeline and same conditioning signal; **medium** = same pipeline or same idea, but a
different conditioning signal or a different stage (loss, explanation, zero-shot); **low** = shares one ingredient.

### 3.1 Closest work found so far

**Self Assessment Classifier (SAC)** — Tuong Do, Huy Tran, Erman Tjiputra, Quang D. Tran, Anh Nguyen.
"Fine-Grained Visual Classification using Self Assessment Classifier." *2024 IEEE Conference on Artificial
Intelligence (CAI)*, pp. 597–602, DOI 10.1109/CAI59869.2024.00117 (Crossref); arXiv 2205.10529 (May 2022);
code github.com/aioz-ai/SAC. **Read in: FT (arXiv v1).**
- What it does (Sec. 3): the backbone classifier gives the **top-k candidates (k = 10)**; each candidate's *label name*
  is embedded (word embeddings → GRU, learned from scratch, 1024-d); a low-rank bilinear attention between every
  feature-map location and every candidate embedding gives an attention map M ∈ R^{f×k} (one softmax over all
  location–candidate couples) and **one** joint vector J = Σ M_ij (T_u ×1 F_i) ×2 E_j; a second ("fine-grained")
  classifier on J re-predicts; final = λ·Pr₁ + (1 − λ)·Pr₂ (λ = 0.5). A "Dropping" step zeroes the locations most
  correlated with the candidate classes (as training augmentation), and Algorithm 1 crops the image at test time with
  the summed map.
- Numbers (FT, Table 2): average gains +1.3 (CUB), +1.2 (Stanford Dogs), +1.2 (Aircraft) over 7 backbones/methods;
  k-ablation Table 5 (CUB): k = 2 → 87.2, 5 → 88.4, 10 → 88.3, 20 → 87.7, 50 → 85.9 (ResNet-50 + SAC).
- **Closeness: HIGH on the pipeline** (top-k candidates → candidate-conditioned attention over spatial features →
  re-assessment fused with the first classifier). **Differences from RCEA:** (i) each candidate's query is its own
  class embedding (**class-specific, not contrastive**) — no rival subtraction anywhere; (ii) attention is pooled
  into one joint vector over all candidates, not one evidence vector per candidate; (iii) re-scoring is a separate
  classifier on that vector, not a per-candidate inner product added to that candidate's logit; (iv) class
  embeddings come from label words through a GRU. SAC is, in effect, our **"class-specific query" control with a
  joint pooling** — we must cite it as the nearest prior and include an SAC-style baseline.


**Follow-up Differential Descriptions (FuDD)** — Reza Esfandiarpoor, Stephen H. Bach. "Follow-up Differential
Descriptions: Language Models Resolve Ambiguities for Image Classification." ICLR 2024 (arXiv 2311.07593 v2; the PDF
header reads "Published as a conference paper at ICLR 2024"); code github.com/BatsResearch/fudd. **Read in: FT.**
- What it does (Sec. 3.2–3.4): for each image, the **ambiguous set C_A = the k classes most similar to the image**
  under CLIP (cosine of image embedding and class text embedding). For every *pair* of ambiguous classes an LLM writes
  captions that state their visual differences; class c's new description set is **D_c = ∪ over c_i ∈ C_A \ {c}**
  of the pair descriptions (c vs c_i); a follow-up classification is run **only over C_A**, with each class
  represented by the mean text embedding of D_c.
- Numbers on Stanford Dogs (FT, Table 1, zero-shot, top-1): ViT-B/32 — single template 58.01, naive LLM descriptions
  57.86, **FuDD k = 10: 60.03**, FuDD k = |C|: 60.28; ViT-L/14@336 — 73.61, 74.02, **75.99**, 76.34. Fig. 3 / Sec. 4:
  most of the gain comes from the few most ambiguous classes.
- **Closeness: HIGH in the idea, MEDIUM in the mechanism.** It is the clearest prior statement that what separates a
  class depends on *which rivals* the image is confused with, and it conditions each candidate's representation on
  the other members of an image-specific top-k set — exactly RCEA's conditioning signal. **Differences:** zero-shot
  CLIP, no training; the rival information enters through LLM-written *text* about each pair, not through a learned
  embedding difference; no attention over patch tokens (the image embedding is unchanged); the class representation
  is a union/mean of pair descriptions, not e_c − ē_rivals. Must be cited as the conceptual prior ("rival-dependent
  class descriptions").

**Probable-Class Nearest-Neighbour explanations (PCNN)** — Giang (Dexter) Nguyen, Valerie Chen, Mohammad Reza Taesiri,
Anh Totti Nguyen. "PCNN: Probable-Class Nearest-Neighbor Explanations Improve Fine-Grained Image Classification
Accuracy for AIs and Humans." *Transactions on Machine Learning Research*, 08/2024 (OpenReview OcFjqiJ98b; arXiv
2308.13651 v5). **Read in: FT.**
- What it does (Sec. 3.3, Fig. 2): a frozen classifier C gives the top-K classes (K = 10 chosen in App. B.3 from
  {1, 2, 3, 5, 10, 15}); for each candidate the nearest training image of that class is retrieved; a trained binary
  comparator S(x, nn) scores "same class?"; each candidate's confidence is multiplied by S and the top-K is re-ranked.
- Numbers on Dogs-120 = Stanford Dogs (FT, Table 1; ImageNet-pretrained ResNets fine-tuned on Dogs): ResNet-18
  78.75 → 79.58 (+0.83), ResNet-34 82.58 → 83.62 (+1.04), ResNet-50 85.82 → 86.31 (+0.49). The authors note dogs gain
  least of the three datasets (Sec. 4.1).
- **Closeness: MEDIUM.** Same "shortlist then re-score each candidate and fuse with C" pipeline. **Differences:**
  each candidate is checked independently against *its own* exemplar (class-specific, no rivals), the evidence is a
  whole-image comparison with a retrieved neighbour, not patch attention, and the fusion is multiplicative.

**Coarse Classification and Fine Re-ranking (CCFR)** — Shaokang Yang, Shuai Liu, Cheng Yang, Changhu Wang (ByteDance).
"Re-rank Coarse Classification with Local Region Enhanced Features for Fine-Grained Image Recognition." arXiv
2102.09875 v1 (Feb 2021). **Read in: FT.** Venue: none found (UNVERIFIED whether it was published).
- What it does (Sec. 3.3): a coarse classifier gives the top-n classes; region-enhanced embeddings retrieve the top-m
  database samples; the top-n classes are re-scored from the retrieval statistics. Evaluated on CUB, Cars, Aircraft
  (no Dogs result).
- **Closeness: LOW–MEDIUM.** Shares "re-rank the top-n" only; no candidate-conditioned query, no rivals.

**Class-Specific Residual Attention (CSRA)**: Ke Zhu, Jianxin Wu. "Residual Attention: A Simple but Effective Method
for Multi-Label Recognition." *ICCV 2021*, pp. 184–193, DOI 10.1109/ICCV48922.2021.00025 (Crossref); arXiv 2108.02456;
code github.com/Kevinz-code/CSRA. **Read in: FT (Sec. 3.2–3.3).**
- What it does: with spatial features x_k and the classifier weight m_i of class i, the attention over locations is
  s_k^i = softmax_k(T·x_kᵀm_i) (Eq. 2); the class-specific evidence is a_i = Σ_k s_k^i x_k (Eq. 3); the logit is
  **y_i = m_iᵀg + λ·m_iᵀa_i** with g the average-pooled feature (Eqs. 5–7; λ is visible in Eq. 9 and in the Fig. 1
  code, the text extraction drops the symbol). With T → ∞ this becomes "average pooling + λ·max pooling" (Eq. 9).
- **Closeness: HIGH on the mechanism.** This is RCEA's scoring rule exactly: a per-class query attends over spatial
  tokens, the pooled evidence is re-scored by an inner product with the *same* query, and the result is *added to the
  global logit*. The only difference in that rule is that CSRA's query is the class's own weight vector.
  **RCEA's "class-specific query" control (q_c = e_c) is CSRA restricted to the top-K.** Other differences: CSRA
  scores all C classes (no shortlist), its query is not contrastive, it has no learned projections, and it targets
  multi-label recognition (VOC2007 and MS-COCO are named in Sec. 3.1). CSRA must be cited as the origin of the
  scoring form, and a CSRA head is the natural strongest "no rivals" baseline.

### 3.2 Other related work, by group (each checked against the primary source; read depth stated)

**(a) Hypothesis-conditioned (top-down) attention and re-classification**
- **Look and Think Twice**: Cao et al., *ICCV 2015*, pp. 2956–2964, DOI 10.1109/ICCV.2015.338. **FT (Sec. 4.3).**
  GoogLeNet predicts the top-5 labels; for each label a feedback pass gives an attention box; each box is cropped and
  re-classified; the 25 resulting labels are ranked. ImageNet-2014 val, Table 3: top-1 error improves by 1.79 points,
  top-5 error by 1.29. *Medium:* shortlist, then one class-conditioned attention per candidate, then re-score; but the
  attention is class-specific (no rivals), training-free, and the evidence is a crop.
- **Distillation paper, specialists**: Hinton, Vinyals, Dean, arXiv 1503.02531 (NIPS 2014 Deep Learning Workshop).
  **FT (Sec. 5.2–5.4).** Specialists are trained on *confusable subsets* found by clustering the covariance of the
  generalist's predictions; at test time the generalist's top-k classes activate the specialists whose subsets meet
  them. *Medium–low:* the rivals decide *which expert* runs; the subsets are fixed, not formed per image.
- **Network of Experts**: Ahmed, Baig, Torresani, *ECCV 2016*, LNCS pp. 516–532, DOI 10.1007/978-3-319-46478-7_32.
  **ABS.** Branches specialise in learned subsets of classes that are "difficult to tell apart". *Low–medium.*
- **AbSViT**: Shi, Darrell, Wang, *CVPR 2023*, pp. 2102–2112, DOI 10.1109/CVPR52729.2023.00209. **ABS.** A top-down
  goal signal modulates ViT attention. *Low* (goal-conditioned, not rival-conditioned).

**(b) Class-query cross-attention heads (class-specific queries, all classes)**
- **Query2Label**: Liu, Zhang, Yang, Su, Zhu, arXiv 2107.10834 (2021); no published venue found (UNVERIFIED).
  **ABS.** Label embeddings are decoder queries that cross-attend to the feature map; one binary classifier per label.
- **ML-Decoder**: Ridnik, Sharir, Ben-Cohen, Ben-Baruch, Noy, *WACV 2023*, pp. 32–41,
  DOI 10.1109/WACV56688.2023.00012. **ABS.** A query-based head with group decoding; the abstract reports 80.7%
  ImageNet top-1 with a vanilla ResNet-50.
- **INTR**: Paul et al. (14 authors), *ICLR 2024* (poster; OpenReview forum bkdWThqE6q), arXiv 2311.04157.
  **FT (Sec. 3–4).** One learned query per class cross-attends to DETR encoder features; each class's logit is the
  inner product of its output vector with one shared class-agnostic vector. Stanford Dogs (Table 2): INTR 72.5 vs
  ResNet-50 77.1. It is an interpretability method, not an accuracy method.
- **MCTformer**: Xu, Ouyang, Bennamoun, Boussaid, Xu, *CVPR 2022*, pp. 4300–4309, DOI 10.1109/CVPR52688.2022.00427.
  **ABS.** One class token per class; class-to-patch attention gives class-specific maps (weakly supervised
  segmentation).
- **AttentionXML**: You et al., *NeurIPS 2019* (arXiv comment and Semantic Scholar), arXiv 1811.01727. **ABS.**
  Label-specific attention over the text, with a probabilistic label tree that shortlists labels level by level.
- **Cross Attention Network** (Hou, Chang, Ma, Shan, Chen, *NeurIPS 2019*, arXiv 1910.07677) and **CrossTransformers**
  (Doersch, Gupta, Zisserman, *NeurIPS 2020*, arXiv 2007.11498): few-shot; cross-attention between a query image and
  *each support class separately*. **ABS.**
- *All medium–low:* they supply the "per-class query cross-attends to patches" ingredient; none forms the query from a
  shortlist or from rivals.

**(c) Contrastive explanations: the same contrast, used post hoc, never trained and never used to re-score**
- **Excitation Backprop (c-MWP)**: Zhang, Lin, Brandt, Shen, Sclaroff, *ECCV 2016*, LNCS pp. 543–559,
  DOI 10.1007/978-3-319-46493-0_33; extended in *IJCV* 126(10):1084–1102, DOI 10.1007/s11263-017-1059-x (adds
  S. A. Bargal). **FT (Sec. 3.3).** The contrastive map subtracts the map of a *dual unit whose weights are the
  negated target weights* ("non-elephant").
- **CLRP**: Gu, Yang, Tresp, *ACCV 2018*, LNCS pp. 119–134 (published 2019), DOI 10.1007/978-3-030-20893-6_8.
  **FT (Sec. 4).** R_CLRP = max(0, R − R_dual); the dual "virtual class" is either **all classes except the target,
  with the score redistributed uniformly (CLRP1)** or the negated target weights (CLRP2). CLRP1 is our *globally
  centred* control in explanation form.
- **SGLRP**: Iwana, Kuroki, Uchida, *ICCV Workshops 2019*, pp. 4176–4185, DOI 10.1109/ICCVW.2019.00513.
  **FT (Sec. 4, Eq. 10).** The initial relevance is the softmax gradient: y_t(1 − y_t) for the target and
  **−y_t·y_n for every other class n**, so the target is contrasted with the other classes *weighted by their
  probabilities*; the authors set this explicitly against CLRP's uniform penalty. SGLRP is our *probability-weighted*
  variant in explanation form (over all classes, not a top-K).
- **Contrastive Explanations ("Why P, rather than Q?")**: Prabhushankar, Kwon, Temel, AlRegib, *ICIP 2020*,
  pp. 3289–3293, DOI 10.1109/ICIP40778.2020.9190927. **FT (Sec. 3).** Grad-CAM driven by the gradient of J(P, Q) for
  one chosen contrast class Q; demonstrated on fine-grained recognition.
- **SCOUT**: Wang, Vasconcelos, *CVPR 2020*, pp. 8978–8987, DOI 10.1109/CVPR42600.2020.00900. **ABS.** Heatmaps of
  regions informative of the prediction but not of a counter class.
- **Counterfactual Visual Explanations**: Goyal, Wu, Ernst, Batra, Parikh, Lee, *ICML 2019*, PMLR 97, pp. 2376–2384
  (PMLR page). **ABS.** Regions to swap in from a distractor image of class c′ so that the prediction flips to c′.
- *Medium in idea, low in mechanism:* together they establish that "evidence for c against its rivals" is more
  discriminative than "evidence for c". None restricts the rivals to an image-specific shortlist inside a trained head,
  and none feeds the contrast back into the prediction.

**(d) Rival-aware training objectives (no conditioning at test time)**
- **Gradient-boosting CE**: Sun, Cholakkal, Khan, Khan, Shao, *AAAI 2020*, 34(07):12047–12054,
  DOI 10.1609/aaai.v34i07.6882. **FT (Sec. 3.2).** The softmax is normalised only over the ground truth and the
  **top-k negative classes**; a diversification block masks salient CAM regions.
- **Sharpen Focus**: Wang et al., *ICCV 2019*, pp. 512–521, DOI 10.1109/ICCV.2019.00060. **FT (Sec. 3.2).** A loss
  penalises the overlap between the ground-truth class's attention and the attention of the **most confusing class**
  (the highest-probability non-ground-truth class).
- **FocusNet**: Zhang, Sheng, Shen, *Pattern Recognition* 129:108709 (2022), DOI 10.1016/j.patcog.2022.108709.
  **FT (Sec. 1, 3.1–3.2).** A frozen baseline names each training image's confusing classes; a focus-picking loss
  trains a second network on them; only the second network runs at test time.
- **API-Net**: Zhuang, Wang, Qiao, *AAAI 2020*, 34(07):13130–13137, DOI 10.1609/aaai.v34i07.7016. **FT (Sec. 3).**
  Pairs of *images* (each image with its most similar intra- and inter-class images in the batch) exchange gated
  features; the module is removed at test time. The abstract reports 90.3% on Stanford Dogs.
- *Medium–low:* rivals shape training only.

**(e) Pair-specific features**
- **POOF**: Berg, Belhumeur, *CVPR 2013*, pp. 955–962, DOI 10.1109/CVPR.2013.128. **ABS + introduction.** Each
  feature specialises in separating **two particular classes** at one part (needs part labels). *Medium–low:* pairwise
  rival evidence, but precomputed for all pairs, not chosen per image.

**(f) Conditioning on the set of competing classes (few-shot)**
- **FEAT**: Ye, Hu, Zhan, Sha, *CVPR 2020*, pp. 8805–8814, DOI 10.1109/CVPR42600.2020.00883. **ABS.** A set-to-set
  Transformer adapts the class prototypes of an episode jointly, making them "task-specific and discriminative".
- **CTM**: Li, Eigen, Dodge, Zeiler, Wang, *CVPR 2019*, pp. 1–10, DOI 10.1109/CVPR.2019.00009. **ABS.** Traverses the
  whole support set to keep the feature dimensions that show intra-class commonality and inter-class uniqueness.
- *Medium in idea:* the class representation depends on the other classes present. But the set is the given episode,
  not a classifier's top-K for this image, and there is no patch attention driven by a difference query.

**(g) Shortlist, then reason with a language model (rival-aware through text)**
- **Comparative descriptors**: Lee, Seo, Choi, Jung, Song, Jung, *WACV 2025*, pp. 5274–5283,
  DOI 10.1109/WACV61041.2025.00515. **ABS + method section (searched).** For each class, n similar classes are fixed
  *in advance* from CLIP text similarity; an LLM describes what distinguishes the target from each; descriptors are
  filtered against class-mean image features. *Medium in idea:* rivals are chosen per class, not per image.
- **CascadeVLM**: Canshi Wei, *Findings of EMNLP 2024*, pp. 1857–1871, DOI 10.18653/v1/2024.findings-emnlp.102.
  **ABS + method section (searched).** CLIP gives the top-k candidates; a large VLM picks one of them.
- **RAR**: Liu et al. (9 authors), *IEEE TIP* 35:388–401 (2026), DOI 10.1109/TIP.2025.3644175; arXiv 2403.13805.
  **ABS.** CLIP retrieves the top-k class names; an MLLM ranks them.
- *Medium–low:* the shortlist goes to a model that sees all candidates together, but nothing like a learned
  difference query.

**(h) Other low-closeness items and name clashes**
- **Counterfactual Attention Learning (CAL)**: Rao, Chen, Lu, Zhou, *ICCV 2021*, pp. 1005–1014,
  DOI 10.1109/ICCV48922.2021.00106. **ABS.** Maximises the effect of the learned attention over a counterfactual
  (random) attention: an analogue of our random-rivals control, not of the query.
- **CHM-Corr**: Taesiri, Nguyen, Nguyen, *NeurIPS 2022* (vol. 35), pp. 34287–34301, DOI 10.52202/068431-2485 (author
  order as in Crossref; arXiv 2208.00780 lists Giang Nguyen first). **ABS.** Self-interpretable classifiers built on
  patch correspondences with exemplars.
- **RaDiCal**: S. Liu et al., *EMNLP 2026* (arXiv comment), arXiv 2609.00667. **ABS.** Token pruning for VLM listwise
  rerankers that discounts "order-neutral patterns shared across candidates": the only other construction found that
  scores something *relative to the other candidates*, and it is in retrieval, not classification.
- **CLIP-guided label-free region scoring**: Y. Zhu, arXiv 2607.13437 (July 2026). **ABS.** Margin- and entropy-based
  CLIP scores for choosing regions, training-free. *Low.*
- **Name clash, Differential Transformer**: Ye et al., *ICLR 2025* (oral; OpenReview), arXiv 2410.05258. **ABS.**
  "Differential attention" there is the difference of two softmax attention maps, used to cancel noise; it has
  nothing to do with rivals. Avoid the words "differential attention" and "contrastive attention" (the latter is also
  Excitation Backprop's term) in our method's name. No method using the term "rival" was found.

### 3.3 Closeness table

| Work (venue) | What it conditions on | Where the conditioning acts | Main difference from RCEA | Closeness |
|---|---|---|---|---|
| **CSRA** (ICCV 2021) | each class's own classifier weight, all C classes | spatial softmax attention; pooled evidence · same weight, **added to the logit** | no shortlist, no rivals (class-specific query); multi-label | **High (mechanism)** |
| **SAC** (IEEE CAI 2024) | top-k (k = 10) candidates' label-word embeddings | bilinear attention over locations × candidates → one joint vector → second classifier, λ-fusion | class-specific queries, joint pooling, separate classifier | **High (pipeline)** |
| **FuDD** (ICLR 2024) | the image's top-k ambiguous classes; pairwise LLM-written differences | class text embeddings for a follow-up classification over the top-k | zero-shot, text only, no patch attention | **High (idea)**, medium (mechanism) |
| Look and Think Twice (ICCV 2015) | each of the top-5 labels | feedback attention → crop → re-classify | class-specific, training-free | Medium |
| PCNN (TMLR 2024) | each top-K class's nearest training image | learned comparator × confidence | exemplar check, no rivals | Medium |
| SGLRP (ICCVW 2019) | all other classes, softmax-probability weighted | post-hoc relevance map | explanation only, all classes | Medium (= our prob.-weighted variant, post hoc) |
| CLRP (ACCV 2018) | all other classes, uniform (CLRP1); negated weights (CLRP2) | post-hoc relevance map | explanation only | Medium (= our globally-centred control, post hoc) |
| Comparative descriptors (WACV 2025) | n most similar classes fixed per class | LLM descriptors in CLIP | rivals per class, not per image; zero-shot | Medium (idea) |
| FEAT (CVPR 2020) / CTM (CVPR 2019) | the other classes of the episode | prototype adaptation / feature mask | episode, not a shortlist; no difference query | Medium (idea) |
| Contrastive Grad-CAM (ICIP 2020) | one chosen contrast class Q | Grad-CAM | explanation only | Medium–low |
| Hinton et al. specialists (2015) | top-k → specialists of fixed confusable subsets | which expert model runs | fixed subsets, separate models | Medium–low |
| Gradient-boosting CE (AAAI 2020) | top-k negative classes | training loss | train-time only | Medium–low |
| Sharpen Focus (ICCV 2019) | most confusing class | training loss on attention overlap | train-time only | Medium–low |
| POOF (CVPR 2013) | fixed class pairs × parts | pair-specific features | all pairs precomputed, part labels | Medium–low |
| Query2Label / ML-Decoder (WACV 2023) / INTR (ICLR 2024) / MCTformer (CVPR 2022) | each class's own query | cross-attention over tokens | class-specific, all classes | Medium–low |
| CascadeVLM (Findings EMNLP 2024) / RAR (TIP 2026) | CLIP top-k given to an (M)LLM | prompt | no learned query | Medium–low |
| FocusNet (PR 2022) / API-Net (AAAI 2020) | confusing classes / most similar images | training only | removed at test time | Low–medium |
| Excitation Backprop c-MWP (ECCV 2016) / SCOUT (CVPR 2020) / Goyal et al. (ICML 2019) | negated weights / a counter class / a distractor image | explanation | not a shortlist, not trained | Low–medium |
| CCFR (arXiv 2021) / Network of Experts (ECCV 2016) / AttentionXML (NeurIPS 2019) | retrieval statistics / learned subsets / label tree | re-rank / branches / shortlist | no rival query | Low–medium |
| CAN (NeurIPS 2019), CrossTransformers (NeurIPS 2020), CHM-Corr (NeurIPS 2022), CAL (ICCV 2021), AbSViT (CVPR 2023), RaDiCal (EMNLP 2026), label-free region scoring (arXiv 2026), Differential Transformer (ICLR 2025) | various | various | share one ingredient or only the name | Low |

### 3.4 Three algebraic facts that shape the claim (derived here; not stated in any paper read)

Notation: shortlist S = top-K of the global classifier, embeddings e_j, candidate c ∈ S.

1. **The mean-rival query is the shortlist-centred embedding.** With ē_S = (1/K) Σ_{j∈S} e_j,
   q_c = e_c − (1/(K − 1)) Σ_{j∈S, j≠c} e_j = **(K/(K − 1)) · (e_c − ē_S)**.
   So the K queries sum to zero, and the planned controls form one clean axis: *no centring* (q_c = e_c, i.e. CSRA on
   the top-K), *global centring* (e_c − ē_all, the CLRP1 contrast), *centring on a random set* (random rivals), and
   *centring on the image's own shortlist* (RCEA). For K = 2, q_1 = e_1 − e_2 = −q_2: the two candidates look for
   opposite evidence. The contribution, stated honestly, is **where the centring set comes from**.
2. **The probability-weighted query is the top-K log-softmax gradient (SGLRP's weighting).** With p the softmax over S
   and π_j = p_j / (1 − p_c) for j ≠ c, q_c = e_c − Σ_{j≠c} π_j e_j. If e_j are the global classifier's weights w_j on
   a pooled feature f, then ∇_f log p_c = w_c − Σ_{j∈S} p_j w_j = **(1 − p_c) · q_c**. So with tied weights and no
   projections, the attention logits ⟨q_c, x_i⟩ are (up to scale) a Grad-CAM-of-log-softmax map restricted to the
   shortlist, which can be computed after training without any new parameters. Untying e_c from the classifier and
   learning W_Q, W_K is what makes the probability-weighted variant more than a re-use of SGLRP's weighting.
   (Implementation note: detach p when forming π, so the re-scoring head cannot reshape the global classifier's
   probabilities to suit itself.)
3. With identity projections, RCEA's score is s_c = Σ_i softmax_i(τ⟨q_c, x_i⟩)·⟨q_c, x_i⟩: a soft maximum over patches
   of *c's per-patch margin over its rivals*. That is CSRA's Eq. 8 with m_c replaced by q_c (and CSRA's Eq. 9 gives
   the hard max as τ → ∞).

### 3.5 Verdict

**Search done:** more than ten web-search phrasings (standard mode; one extended search for the exact mechanism,
which found nothing), arXiv / Crossref / Semantic Scholar / OpenReview / PMLR lookups for every candidate, and full-text
reading of the closest papers (SAC, CSRA, FuDD, PCNN, CCFR, Look and Think Twice, SGLRP, CLRP, Excitation Backprop,
Contrastive Explanations, Sun et al., Sharpen Focus, FocusNet, API-Net, INTR, Hinton et al.).

**No prior work found does all four of:** (1) take an image-specific top-K from a classifier; (2) form, for each
candidate, a query from its embedding minus the (mean or probability-weighted) embedding of the other K − 1;
(3) cross-attend with that query over patch tokens; (4) re-score the pooled evidence by an inner product with the same
query and add it to that candidate's logit. **But every ingredient exists:** CSRA has (3) + (4) with a class-specific
query; SAC has (1) + (3) in FGVC, on Stanford Dogs, with class-specific label embeddings; FuDD has (1) + (2) in
spirit, through LLM text and without patch attention; SGLRP and CLRP have the probability-weighted and uniform
contrasts of (2), as post-hoc explanations over all classes. By fact 1, the mean variant is "CSRA with queries
centred on the image's shortlist".

**Verdict: novel as a specific mechanism; incremental as an idea.** It is defensible only if the experiments show
that *shortlist-conditioning* is what helps. **The three closest works: CSRA (mechanism), SAC (pipeline),
FuDD (idea).** The explanation methods SGLRP and CLRP supply the logic of the controls and must be cited with them.

**Smallest change that makes the contribution distinct and defensible:**
1. Present it as "residual class-query attention (CSRA form) whose queries are contrasted against the image's own
   top-K shortlist", citing CSRA, SAC, FuDD, SGLRP and CLRP up front, not in a footnote.
2. **Untie the query embeddings from the classifier weights and learn W_Q and W_K.** Otherwise the probability-weighted
   variant is the top-K log-softmax gradient (fact 2), i.e. a post-hoc Grad-CAM/SGLRP-style map.
3. The minimum comparison set, all at the same backbone, resolution, pretraining and K:
   CSRA over all classes; CSRA restricted to the top-K (= the class-specific control); global centring (= the CLRP1
   contrast); random rivals; SAC; and a training-free "top-K log-softmax gradient" re-scorer. Report mean ± sd over
   at least 3 seeds. The claim stands only if the shortlist-centred query beats the class-specific and globally
   centred queries by more than the seed spread, and the gain disappears with random rivals.
4. If a wider novelty margin is wanted: let rivals enter **pairwise** (q_{c,r} = e_c − e_r, one pooled evidence per
   pair, then a soft minimum over r, i.e. "c must beat its strongest rival"). That cannot be rewritten as a centred
   class query; it is the learned, visual counterpart of FuDD's pairwise descriptions and POOF's pair features.
5. Name: avoid "differential attention" (Differential Transformer, ICLR 2025) and "contrastive attention"
   (Excitation Backprop's term). No method was found using "rival".

---

## Part 4 — protocol changes this dossier supports

1. **Tsinghua Dogs:** the official split only (65,228 train / 5,200 val, 40 per breed). The 5,200 "val" is the de facto
   test set, so carve a model-selection split out of train and never tune on val.
2. **Boxes:** detector crops are the main result. Ground-truth-box crops appear only as an oracle row, for our method
   and for the head–body baseline (the CMC 2025 paper never states which it used for its 90.04 %).
3. **Contamination, Tsinghua:** report val accuracy separately for Stanford-origin and other images (provenance by
   perceptual hash against Stanford Dogs). About 18,000 Tsinghua images come from Stanford Dogs, so at least 6,000 of
   them are Stanford *test* images.
4. **Contamination, Stanford Dogs:** keep the standard 12,000 / 8,580 split, but count Stanford-test ∩ ILSVRC-2012
   train by file name (the ImageNet ids are kept) and report the number, citing Krause et al. 2016 Sec. 5.1. Add one
   ablation with dog classes removed from pretraining (Krause et al. 2016; MetaFGNet: 69.3% with ResNet-34).
5. **Independent tests:** the 120 dog classes of ImageNetV2 (near-duplicates against ImageNet removed by its authors),
   and the dog breeds of Oxford-IIIT Pet after removing BiT's listed near-duplicates (58 Pets test images match
   ILSVRC-2012 train) and de-duplicating against Stanford and Tsinghua.
6. **No cross-dataset train/test without de-duplication** between Tsinghua and Stanford.
7. **Metrics:** top-1 and macro (per-class) accuracy; Tsinghua train is long-tailed while val is balanced, so also
   report the tail breeds. Mean ± sd over at least 3 seeds.
8. **Inductive inference only:** one image at a time, no batch-dependent test features (RBI's transductive batch
   inference makes its 95.79 / 97.79% non-comparable).
9. **Report what the four papers omit:** parameters, FLOPs, input resolution, pretraining source (1k vs 21k), code.
   Re-run baselines at equal backbone and resolution instead of copying numbers (CMC 2025 copied the 2020 table).
10. **For RCEA specifically:** report the global classifier's top-K recall (the ceiling of any re-ranker), accuracy
    when the truth is inside the top-K, and a K sweep (SAC's Table 5 found k = 5–10 best on CUB).
