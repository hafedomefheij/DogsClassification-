# Related-work dossier — verified table (FGVC, query/attention heads, confusion-aware learning, backbones, reliability, adaptive computation)

Started 2026-11-08. Companion to `lit_core.md` (the four named papers and the novelty search live there, not here).
Written incrementally: every entry is appended as soon as it is checked, so partial work survives an interruption.

Rules followed:
- Bibliographic data (title, authors, venue, year, DOI/arXiv id) checked against Crossref, the Semantic Scholar API,
  arXiv, CVF open access or the publisher page. Anything not confirmed is marked **UNVERIFIED**.
- Numbers come only from the paper itself, with the table cited. "FT" = read in the full text (arXiv / ar5iv HTML,
  CVF HTML/PDF, OpenReview); "ABS" = abstract / metadata only. A number seen only on a third-party leaderboard
  (Papers with Code etc.) is labelled as such and is not put in the accuracy columns.
- Machine-readable table: `results/literature_table.csv`. BibTeX (verified entries only): `paper/refs_related.bib`.

Key caveat carried through the whole table: **Stanford Dogs is built from ImageNet images** (Khosla et al., 2011),
so any backbone pretrained on ImageNet-1k or ImageNet-21k/22k (or on a corpus seeded with them, e.g. DINOv2's
LVD-142M) may have seen Stanford Dogs *test* images during pretraining. Each entry records its pretraining.

---

## 1. FGVC methods (CNN era)

Bibliographic check for all of section 1: one Semantic Scholar batch call on the arXiv ids (titles, authors, venues all
matched), then Crossref for the publisher DOI / pages of each conference or journal version.

**B-CNN** — Lin, RoyChowdhury, Maji, "Bilinear CNN Models for Fine-Grained Visual Recognition", ICCV 2015,
pp. 1449–1457, doi:10.1109/ICCV.2015.170 (Crossref). arXiv 1504.07889 is now the extended TPAMI version.
- No Stanford Dogs result in the paper (FT, ar5iv: "dogs" appears only in the introduction). The 82.1 % (VGG-19)
  often attributed to B-CNN on Dogs is a **third-party** number, e.g. API-Net Table 9.
- CUB 84.1 / Cars 91.3 / Aircraft 84.1 (ICCV numbers as tabulated by PMG Table 1 and MC-Loss Table IV).

**NTS-Net** — Yang, Luo, Wang, Hu, Gao, Wang, "Learning to Navigate for Fine-grained Classification", ECCV 2018,
LNCS pp. 438–454, doi:10.1007/978-3-030-01264-9_26 (Crossref). FT (ar5iv): CUB 87.5 (Table 2), Aircraft 91.4 and
Cars 93.9 (Table 3). **No Stanford Dogs.** ResNet-50, 448 px.

**WS-DAN** — Hu, Qi, Huang, Lu, "See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for
Fine-Grained Visual Classification", arXiv:1901.09891 (2019; arXiv only — S2 lists no venue). FT (ar5iv):
**Stanford Dogs 92.2 % (Table 7)**, Inception-v3 baseline 88.9 %, ResNet-101 85.8 %; CUB 89.4 (T4), Aircraft 93.0 (T5),
Cars 94.5 (T6). Backbone Inception-v3 (Mixed6e), 32 attention maps; input size not stated in Sec. 4.2.
Note how strong the *plain* ImageNet backbone already is on Dogs (88.9 %) — consistent with the ImageNet overlap.

**API-Net** — Zhuang, Wang, Qiao, "Learning Attentive Pairwise Interaction for Fine-Grained Classification",
AAAI 2020, 34(07):13130–13137, doi:10.1609/aaai.v34i07.7016 (Crossref). FT (ar5iv):
**Stanford Dogs (Table 9): 88.3 (ResNet-50), 90.3 (ResNet-101), 89.4 (DenseNet-161)**; DenseNet-161: CUB 90.0,
Aircraft 93.9, Cars 95.3, NABirds 88.1. ResNet-101 "pretrained on ImageNet"; 512 → 448 crop.
*Why it matters to us:* API-Net is the clearest precursor of rival-conditioned processing — it builds a mutual
vector from an image **pair**, preferring similar inter-class pairs, and gates each image's features by what differs
from the partner, with a score-ranking loss. But the interaction is used only in training; at test time it is a plain
CNN ("our testing manner is just the same as that of a plain CNN"). Our method conditions on the rival **classes**
at test time.

**PMG** — Du, Chang, Bhunia, Xie, Ma, Song, Guo, ECCV 2020, LNCS pp. 153–168, doi:10.1007/978-3-030-58565-5_10.
FT: ResNet-50 combined accuracy CUB 89.6 / Cars 95.1 / Aircraft 93.4 (Table 1); 550 → 448. **No Dogs.**

**DCL** — Chen, Bai, Zhang, Mei, "Destruction and Construction Learning for Fine-Grained Image Recognition",
CVPR 2019, pp. 5152–5161, doi:10.1109/CVPR.2019.00530. FT (CVF PDF): ResNet-50 CUB 87.8 / Cars 94.5 /
Aircraft 93.0 (Table 1); ImageNet-pretrained; 512 → 448. **No Dogs.**

**Cross-X** — Luo et al., ICCV 2019, pp. 8241–8250, doi:10.1109/ICCV.2019.00833. FT (ar5iv):
**Stanford Dogs 88.9 (ResNet-50) / 88.2 (SENet-50) (Table 6)** against their own ResNet-50 baseline of 88.1 and
SENet-50 87.1; CUB 87.7, Cars 94.6, Aircraft 92.6, NABirds 86.4. ImageNet init; 448 crops; lr 0.001 for Dogs only.

**MC-Loss** — Chang, Ding, Xie, Bhunia, Li, Ma, Wu, Guo, Song, IEEE TIP 29:4683–4695 (2020),
doi:10.1109/TIP.2020.2973812. FT: ResNet-50 CUB 87.3 / Aircraft 92.6 / Cars 93.7 (Table IV). **No Dogs.**

**CAL** — Rao, Chen, Lu, Zhou, "Counterfactual Attention Learning for Fine-Grained Visual Categorization and
Re-identification", ICCV 2021, pp. 1005–1014, doi:10.1109/ICCV48922.2021.00106. FT: ResNet-101, 448 px,
multi-crop + flip at inference: CUB 90.6 / Cars 95.5 / Aircraft 94.2 (Table 1). **No Dogs.** Useful control idea:
compare learned attention against random / uniform / reversed / shuffled attention (their Table 4).

## 1b. FGVC methods (transformer era)

**TransFG** — He, Chen, Liu, Kortylewski, Yang, Bai, Wang, AAAI 2022, 36(1):852–860, doi:10.1609/aaai.v36i1.19967
(Crossref, the AAAI page and arXiv v5 list 7 authors; Semantic Scholar also lists A. Yuille). FT (ar5iv):
**Stanford Dogs 92.3 (Table 2) vs plain ViT-B/16 91.7**; CUB 91.7, Cars 94.8 (T1), NABirds 90.8 (T3).
"We load intermediate weights from official ViT-B_16 model pretrained on ImageNet21k"; 448 px, overlapping
patches (stride 12); lr 0.003 for Dogs. → **ImageNet-21k pretraining.**

**FFVT** — Wang, Yu, Gao, "Feature Fusion Vision Transformer for Fine-Grained Visual Categorization", BMVC 2021
(arXiv:2107.02341v3 comment: "accepted by BMVC2021"; no DOI). FT (ar5iv):
**Stanford Dogs 91.5 (Table 3)**; in the same table the authors' own re-runs give ViT-B/16 90.2 and TransFG 90.6
(TransFG's paper says 92.3). CUB 91.6. "the backbone network (ViT) of FFVT is pretrained on the ImageNet21K dataset";
600 → 448; K = 24 tokens and batch size 4 on Dogs. → **ImageNet-21k.**
Slip in the paper: Table 1 gives Stanford Dogs as 196 classes / 8,144 / 8,041 images — those are the Stanford Cars
statistics (Dogs is 120 / 12,000 / 8,580).

**RAMS-Trans** — Hu, Jin, Zhang, Hong, Zhang, He, Xue, ACM MM 2021, pp. 4239–4248, doi:10.1145/3474085.3475561.
FT (ar5iv): **Stanford Dogs 92.4 (Table 3)** vs ViT-B/16 92.2 and their TransFG&PSM re-run 90.0; CUB 91.3,
iNat2017 68.5. "official ViT-B_16 model pre-trained on ImageNet21k"; on Dogs the images are resized (short side 448)
and **cropped to 224 × 224**. → **ImageNet-21k.**

> **Methodological point for our study.** The *same* backbone (ViT-B/16, ImageNet-21k) is reported on Stanford Dogs
> as 91.7 (TransFG, 448 px), 92.2 (RAMS-Trans, 224 px) and 90.2 (FFVT re-run, 448 px, batch 4). A ~2-point spread for
> one backbone is larger than most claimed method gains (+0.2 to +0.9). Any claim we make must use one protocol,
> several seeds and our own re-run baselines.

**SIM-Trans** — Sun, He, Peng, ACM MM 2022, pp. 5853–5861, doi:10.1145/3503161.3548308. FT: CUB 91.8 (Table 4),
iNat2017 69.9 (Table 3); ViT-B/16 ImageNet-21k. **No Dogs.**

**IELT** — Xu, Wang, Jiang, Luo, "Fine-Grained Visual Classification via Internal Ensemble Learning Transformer",
IEEE TMM 25:9015–9028 (2023), doi:10.1109/TMM.2023.3244340 (Crossref). Closed access (Semantic Scholar: CLOSED), so
the tables were **not** read. The authors' GitHub README (github.com/mobulan/IELT) lists **Stanford Dogs 91.84**,
CUB 91.81, NABirds 90.78 and points to the ImageNet-21k ViT-B_16 checkpoint. Papers with Code shows 91.8.
→ table row marked *partial*.

**HERBS** — Chou, Kao, Lin, arXiv:2303.06442 (v2; arXiv only). FT: CUB 93.1, NABirds 93.0 (Table I); Swin-Large
alone 92.0 on CUB (Table II). **Birds only, no Dogs.** Pretraining of the Swin weights not stated in the parts read.

**DCAL** — Zhu, Ke, Li, Liu, Tian, Shan, "Dual Cross-Attention Learning for Fine-Grained Visual Categorization and
Object Re-Identification", CVPR 2022, pp. 4682–4692, doi:10.1109/CVPR52688.2022.00465. FT: R50-ViT-B/16
CUB 92.0 / Cars 95.3 / Aircraft 93.3 (Table 1); 448 px; "ViT backbones are pre-trained on ImageNet-21k ... DeiT
backbones are pre-trained on ImageNet-1k". **No Dogs.** Relevant mechanism: pair-wise cross-attention (PWCA) attends
over the concatenated keys/values of the image and a distractor image — "PWCA is only used for training and will be
removed for inference". Like API-Net, the cross-image conditioning is a training regulariser, not an inference-time
comparison against named rival classes.

**MetaFormer** — Diao, Jiang, Wen, Sun, Yuan, arXiv:2203.02751 (arXiv only). FT: iNat 2017/2018/2021, CUB, NABirds
with and without meta-information; pretraining compared explicitly (ImageNet-1k vs 21k vs iNat21: with iNat21,
MetaFormer-2 reaches CUB 92.3 / NABirds 92.7, image only). **No Dogs.** Useful as evidence that the pretraining source
moves FGVC accuracy by more than most heads do.

## 1c. The ~96–97 % Stanford Dogs outliers (CNN, 224 px)

**CAP** — Behera, Wharton, Hewage, Bera, "Context-aware Attentional Pooling (CAP) for Fine-grained Visual
Classification", AAAI 2021, 35(2):929–937, doi:10.1609/aaai.v35i2.16176. FT (ar5iv of the extended arXiv version):
**Stanford Dogs 96.1 (Table 1)**, with Xception at 224 × 224 (resize 256, crop 224); Aircraft 94.9, Food-101 98.6,
Cars 95.7, CUB 91.8, Flowers 97.7, Pets 97.3, NABirds 91.0. Table 1 cites the previous Dogs best as 93.9 (primary
data only) and 97.1 (with secondary data), both from Ge, Lin, Yu (CVPR 2019).

**SR-GNN** — Bera, Wharton, Liu, Bessis, Behera, IEEE TIP 31:6017–6031 (2022), doi:10.1109/TIP.2022.3205215.
FT (ar5iv): **Stanford Dogs 97.3 (Table I)**, Xception, "Pre-trained ImageNet weights", 256 → 224 crop; Aircraft 95.4,
CUB 91.9, Cars 96.1, Flowers 97.9, NABirds 91.2. Sec. IV notes that the earlier 97.1 used COCO as a secondary dataset.

> These two are ~5 points above every ImageNet-21k ViT result although they use a 224-px ImageNet-1k CNN. Neither
> paper explains the gap. Both start from ImageNet-1k weights, whose training images include Stanford Dogs test
> images. **We should not put them in the same comparison column as the ViT results without this caveat.**

**Relational Proxies** — Chaudhuri, Mancini, Akata, Dutta, NeurIPS 2022 (Advances in NeurIPS 35, pp. 31145–31157,
doi:10.52202/068431-2258, Crossref). FT (ar5iv): main Table 1 has no Dogs (Aircraft 95.25, Cars 96.30, CUB 92.00,
NABirds 91.20, iNat 72.15, Cotton 69.81, Soy 51.20, each ± ≤ 0.04). Table 4 uses "Dogs ImageNet" (Stanford Dogs)
against Tiny ImageNet: **92.75 on Dogs** vs 91.30 for their relation-agnostic ResNet-50 encoder (gain 1.45, vs 0.16 on
Tiny ImageNet). ResNet-50 "pretrained on ImageNet".

**PDiscoNet** — van der Klis, Alaniz, Mancini, Dantas, Ienco, Akata, Marcos, ICCV 2023, pp. 1866–1876,
doi:10.1109/ICCV51070.2023.00179. FT (arXiv HTML): CUB classification 87.49 (K = 16) vs a ResNet-101 baseline of
85.35; PartImageNet 89.00. **No Dogs.** Interpretable part maps; part dropout.

## 1d. 2022–2025 methods that report Stanford Dogs (search: WebSearch "standard", the comparison tables of the
## newest papers found, then each paper read in full)

No 2024–2026 paper found reports **Tsinghua Dogs** apart from the ones the other agent is checking (the CMC 2025
head/body paper; CSQA-Net; the ICT Express 2026 paper). The Tsinghua Dogs project page itself lists WS-DAN
(Inception-v3) 86.4, TBMSL-Net 83.7, PMG 83.5, Inception-v3 77.7 (web search snippet; that page belongs to the
other agent's Tsinghua Dogs entry, so it is not repeated in the table).

**MP-FGVC** — Jiang, Tang, Gao, Du, He, Li, "Delving into Multimodal Prompting for Fine-grained Visual
Classification", AAAI 2024, 38(3):2570–2578, doi:10.1609/aaai.v38i3.28034 (Crossref). FT (arXiv HTML):
**Stanford Dogs 91.0 (Table 1)**; CUB 91.8, NABirds 91.0. ViT-B/16 "pre-trained on ImageNet21K" + CLIP text encoder,
448 px. The valuable part: they **re-ran** TransFG, IELT and SIM-Trans from public code and got 90.5 / 90.6 / 89.2 on
Dogs — 1.2 to 1.8 points under what those papers report (92.3 / 91.84 / not reported).

**H3Former** — Zhang, Li, Huang, Gao, Jiang, arXiv:2511.10260v2 (comment: "Accepted by IEEE Transactions on Image
Processing"; no DOI found yet). FT (arXiv HTML): **Stanford Dogs 95.8 (Table III)** with Swin-B at 448 px; without
its two modules the same backbone gets 91.1 (Table VI). Sec. IV-B: Swin-B "pre-trained on ImageNet-22K" for CUB,
NABirds and Flowers, "while for the Stanford-Dogs dataset, we use the version pre-trained on ImageNet-1K". CUB 92.7,
NABirds 91.6. Its Dogs table also lists ViT-NeT 93.6 (ICML 2022), "ACC-ViT" 92.9 (cited as TCSVT 2025; the citation is
inconsistent inside H3Former and the paper could not be located: **UNVERIFIED**), LGTF 92.1, FAL-ViT 91.1.

**ViT-NeT** — Kim, Nam, Ko, "ViT-NeT: Interpretable Vision Transformers with Neural Tree Decoder", ICML 2022,
PMLR 162:11162–11172 (PMLR page). FT (PMLR PDF): **Stanford Dogs (Table 2): 93.6 with DeiT-B, 90.3 with Swin-B**;
the plain DeiT-B gets 91.5 and Swin-B 88.0. Swin layers "pre-trained using ImageNet-22K"; 448 px. (The text calls
Stanford Dogs "100 different breeds"; it has 120.) Journal extension: "Neural Tree Decoder for Interpretation of
Vision Transformers", IEEE TAI 5(5):2067–2078 (2024), doi:10.1109/TAI.2023.3312645 (Crossref; not read).

**LGTF** — Zhu, Chen, Yin, See, Liu, "Learning Gabor Texture Features for Fine-Grained Recognition", ICCV 2023,
pp. 1621–1631, doi:10.1109/ICCV51070.2023.00156. FT (CVF PDF): **Stanford Dogs 92.1 (DenseNet-161) / 91.7
(ResNet-101) (Table 3)**; CUB 91.5, NABirds 90.4. Note their ResNet-101 baseline on Dogs is only 80.6.

## 1e. The frozen-backbone protocol on Stanford Dogs

**VPT** — Jia, Tang, Chen, Cardie, Belongie, Hariharan, Lim, "Visual Prompt Tuning", ECCV 2022, LNCS pp. 709–727,
doi:10.1007/978-3-031-19827-4_41 (Crossref). FT (ar5iv), Table 14, ViT-B/16 **supervised ImageNet-21k**, 224 px,
frozen backbone unless stated: **Stanford Dogs — linear probe 86.2, MLP-2 85.4, bias tuning 91.2, adapter-256 89.9,
VPT-shallow 90.7, VPT-deep 90.2, full fine-tuning 89.4.** Hyper-parameters are picked on a random 90/10 split of the
training set. This "FGVC" suite (CUB, NABirds, Flowers, Dogs, Cars) is the standard parameter-efficient benchmark, so a
frozen-DINOv2 head should be compared with these numbers, with the pretraining difference stated.

---

## 2. Class-specific / query-based attention heads, attentive probing, prototypes

**CSRA** — Zhu, Wu, "Residual Attention: A Simple but Effective Method for Multi-Label Recognition", ICCV 2021,
pp. 184–193, doi:10.1109/ICCV48922.2021.00025 (Crossref). FT (ar5iv): one spatial attention map **per class**,
computed from that class's own classifier weight with a softmax temperature, pools a class-specific feature that is
added to class-agnostic average pooling ("with only 4 lines of code" it improves pretrained models without extra
training). Multi-label only (VOC2007, MS-COCO, NUS-WIDE, WIDER-Attribute). → Natural ablation for us: class-specific
pooling for the top-K candidates *without* rival conditioning.

**ML-Decoder** — Ridnik, Sharir, Ben-Cohen, Ben-Baruch, Noy, WACV 2023, pp. 32–41, doi:10.1109/WACV56688.2023.00012.
FT: transformer-decoder head **without self-attention** (cost linear in the number of queries) and with group decoding
(K group queries serve N classes); works for single-label, multi-label and zero-shot. Abstract: MS-COCO 91.4 mAP,
NUS-WIDE ZSL 31.1 mAP, ImageNet ResNet-50 80.7 %; Appendix J adds CIFAR-100 and Stanford Cars. No Dogs.

**Query2Label** — Liu, Zhang, Yang, Su, Zhu, arXiv:2107.10834 (arXiv only). ABS: one label-embedding query per class
cross-attends to the backbone feature map; MS-COCO 91.3 mAP. Multi-label.

> Gap these three leave open: their queries are the **same for every image and cover all classes**. None restricts the
> queries to an image's own top-K candidates, and none lets a candidate's query see the competing candidates.

**Attentive probing of frozen features**
- **I-JEPA** (Assran et al., CVPR 2023, pp. 15619–15629, doi:10.1109/CVPR52729.2023.01499). FT: evaluated by a
  **linear** probe ("the model weights are frozen and a linear classifier is trained on top"). It is *not* an
  attentive-probe paper — the brief's assumption should be corrected to V-JEPA.
- **V-JEPA** (Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video",
  TMLR 2024 — OpenReview id QaCCuDfBk2, "Accepted by TMLR"). FT: "we learn a cross-attention layer with a learnable
  query token. The output of the cross-attention layer is then added back to the query token (residual connection),
  and then fed into two-layer MLP with a single GeLU activation, followed by a LayerNorm, and finally a linear
  classifier." One generic query.
- **CAE** (Chen et al., "Context Autoencoder for Self-supervised Representation Learning", IJCV 132(1):208–223, 2024,
  doi:10.1007/s11263-023-01852-4). Introduced attentive probing for MIM features (frozen encoder + cross-attention
  pooling + linear classifier); V-JEPA cites it for this.
- **AIM** (El-Nouby et al., ICML 2024, PMLR 235:12371–12384). Attentive probe with a learnable query on a frozen trunk;
  15 downstream benchmarks; pretraining on DFN-2B+ (DFN-2B filtered from DataComp-12.8B, with a small share of IN-1k).
- **Efficient Probing (EP)** — Psomas, Christopoulos, Baltzi, Kakogeorgiou, Aravanis, Komodakis, Karantzalos,
  Avrithis, Tolias, "Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency", **ICLR 2026**
  (OpenReview id PXo0gtT7Al, "ICLR 2026 Poster"; arXiv:2506.10178v3). FT: unifies AIM, CAE, CAPI, V-JEPA, SimPool and
  other poolings in one framework and proposes a multi-query cross-attention probe with fewer projections.
  **Table 2 (ImageNet-1k): DINOv2 ViT-B/14 — k-NN 81.8, linear 83.2, EP 84.0 (+0.8); ViT-L/14 — 85.2 → 85.6 (+0.4)**,
  while MAE ViT-S/16 goes 47.4 → 64.6 (+17.2). Datasets include CUB, Aircraft, Cars, Food-101 — **not Stanford Dogs**.
  *Implication for us:* on DINOv2, attentive pooling by itself adds < 1 point on ImageNet; a gain from a
  rival-conditioned head must therefore come from the conditioning, and EP (same parameter budget) is the baseline to
  beat.

**Prototype methods**
- **ProtoPNet** — Chen, Li, Tao, Barnett, Rudin, Su, "This Looks Like That: Deep Learning for Interpretable Image
  Recognition", NeurIPS 2019. FT (ar5iv): CUB cropped to the bird box, single backbones about 80 %, a three-backbone
  combination 84.8 % (Table 1); also a Stanford Cars case study. 10 prototypes per class.
- **PIP-Net** — Nauta, Schlötterer, van Keulen, Seifert, CVPR 2023, pp. 2744–2753, doi:10.1109/CVPR52729.2023.00269.
  FT (CVF PDF): ConvNeXt-tiny CUB 84.3 ± 0.2, Cars 88.2 ± 0.5, Pets 92.0 ± 0.3 (Table 1, 3 seeds); ProtoPNet listed at
  CUB 79.2 / Cars 86.1. Can abstain on out-of-distribution inputs (Table 2). No Dogs.

---

## 3. Confusion-aware learning

**Pairwise Confusion (PC)** — Dubey, Gupta, Guo, Raskar, Farrell, Naik, ECCV 2018, LNCS pp. 71–88,
doi:10.1007/978-3-030-01258-8_5 (Crossref; arXiv v1 was titled "Training with Confusion for FGVC"). FT (ar5iv, v3):
a training-only "Euclidean Confusion" term pulls together the predicted class distributions of two images with
*different* labels. **Stanford Dogs, Table 2(F), mean of 5 trials: DenseNet-161 81.18 → 83.75, Bilinear CNN
82.13 → 83.04, ResNet-50 69.92 → 73.35.** Table 3: on ImageNet-Dogs (117 dog classes of IN-1k) GoogLeNet gains
62.35 → 64.17, against 71.85 → 72.09 on a random 117-class subset — the confusable dog classes gain most.

**MaxEnt** — Dubey, Gupta, Raskar, Naik, "Maximum-Entropy Fine-Grained Classification", NeurIPS 2018 (arXiv
1809.05934 comment; Semantic Scholar). FT (ar5iv): maximise output entropy while fine-tuning, motivated by the small
visual diversity of FGVC data. **Stanford Dogs, Table 1(E), mean of 6 trials: DenseNet-161 81.18 → 83.63,
B-CNN 82.13 → 83.18, ResNet-50 69.92 → 73.56, VGG-16 61.92 → 65.44, GoogLeNet 55.76 → 62.01.** Table 3:
Dogs-ImageNet ResNet-50 73.81 → 75.66 vs Random-ImageNet 82.01 → 82.29.
Note the ResNet-50 baseline of 69.92 in both papers, against ~88 in API-Net: the recipe, not the backbone, sets
most of the Dogs number.

**HD-CNN** — Yan et al., ICCV 2015, pp. 2740–2748, doi:10.1109/ICCV.2015.314. ABS: coarse classifier for easy
classes, dedicated fine classifiers for confusable ones, conditional execution; CIFAR-100 and ImageNet only.
Our per-image top-K rival set is the dynamic counterpart of HD-CNN's fixed fine groups.

**Seesaw Loss** — Wang et al., CVPR 2021, pp. 9690–9699, doi:10.1109/CVPR46437.2021.00957. ABS: per-class-pair
mitigation (frequency ratio) and compensation (penalise misclassified negatives) factors; LVIS only.

**Online Label Smoothing (OLS)** — Zhang, Jiang, Hou, Wei, Han, Li, Cheng, "Delving Deep Into Label Smoothing",
IEEE TIP 30:5984–5996 (2021), doi:10.1109/TIP.2021.3089942. FT (partly): soft labels accumulated from the model's
own predictions per target class; CIFAR-100, ImageNet, CUB, Flowers-102, Cars, Aircraft (Tables V–VI; values not
transcribed). No Dogs.

**FuDD** — Esfandiarpoor, Bach, "Follow-Up Differential Descriptions: Language Models Resolve Ambiguities for Image
Classification", ICLR 2024 (bib entry in refs_core.bib, key `esfandiarpoor2024fudd`). FT (arXiv HTML v2), Sec. 3.2:
the ambiguous set C_A is "the k most similar classes" to the image under CLIP; an LLM then writes, for every pair in
C_A, captions of their visual differences. **Stanford Dogs, zero-shot (Table 1): ViT-B/32 58.01 → 60.03 (k = 10)
/ 60.28 (k = |C|); ViT-L/14@336px 73.61 → 75.99 / 76.34.** Table 2, Dogs: differential 60.31 / 75.96 vs
non-differential 59.30 / 75.41. → The closest conceptual precedent for us (top-k candidates + rival-pair-conditioned
evidence), but in text space, zero-shot, and ~15 points below supervised FGVC on Dogs.

**Menon & Vondrick** — "Visual Classification via Description from Large Language Models", ICLR 2023 (Semantic
Scholar venue; arXiv 2210.07183). FT (ar5iv), Table 1, ViT-L/14@336px vs class-name CLIP: ImageNet 76.16 vs 72.97,
CUB 65.26 vs 63.41, Oxford Pets 91.69 vs 88.20. No Dogs in the paper; FuDD reports these descriptors on Dogs as
57.86 / 74.02 (third-party). Descriptors ignore which classes compete — exactly what FuDD fixes.

---

## 4. Backbones, pretraining data, and the Stanford Dogs ⊂ ImageNet problem

**Why this section matters.** Krause et al. (ECCV 2016, key `krause2016unreasonable` in refs_core.bib), FT Sec. 5:
"For experiments on Stanford Dogs, we use the CNN of [25], which is pre-trained on a version of ILSVRC with dog data
removed, since Stanford Dogs is a subset of ILSVRC training data." The TensorFlow Datasets catalogue page
(tensorflow.org/datasets/catalog/stanford_dogs) likewise says the set was "built using images and annotation from
ImageNet" (120 breeds, 20,580 images, 12,000 train / 8,580 test, no validation split). Their dog-free reference
numbers (Table 1): **80.6** (Stanford train only), **84.0** with L-Dog web data, **85.9** with multi-crop.
Consequence: every supervised IN-1k / IN-21k backbone in this table has been trained on Stanford Dogs *test*
images with their breed labels. None of the FGVC papers in sections 1–1e mentions this.

**DINOv2** — Oquab et al., TMLR 2024 (arXiv 2304.07193v2; OpenReview a68SUt6zFt). FT, quoted:
- "Our selection of curated datasets is detailed in the appendix (Table 15) and contains ImageNet-22k, the train
  split of ImageNet-1k, Google Landmarks and several fine-grained datasets."
- "We apply the copy detection pipeline of Pizzi et al. (2022) to the uncurated data and remove near-duplicate
  images. We also remove near-duplicates of images contained in the test or validation set of any benchmark used in
  this work."
- App. A.3, relative deduplication: "we discard remaining images of our self-deduplicated data source that are too
  similar to train and test splits of our evaluation datasets" (1.3B → 1.1B after self-dedup → 744M after relative
  dedup).
- Table 15 (LVD-142M, 142,109,386 images): **ImageNet-22k "as is" 14,197,086**; ImageNet-22k sample-retrieved
  56,788,344; ImageNet-1k train sample-retrieved 40,997,344; retrieval seeds also include Caltech101, CUB, DTD,
  FGVC-Aircraft, Flowers, Food-101, Oxford-IIIT Pets, Stanford Cars, SUN397, VOC2007. Table 18 marks ImageNet-22k as
  used for pretraining as is and for retrieval, not for evaluation.
- **Stanford Dogs appears nowhere in the paper**, so it was never a de-duplication target; the de-duplication
  applies to the *uncurated web pool*, not to the ImageNet-22k part that is included as is.
- Conclusion (our inference, to be checked empirically): DINOv2 has almost certainly seen the Stanford Dogs test
  images — without labels — during pretraining. Its frozen-probe numbers on Pets (96.2 / 96.6 / 96.7 for B/L/g,
  Table 8) and CUB (89.6 / 90.5 / 91.6) also benefit from those sets being retrieval seeds. ImageNet-1k linear:
  84.5 / 86.3 / 86.5 (Table 4).

**Registers** — Darcet, Oquab, Mairal, Bojanowski, "Vision Transformers Need Registers", ICLR 2024. FT: artifact
tokens with ~10× norm on ~2 % of patches (redundant background), holding global information; 4 register tokens
remove them. DINOv2 ViT-L/14 linear ImageNet 84.3 → 84.8, ADE20k 46.6 → 47.9 (Table 2a). → For an attention head
over DINOv2 patch tokens this is a first-order issue: use the register checkpoints or mask high-norm tokens.

**DINOv3** — Siméoni et al., arXiv 2508.10104v1 (2025). FT, Sec. 3: ~17B Instagram images → LVD-1689M (hierarchical
k-means curation) + a retrieval-curated part + **raw ImageNet-1k, ImageNet-22k and Mapillary**; homogeneous
ImageNet-1k batches are 10 % of training. Searching the full HTML for dedup / duplicate / copy detection /
contamination returns nothing: **no de-duplication against evaluation sets is described.** Fine-S linear (Table 22),
ViT-7B/16: Pets 97.0, CUB 92.4, average 93.0 (DINOv2 g/14 under the same protocol: 96.8, 91.6, 92.6). Uses 4
registers and Gram anchoring. No Dogs.

**CLIP** — Radford et al., ICML 2021, PMLR 139:8748–8763. FT Sec. 5 "Data Overlap Analysis": a purpose-built
near-duplicate detector (App. C) over 35 datasets; median overlap 2.2 %, average 3.2 %, largest accuracy gain 0.6 %
(Birdsnap, 12.1 % overlap). Stanford Dogs is not in the paper. → A template for the overlap check we should run
(Stanford Dogs test vs ImageNet-21k and vs LVD-142M neighbours).

**ViT** — Dosovitskiy et al., ICLR 2021. FT Sec. 4.1: "We de-duplicate the pre-training datasets w.r.t. the test sets
of the downstream tasks following Kolesnikov et al. 2020." Dogs is not one of those tasks, so the IN-21k ViT-B/16
initialisation used by TransFG, FFVT, RAMS-Trans, SIM-Trans, IELT, MP-FGVC, DCAL and VPT is not de-duplicated
against it.

**ResNet** (CVPR 2016, doi:10.1109/CVPR.2016.90), **ConvNeXt** (CVPR 2022, doi:10.1109/CVPR52688.2022.01167;
ConvNeXt-XL IN-22k 87.8 % IN-1k), **EfficientNet** (ICML 2019, PMLR 97:6105–6114; transfer Table 5: Cars 94.7,
Aircraft 92.9, Pets 95.4 — no Dogs): bibliographic and backbone references only.

**Which methods in the table use ImageNet-21k/22k.** ViT-B/16 IN-21k: TransFG, FFVT, RAMS-Trans, SIM-Trans, IELT
(README), MP-FGVC (+ CLIP text encoder), DCAL (ViT variant), VPT. Swin IN-22k: ViT-NeT's Swin variant, H3Former on
CUB/NABirds/Flowers (it switches to IN-1k Swin-B for Dogs). Label-free IN-22k: DINOv2, DINOv3 (and I-JEPA ViT-H/14).
All the rest use supervised IN-1k, which contains Stanford Dogs as well. Only Krause et al. remove the dog data.

---

## 5. Reliability: calibration, selective prediction, OOD and open-set

**Temperature scaling** — Guo, Pleiss, Sun, Weinberger, ICML 2017, PMLR 70:1321–1330. FT: ECE with M = 15 bins
(Table 1): Birds ResNet-50 9.19 → 1.85, Cars ResNet-50 4.3 → 2.35, ImageNet DenseNet-161 6.28 → 1.99. Their CUB and
Cars validation sets are halves of the test sets (5994/2897/2897; 8041/4020/4020) — not acceptable for us; on Dogs
the temperature must be fitted on a held-out part of the 12,000 training images.

**ECE** — Naeini, Cooper, Hauskrecht, AAAI 2015, doi:10.1609/aaai.v29i1.9602 (Crossref; Europe PMC PMC4410090).
Full text not reachable (AAAI site behind a bot check); cited for ECE via Guo et al. 2017 Sec. 2. verified = partial.

**Selective classification** — Geifman, El-Yaniv, NeurIPS 2017. ABS: reject option with a user-set risk guaranteed
with high probability; ImageNet top-5 risk 2 % at almost 60 % coverage. → Report risk–coverage for "abstain when the
two top rivals cannot be separated".

**MSP** — Hendrycks, Gimpel, ICLR 2017 (ABS). **Energy** — Liu, Wang, Owens, Li, NeurIPS 2020 (ABS: −18.03 % FPR95
vs softmax on CIFAR-10 WRN).

**Open-set / SSB** — Vaze, Han, Vedaldi, Zisserman, ICLR 2022 (oral). FT: open-set AUROC tracks closed-set accuracy;
maximum-logit score is a strong baseline; the Semantic Shift Benchmark has CUB, Stanford Cars and FGVC-Aircraft
(Easy/Medium/Hard) and ImageNet-1k vs ImageNet-21K-P (Easy/Hard by WordNet distance). **No dog split** — an
unseen-breed protocol for Dogs has to be built by us.

---

## 6. Adaptive computation

**GFNet** — Wang, Lv, Huang, Song, Yang, Huang, NeurIPS 2020 (ABS): glance at low resolution, then RL-chosen crops,
stop when confident; ImageNet only (MobileNet-V3 latency −20 % on iPhone XS Max). **AdaFocus** (ICCV 2021,
doi:10.1109/ICCV48922.2021.01594), **AdaFocus V2** (CVPR 2022, doi:10.1109/CVPR52688.2022.01943), **AdaFocusV3**
(ECCV 2022, doi:10.1007/978-3-031-19772-7_14): the same idea for video (ABS only). → Relevant as the "only pay for
the second stage when the first is unsure" pattern; none is fine-grained.

---

## 7. Summary for the study

**Best verified Stanford Dogs top-1 in this table** (the other agent's four papers excluded):
SR-GNN 97.3 and CAP 96.1 (Xception, IN-1k, 224 px — outliers, see 1c); H3Former 95.8 (Swin-B IN-1k, 448;
arXiv, "accepted by IEEE TIP"); ViT-NeT 93.6 (DeiT-B, 448); Relational Proxies 92.75 (ResNet-50 IN-1k);
RAMS-Trans 92.4 and TransFG 92.3 (ViT-B/16 IN-21k); WS-DAN 92.2; LGTF 92.1. Re-runs by MP-FGVC put TransFG / IELT /
SIM-Trans at 90.5 / 90.6 / 89.2 (1–2 points below their papers). Frozen ViT-B/16 IN-21k (VPT Table 14): linear
86.2, bias tuning 91.2, full fine-tuning 89.4. Dog-free pretraining (Krause 2016): 80.6–85.9. Zero-shot CLIP
L/14@336 with FuDD: 76.34. Confusion-aware CNN losses: 83.6–83.8 (DenseNet-161).

**Tsinghua Dogs:** no 2024–2026 paper outside the other agent's four was found to report it.

**The five works most relevant to top-K re-ranking with rival-conditioned attention over frozen DINOv2 patch tokens:**
1. FuDD (ICLR 2024) — top-k ambiguous classes + pairwise differential evidence; reports Stanford Dogs.
2. ML-Decoder (WACV 2023), with CSRA and Query2Label — class-query cross-attention heads; ours restricts the queries
   to the K candidates and lets each see its rivals.
3. Efficient Probing (Psomas et al., ICLR 2026), with the V-JEPA / AIM attentive probes — the equal-budget frozen
   baseline; on DINOv2 ViT-B/14 it adds only +0.8 on ImageNet, so a gain must come from the rival conditioning.
4. API-Net (AAAI 2020), with DCAL's pair-wise cross-attention — attention conditioned on a confusable counterpart,
   but between images and at training time only.
5. DINOv2 (TMLR 2024) + Registers (ICLR 2024) — the backbone, its test-set exposure, and the patch-token artifacts.
Protocol anchors next to these: VPT's frozen Dogs numbers and Krause et al.'s dog-free pretraining.
Prior work on top-K re-ranking for FGVC itself is covered by the novelty-search agent, not here.
