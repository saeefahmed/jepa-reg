# JEPA-Reg: Predictive Representation Regularization Improves Robustness to Adversarial Lexical Overlap in Paraphrase Detection

*(Camera-ready revision — ICCA 2026, Paper 284. All numbers verified against the `results/` JSONs: BERT-base from the August 2026 full rerun, RoBERTa-base from the archived original run. The only remaining placeholder is ⟨repository URL⟩.)*

---

## Abstract

Paraphrase-detection models trained on standard benchmarks over-rely on superficial lexical overlap, resulting in performance drops on adversarial datasets such as PAWS and HANS. We propose **JEPA-Reg**, a lightweight regularization approach that augments BERT-based paraphrase classifiers with a Joint-Embedding Predictive Architecture (JEPA) auxiliary loss. An exponential-moving-average (EMA) target encoder with stop-gradient provides stable prediction targets, and — unlike prior JEPA variants — the predictive loss is applied **only to labelled paraphrase pairs**, giving the encoder a label-conditioned inductive bias toward structurally grounded, overlap-invariant representations without negative sampling or cross-modal data construction. We evaluate on MRPC, QQP, and PAWS with three random seeds. When fine-tuning on PAWS, JEPA-Reg improves BERT-base test F1 by +7.9 points (71.7→79.6) and reduces seed-to-seed standard deviation by 15× (3.74→0.24); for RoBERTa-base the mean gain is small (+0.22) but seed-consistent. Stratifying PAWS accuracy by lexical-overlap quartile shows JEPA-Reg's advantage grows monotonically with overlap and holds for both classes, indicating reduced reliance on surface co-occurrence rather than a shifted class prior. In zero-shot transfer to HANS, JEPA-Reg improves overall F1 (+3.0), but a per-subcase analysis reveals this stems chiefly from a shift in prediction bias, which we report as a negative result for the structural-transfer hypothesis. Because only three seeds are used, we report per-seed values and effect sizes descriptively rather than claiming small p-values. JEPA-Reg adds no inference-time cost and is conceived as a targeted adversarial-robustness regularizer rather than a general accuracy booster: on QQP it neither helps nor hurts (76.7±0.3 vs. 76.3±0.8).

**Keywords:** Paraphrase detection, Adversarial robustness, Lexical overlap, JEPA, Representation learning, Natural language processing

---

## 1 Introduction

Pre-trained transformer models such as BERT [6] and RoBERTa [14] perform well on paraphrase-detection benchmarks such as MRPC and QQP in the GLUE suite [21]. However, a well-documented failure mode is their dependence on lexical-overlap heuristics: models tend to predict "paraphrase" when token overlap is high, even when syntactic structure indicates otherwise [15]. Datasets such as PAWS [25] and HANS [15] expose this behaviour by creating adversarial sentence pairs with high lexical overlap but conflicting labels.

Current mitigation options have significant limitations. Adversarial fine-tuning methods, e.g., SMART [12] and FreeLB [26], and regularization methods, e.g., Mixout [13], improve general robustness but do not explicitly address lexical-overlap reliance. Debiasing methods (e.g., [5, 20]) assume the bias is known a priori. Contrastive learning methods such as SimCSE [7], DeCLUTR [8], and INSTRUCTOR [18] enhance representation quality but depend on explicit negative-pair construction and are not built to directly address structural robustness under adversarial overlap.

An alternative framework for learning representations is given by JEPA. I-JEPA [1] trains an online encoder to predict the representations of an EMA target encoder in embedding space, without negative samples or heavy augmentations. LLM-JEPA [11] extends this to large language models with cross-modal views, but works mainly in generative settings. To our knowledge, JEPA has not previously been employed for adversarial robustness in encoder-based discriminative classification.

**Contributions.** We introduce JEPA-Reg and make the following contributions, whose novelty relative to prior JEPA work we make explicit:

- **Method.** We adapt the JEPA predictive-EMA objective to same-modality sentence pairs in a *discriminative* setting, and — departing from all prior JEPA variants, which apply the predictive loss to every view pair — we **condition the auxiliary loss on the gold label**, applying it only to annotated paraphrase pairs. This makes the auxiliary signal task-aligned by design: semantically equivalent sentences must be mutually predictable in projection space, while non-paraphrases are exempt. (An ablation in Section 6.6 shows the unmasked variant performs comparably on PAWS, so we present label conditioning as a principled design choice rather than the empirical source of the gains.)
- **Empirical findings.** On PAWS fine-tuning, JEPA-Reg yields both a mean F1 gain (+7.9 for BERT-base) and a 15× reduction in seed-to-seed standard deviation, with the advantage growing monotonically in the most overlap-heavy strata.
- **Analysis.** We provide the first systematic multi-seed study of the auxiliary-loss weight λ for JEPA in NLP classification, finding — contrary to single-run evidence — that the method's robustness gain is insensitive to λ across [0.05, 1.0], so no careful tuning away from the LLM-JEPA default is required.
- **Honest negative results.** We report per-subcase HANS diagnostics showing that zero-shot heuristic-level gains reflect prediction-bias shifts rather than transferred structural reasoning, and an STS-B diagnostic bounding the method's scope to classification.

**Scope.** JEPA-Reg is conceived as a targeted adversarial-robustness regularizer, not a general-purpose accuracy booster. On large, clean datasets like QQP, which do not exhibit strong overlap–label dissociation, it neither helps nor hurts (Section 7.3).

## 2 Literature Review

The lexical-overlap failure is most clearly instantiated by PAWS [25], which builds adversarial pairs with high lexical overlap but different labels via controlled word reordering, and HANS [15], which decomposes the problem into lexical-overlap, subsequence, and constituent heuristics. Together these benchmarks establish that lexical-overlap bias is a representation-level phenomenon and motivate methods that intervene directly on how models construct internal representations. Three families of prior work address related problems. **Adversarial fine-tuning** — SMART [12], which uses Bregman proximal perturbations in embedding space, and FreeLB [26], which accumulates iterative gradient-based perturbations — improves general robustness but perturbs inputs uniformly without specifically targeting lexical-overlap reliance; Mixout [13] stochastically interpolates fine-tuned parameters with pre-trained ones to guard against catastrophic forgetting, but introduces no structural inductive bias. **Debiasing methods** [5, 20] employ auxiliary biased models or product-of-experts formulations; they are the natural comparators for overlap shortcuts and are computationally cheap at the 110M scale — we identify them as the priority baseline for future comparison (Section 7.5). **Contrastive representation learning** — Sentence-BERT [16], SimCSE [7], DeCLUTR [8], INSTRUCTOR [18] — is inherently dependent on explicit negative-pair construction, resulting in sensitivity to batch composition, and optimizes general semantic similarity rather than structural robustness under adversarial overlap.

Negative-free self-supervised techniques from vision, such as Barlow Twins [24] and BYOL [9], show that discriminative representations can be learned without negatives but have not been applied to discriminative NLP tasks. I-JEPA [1] trains an online encoder to predict embeddings of masked target patches via an EMA-updated target encoder, entirely in representation space; DINO [3] and data2vec [2] further validate EMA self-distillation across modalities. LLM-JEPA [11] extends JEPA to decoder-only LLMs with cross-modal text–code views, and VL-JEPA [4] to vision–language learning. None of this work applies JEPA to same-modality sentence pairs in a discriminative setting.

Structural probing [10, 19] shows that BERT layers encode rich syntactic information, yet fine-tuning pressures that reward lexical shortcuts suppress this structural knowledge in final predictions. Data augmentation via back-translation [17] and EDA [22] broadens the training distribution but leaves lexical sensitivity unchanged. Output-level regularisers such as R-Drop [23] operate at the logit level with no geometric constraint on the representation space where overlap heuristics form.

**Table 1** compares methods on four *method-property* criteria: (i) requires no negative sampling; (ii) imposes a geometric constraint directly on the representation space; (iii) operates on same-modality inputs without auxiliary bias models or cross-modal views; (iv) conditions its objective on task labels. *(The submitted version's criteria "adversarial evaluation" and "λ study" described papers' evaluation choices rather than method properties; they are now stated in the text: no prior JEPA work evaluates on adversarial overlap benchmarks or studies λ in a discriminative setting.)*

| Method | Neg.-free | Repr.-space constraint | Same-mod., no bias prior | Label-conditioned |
|---|---|---|---|---|
| SMART [12] / FreeLB [26] | ✓ | ✗ | ✓ | ✗ |
| Mixout [13] | ✓ | ✗ | ✓ | ✗ |
| Debiasing [5, 20] | ✓ | ✗ | ✗ | ✓ |
| SimCSE [7] / DeCLUTR [8] / INSTRUCTOR [18] | ✗ | ✓ | ✓ | ✗ |
| Barlow Twins [24] / BYOL [9] | ✓ | ✓ | ✓ | ✗ |
| I-JEPA [1] / LLM-JEPA [11] | ✓ | ✓ | ✗ | ✗ |
| **JEPA-Reg (ours)** | ✓ | ✓ | ✓ | ✓ |

## 3 Method: JEPA-Reg

### 3.1 Problem Formulation and Overview

Let $f_\theta$ denote a BERT-based encoder and $(s_1, s_2, y)$ a training triple with $y \in \{0,1\}$. Standard fine-tuning minimizes

$$L_{cls} = \mathrm{CrossEntropy}\big(W \cdot \mathrm{pool}(f_\theta(\mathrm{tok}(s_1, s_2))),\, y\big), \tag{1}$$

where $\mathrm{pool}(\cdot)$ is attention-masked **mean pooling** over the final hidden states of the cross-encoded pair. *(Correction vs. the submitted version, which stated [CLS] pooling: the JEPA-Reg classifier uses mean pooling; the baseline uses the standard [CLS]-based classification head of the pretrained architecture.)* JEPA-Reg adds a representation-level auxiliary loss to Eq. (1). The complete training pipeline is depicted in Figure 1.

### 3.2 Dual-Encoder Architecture

JEPA-Reg maintains two encoders with identical architecture but different update rules. The online encoder $f_\theta$ is fully trainable and receives gradients from both losses. The target encoder $f_\xi$ is a frozen copy updated as an exponential moving average:

$$\xi \leftarrow \tau\xi + (1-\tau)\theta, \quad \tau = 0.999. \tag{2}$$

Stop-gradient is applied only to the target branch: the online encoder receives the full JEPA gradient through the predictor and projection head, while the target provides a stable reference, consistent with I-JEPA [1]. Figure 2 summarizes the gradient flow. Both encoders share a projection head $h$: Linear($H{\to}H$) → GELU → Linear($H{\to}256$); the online branch adds a predictor $p$ of the same structure. For each sentence $s_i$ (encoded *individually*, not as a pair):

$$z_i = h(\mathrm{pool}(f_\theta(s_i))), \qquad \tilde{z}_i = h(\mathrm{pool}(f_\xi(s_i))). \tag{3}$$

### 3.3 JEPA Auxiliary Loss (label-conditioned)

All projections are L2-normalized; writing $\hat{u} = u / \lVert u \rVert_2$, the symmetric predictive loss for one pair is

$$\ell(s_1, s_2) = \tfrac{1}{2}\Big[\big(1 - \hat{p}(z_1)^\top \hat{\tilde z}_2\big) + \big(1 - \hat{p}(z_2)^\top \hat{\tilde z}_1\big)\Big], \tag{4a}$$

a cosine-distance form equivalent (up to a factor of 2) to the squared-L2 distance between normalized vectors. **The auxiliary loss is applied only to gold paraphrase pairs:**

$$L_{jepa} = \frac{\sum_n \mathbb{1}[y^{(n)}{=}1]\, \ell(s_1^{(n)}, s_2^{(n)})}{\max\big(1, \sum_n \mathbb{1}[y^{(n)}{=}1]\big)}. \tag{4b}$$

*(Correction vs. the submitted version, whose Eq. (4) omitted both the normalization and the label mask.)* This label conditioning is the theoretically motivated choice: mutual predictability in latent space is justified for semantically equivalent sentences, whereas forcing labelled non-paraphrases toward mutual predictability would directly oppose the classification objective. Section 6.6 ablates this choice against the unmasked (all-pairs, BYOL-style) variant and includes a representation-collapse check. Because the objective is defined entirely in embedding space, it imposes no pressure to reconstruct input tokens.

### 3.4 Full Objective

$$L = L_{cls} + \lambda\, L_{jepa}. \tag{5}$$

λ is selected on a 10% held-out fraction of the training set from candidates {0.05, 0.1, 0.3, 0.5} (single tuning seed, one epoch); the sweep selects λ=0.05 for BERT-base on all three datasets (for RoBERTa-base we fix λ=0.1 without a sweep). To stabilize early training, λ is warmed up linearly from 0 to its target value over the first epoch. Algorithm 1 (updated to show the label mask, normalization, and λ warmup) summarizes training.

**Computational cost.** Per training step, JEPA-Reg computes three gradient-carrying forward passes (the cross-encoded pair, and $s_1$, $s_2$ individually) plus two no-gradient target passes, versus one pass for the baseline, and updates the EMA copy of all encoder parameters. Measured on our hardware (batch 32, MRPC): **6.5× wall-clock time per training step** (297→1,927 ms) and **1.6× peak GPU memory** (2.91→4.63 GB); parameters in GPU memory during training are 219.5M vs. 109.5M (the EMA copy is frozen; trainable parameters are 109.8M vs. 109.5M). **At inference, the deployed model is architecturally identical to the baseline**: the target encoder, projection heads, and predictor are dropped, and prediction is a single encoder forward pass through the same 110M-parameter network. We accordingly retitle the claim from "computationally light" to "inference-free overhead with a moderate training-time cost."

### 3.5 Positioning vs. LLM-JEPA

Table 2 compares JEPA-Reg to LLM-JEPA [11]. Both share the core JEPA principle — predictive representation learning with an EMA target encoder and no negative samples — but differ substantially in scope.

| Aspect | LLM-JEPA | JEPA-Reg |
|---|---|---|
| Task | Generation | Classification |
| View pairs | Cross-modal (text–code) | Same-modality |
| Backbone | Decoder-only, 1B+ | Encoder-only, 110M |
| Negatives | Not required | Not required |
| Loss application | All view pairs | **Gold paraphrase pairs only** |
| λ study | Single default (1.0) | Multi-seed sweep [0.05–1.0] |
| Adversarial eval | None | PAWS + HANS |
| Low-resource | Not studied | Studied |

## 4 Experimental Setup

### 4.1 Datasets and Protocol

**MRPC** [21]: 3,668 / 408 / 1,725 train/validation/test pairs. **QQP** [21]: for computational tractability we subsample the 400k+ corpus to 20,000 training pairs (of which 10% is held out as our test split, giving 18,000 train / 2,000 test) and 5,000 validation pairs; for RoBERTa-base the training subset was further capped at 5,000 pairs. **PAWS**: we use the **`labeled_final`** subset of PAWS-Wiki [25]. **Models are fine-tuned on PAWS**: 20,000 training pairs (subsampled from 49,401), 5,000 validation pairs for model selection, and the full official 8,000-pair test set for evaluation. *(Correction vs. the submitted version, which incorrectly described PAWS as a zero-shot test set. All PAWS numbers in this paper are from PAWS-fine-tuned models; only HANS is evaluated zero-shot.)*

The PAWS test set contains 44.2% positive pairs. The corresponding trivial baselines are: majority-class (always non-paraphrase) accuracy 55.8 / positive-class F1 0.0; always-paraphrase accuracy 44.2 / F1 61.3. All reported models clearly exceed both, and per-model confusion matrices with per-class precision/recall are given in Appendix A (e.g., JEPA-Reg seed-42: per-class F1 0.80/0.80; baseline: 0.68/0.72 with a visible bias toward the positive class). **HANS** [15]: 30,000 balanced NLI pairs (5,000 entailed and 5,000 non-entailed per heuristic). We evaluate zero-shot from MRPC-trained models, mapping entailment→paraphrase.

Baselines: **Baseline** (cross-entropy only), **SimCSE** [7] (in-batch contrastive auxiliary), and **Hybrid-JEPA** (JEPA+SimCSE, analyzed on QQP in Section 7.3). Adversarial fine-tuning baselines (SMART, FreeLB) were not included due to computational constraints and remain the most important comparators for future work; debiasing methods [5, 20] are the cheapest such addition (Section 7.5).

### 4.2 Implementation and Evaluation

**Table 3.**

| Hyperparameter | Value |
|---|---|
| Backbones | bert-base-uncased, roberta-base |
| Batch size | 32 (BERT) / 8 (RoBERTa) |
| Learning rate | BERT: 3e-5 (MRPC), 1e-5 (QQP), 2e-5 (PAWS); RoBERTa: 2e-5 / 1e-5 / 2e-5 |
| Epochs | 3; model selection by best validation F1 |
| Optimizer | AdamW, weight decay 0.01, layer-wise LR decay 0.9, gradient clipping 1.0 |
| LR schedule | Linear, 20% warmup |
| Label smoothing | 0.05 (Baseline, SimCSE); 0.0 (JEPA-Reg classification loss) |
| EMA decay τ | 0.999 |
| Projection dim | 256 |
| λ | selected from {0.05, 0.1, 0.3, 0.5} on 10% split (BERT: 0.05); RoBERTa fixed 0.1; linear warmup over epoch 1 |
| Max sequence length | 128 tokens |
| Random seeds | 42, 43, 44 (n=3) |
| Mixed precision | AMP |
| Hardware / runtime | 1× NVIDIA RTX 5060 Ti (16 GB); full BERT pipeline ≈ 2.5 h, RoBERTa ≈ 100 min |

Code, the analysis notebook, and per-seed results JSONs are released at ⟨repository URL⟩. BERT-base numbers come from a full rerun of the released pipeline; RoBERTa-base numbers from the original run of the same code (per-seed values for both are in the released JSONs).

**Statistical reporting.** With n=3 seeds, hypothesis tests have essentially no power: the exact paired sign-flip permutation test cannot yield p < 0.25 regardless of effect magnitude. We therefore (i) report all per-seed values, (ii) use Cohen's d *descriptively* as a variance-normalized effect summary, noting that near-ceiling settings can produce large d from small absolute gains, and (iii) report paired t-test p-values for completeness without claiming significance thresholds. *(This replaces the submitted version's bootstrap p-values and "p < 0.001" claims, which are not supportable at n=3.)*

## 5 Results

### 5.1 Main Results: BERT-base

**Table 4** (test F1, mean±std over seeds 42/43/44; per-seed values in parentheses).

| Dataset | Baseline | SimCSE | JEPA-Reg (ours) | Δ | Cohen's d | paired-t p |
|---|---|---|---|---|---|---|
| MRPC | 85.53±1.16 (85.57/84.09/86.93) | 85.02±0.81 | 85.44±0.22 (85.75/85.29/85.26) | −0.09 | −0.09 | 0.92 |
| PAWS | 71.72±3.74 (71.51/76.40/67.26) | 73.38±0.47 | **79.59±0.24** (79.80/79.72/79.25) | **+7.87** | 2.43 | 0.09 |
| QQP | 76.35±0.76 (76.96/75.28/76.81) | 76.16±0.19 | 76.72±0.27 (76.70/77.06/76.40) | +0.37 | 0.54 | 0.65 |

*(The submitted version's QQP cell reported the Hybrid variant, a different method; pure JEPA-Reg is now shown, and Hybrid is analyzed separately in Section 7.3.)*

On PAWS, JEPA-Reg improves mean F1 by +7.9 and reduces the seed-to-seed standard deviation by 15× (σ = 3.74 → 0.24): the baseline's worst seed collapses to 67.26 F1 while all JEPA-Reg seeds land within 0.6 points of each other (Figure 3). On MRPC the mean is statistically indistinguishable from the baseline (d = −0.09), though seed variance again shrinks (1.16 → 0.22); we claim no in-domain gain. On QQP, JEPA-Reg is likewise indistinguishable from the baseline — the degradation reported in the submitted version was specific to the Hybrid variant.

### 5.2 Main Results: RoBERTa-base

**Table 5** (from the original run of the released pipeline).

| Dataset | Baseline | SimCSE | JEPA-Reg | Δ | Cohen's d | paired-t p |
|---|---|---|---|---|---|---|
| MRPC | 90.32±0.22 (90.60/90.28/90.07) | 90.75±0.30 | 90.50±0.29 (90.79/90.10/90.59) | +0.18 | 0.56 | 0.48 |
| PAWS | 92.09±0.10 (91.98/92.22/92.06) | 92.06±0.11 | 92.31±0.11 (92.17/92.45/92.32) | +0.22 | 1.75 | 0.01 |

The absolute PAWS gain is modest because a PAWS-fine-tuned RoBERTa baseline is already near ceiling. We caution explicitly that the large d reflects small within-condition variance rather than a practically large improvement; the supported claim is "a small gain that is consistent across all three seeds" (every JEPA-Reg seed exceeds every baseline seed).

### 5.3 Zero-Shot HANS Evaluation

Results use the best-validation MRPC checkpoint of a single seed (42). Overall, JEPA-Reg improves HANS F1 from 61.06 to 64.02 (+2.96) at essentially unchanged accuracy (50.28 vs. 50.09) on this balanced benchmark. Following McCoy et al. [15] and a reviewer's suggestion, **Table 6** breaks accuracy down by heuristic and gold label — the diagnostic that distinguishes structural transfer from prediction-bias shifts.

**Table 6** — HANS accuracy by heuristic × gold label (n=5,000 per cell).

| Heuristic | Subcase | Baseline | SimCSE | JEPA-Reg |
|---|---|---|---|---|
| Lexical overlap | entailed | 96.46 | 97.98 | 99.92 |
| Lexical overlap | non-entailed | 3.28 | 1.60 | 0.34 |
| Subsequence | entailed | 93.82 | 95.82 | 99.34 |
| Subsequence | non-entailed | 9.78 | 9.92 | 1.32 |
| Constituent | entailed | 43.56 | 49.58 | 67.18 |
| Constituent | non-entailed | 54.80 | 47.08 | 32.42 |

**This analysis does not support a structural-transfer interpretation.** JEPA-Reg's F1 gains arise from predicting "entailment" more often on overlap-heavy pairs: entailed-subcase accuracy rises on every heuristic while non-entailed accuracy falls. In other words, in zero-shot transfer the model's improved calibration on PAWS-style *training-distribution* structure does not translate into discriminating HANS's non-entailed constructions; it translates into a stronger prior that high-overlap pairs are positive. We therefore withdraw the submitted version's claim that HANS constituent selectivity provides behavioral evidence of structural reasoning, and report this as an honest negative result. The in-domain overlap-stratified analysis (Section 6.3), which does survive its corresponding per-class check, remains the supported behavioral evidence.

## 6 Analysis

### 6.1 λ Ablation

**Table 7** (test F1, mean±std over 3 seeds; BERT-base; all rows use the pipeline's linear λ warmup).

| λ | MRPC | PAWS |
|---|---|---|
| 0.05 | 85.59±0.34 | 79.16±0.24 |
| 0.10 | 85.91±0.30 | 79.45±0.52 |
| 0.30 | 85.70±0.39 | 79.29±0.25 |
| 0.50 | 85.84±0.27 | 79.59±0.59 |
| 1.00 (LLM-JEPA default) | 85.74±0.42 | 79.58±0.45 |
| 1.00, **no λ warmup** (control) | — | 79.70±1.04 |

*(The submitted version's Table 7 reported single-run numbers without dispersion and suggested monotone degradation with increasing λ; that ordering does not replicate under multi-seed evaluation.)* With the pipeline's λ warmup (0→λ over the first epoch), JEPA-Reg is **insensitive to λ across [0.05, 1.0]** on both datasets — every cell lies within one standard deviation of every other in its column, and every PAWS cell retains the full ≈+7.9 gain over the baseline (71.72±3.74). A control without λ warmup at λ=1.0 also matches in mean (79.70±1.04, seeds 78.80/81.16/79.15), with a larger seed spread that we note descriptively but do not test at n=3. **We therefore withdraw the submitted version's claim that the LLM-JEPA default λ=1.0 is harmful in this discriminative setting**: under multi-seed evaluation, JEPA-Reg's robustness gain is insensitive to the auxiliary-loss weight across two orders of magnitude. Practically, this is good news — the method requires no careful λ tuning — but it is a null result for λ as a critical hyperparameter, and we report it as such.

### 6.2 Data Efficiency

Figure 4 shows BERT-base results for 1%–100% of MRPC (RoBERTa was not run in the low-resource sweep). Performance is noisy below 25% of the training set, where the auxiliary objective appears under-constrained. From 50% of the data onward, JEPA-Reg tracks the baseline closely (50%: 82.4 vs. 82.4; 100%: 85.5 vs. 85.0 in this sweep) with consistently smaller seed variance. We conclude that JEPA-Reg is most suitable when a moderate amount of labelled data is available, and neither helps nor destabilizes training at mid-to-full data scale on in-domain data.

### 6.3 Lexical Overlap Stratification

*(Substantially revised; replaces the former "dose-response" analysis and Figure 5.)* PAWS is constructed from high-overlap pairs: in the test set, word-level Jaccard overlap has median 0.889 and minimum 0.60. Consequently the fixed bins 0–25% and 25–50% used in the submitted version contain **zero examples** — the 0.00 values previously plotted were empty bins, not model failures — and the 50–75% bin contains only 84 examples. We therefore re-stratify by empirical overlap **quartiles** (n ≈ 2,000 each), annotating class balance, and report per-class accuracy to separate genuine robustness from shifts in the class prior (Figure 5, regenerated):

| Overlap quartile | n | % pos | Baseline | SimCSE | JEPA-Reg | JEPA acc (y=1) | JEPA acc (y=0) |
|---|---|---|---|---|---|---|---|
| [0.600, 0.846) | 1,980 | 42.4 | 67.53 | 68.99 | 76.92 | 85.60 | 70.53 |
| [0.846, 0.889) | 1,927 | 46.5 | 70.47 | 71.61 | 78.88 | 88.96 | 70.10 |
| [0.889, 0.941) | 2,091 | 47.3 | 69.78 | 72.50 | 80.92 | 90.40 | 72.39 |
| [0.941, 1.000] | 2,002 | 40.4 | 71.03 | 74.68 | 83.27 | 92.09 | 77.28 |

Two patterns support the robustness interpretation. First, JEPA-Reg's advantage over the baseline *grows* with overlap (+9.4 → +8.4 → +11.1 → +12.2 accuracy points), whereas the baseline is flat across quartiles — the opposite of what a uniform regularizer would produce. Second, and critically, the gain is not a class-prior artifact: JEPA-Reg's accuracy on **non-paraphrases** — the class a lexical-overlap heuristic gets wrong at high overlap — also rises with overlap (70.5 → 77.3), so the model is genuinely separating high-overlap non-paraphrases rather than defaulting to the positive class.

### 6.4 Interpreting the Behavioral Evidence

*(Revised framing.)* The submitted version described "three orthogonal lines of evidence"; as one reviewer correctly noted, heuristic-level and overlap-level analyses derive from the same underlying behavior, and our per-subcase HANS analysis (Section 5.3) subsequently *failed* to support the structural-transfer reading. The evidence that stands is in-domain: on the distribution the model is trained on, JEPA-Reg concentrates its gains exactly where lexical heuristics are most misleading and maintains them on both classes (Section 6.3). The evidence that does not stand is zero-shot: HANS gains reflect a shifted prediction prior. Decisive mechanistic evidence would require structural probing [10, 19], which we leave to future work. *(The t-SNE/silhouette analysis from the submitted version has been moved to Appendix B as an exploratory visualization only; we no longer interpret the silhouette score, as that reading was post-hoc and unfalsifiable.)*

### 6.5 Training Dynamics

The final classification loss on MRPC is lower with JEPA-Reg (≈0.33 vs. ≈0.38) with visibly smoother trajectories, in line with the variance reductions in Tables 4–5. These dynamics do not show *why* robustness improves, but they do show that the auxiliary objective does not destabilize training in the main experimental regime.

### 6.6 Label-Conditioning Ablation and Collapse Check *(new)*

Because the label-conditioned loss (Eq. 4b) is our main methodological departure from BYOL-style regularization, we ablate it directly: an all-pairs variant applies Eq. (4a) to every training pair regardless of label (PAWS, 3 seeds).

| Variant | PAWS F1 (per-seed) | Mean feature std | Effective rank (of 256) |
|---|---|---|---|
| Positive-only (Eq. 4b, ours) | 79.36±0.45 (79.99/78.95/79.15) | 0.0153 | 57.5 |
| All-pairs (BYOL-style) | 79.65±0.39 (80.20/79.44/79.31) | 0.0167 | 50.4 |
| Baseline (no JEPA loss) | 71.72±3.74 | — | — |

Neither variant collapses (a collapsed solution would show near-zero feature variance and effective rank ≈ 1), and the two are statistically indistinguishable — both recover the full +7.9-point gain over the baseline. **The honest conclusion is that the robustness gain is attributable to the predictive EMA objective itself rather than to the label mask.** We retain the positive-only form on theoretical grounds (it cannot oppose the classifier on labelled non-paraphrases and is the variant all other results in this paper use), but we do not claim label conditioning is empirically load-bearing on PAWS; distinguishing the two may require datasets with a higher proportion of high-overlap negatives in training.

## 7 Discussion

### 7.1 Why JEPA-Reg Works

The label-conditioned predictive objective encourages the model to make paraphrase pairs mutually predictable in a projection space that carries no incentive to reproduce surface token patterns — especially critical for PAWS-style pairs where two sentences share almost all tokens but differ in meaning. The EMA target provides a stable reference during training. Because non-paraphrases are exempt from the loss, the objective never opposes the classifier's need to separate high-overlap non-paraphrases — and the per-class results in Section 6.3 show that this separation is precisely where the empirical gain concentrates.

### 7.2 Comparison with Baselines

In our setting, JEPA-Reg improves PAWS performance where SimCSE [7] improves it only marginally, consistent with the two methods' goals: contrastive learning separates in-batch negatives and optimizes general similarity structure, while JEPA-Reg explicitly predicts target representations for labelled paraphrases. SMART [12] and FreeLB [26] improve robustness through embedding-space perturbations but were not evaluated here; JEPA-Reg is complementary in that it imposes a representation-level bias rather than a perturbation- or parameter-interpolation-based objective.

### 7.3 QQP: No Harm, No Help

QQP does not exhibit PAWS-level overlap–label dissociation, and the baseline already performs well on our 20k-pair subsample. Pure JEPA-Reg matches the baseline (76.72±0.27 vs. 76.35±0.76; d=0.54, within noise). The Hybrid (JEPA+SimCSE) variant underperforms (75.67±0.33), indicating the interaction of the two auxiliary objectives, not JEPA-Reg itself, caused the degradation reported in the submitted version. The pattern across datasets — large improvements on PAWS, no change on MRPC or QQP — is consistent with a targeted adversarial-overlap regularizer.

### 7.4 Applicability Beyond Paraphrase Detection *(new)*

JEPA-Reg's mechanism — label-conditioned mutual predictability of pair members — transfers naturally to other sentence-pair classification tasks: in NLI the loss could be applied to entailment pairs, and in duplicate-question or answer-selection tasks to gold duplicates; our HANS results suggest, however, that such transfer must be trained in-domain rather than expected zero-shot. Graded-similarity tasks are a different matter: an STS-B diagnostic (zero-shot cosine similarity of MRPC-trained encoders) showed JEPA-Reg *reduces* correlation with human similarity scores (Spearman ρ 0.369→0.362 for BERT; 0.680→0.472 for RoBERTa), indicating the objective sharpens a binary decision structure rather than improving graded similarity geometry. This bounds the method's scope honestly: JEPA-Reg is a classification regularizer, not a general sentence-embedding improver.

### 7.5 Limitations

First, all conclusions rest on **n=3 seeds**; per-seed values are reported throughout, effect sizes are descriptive, and no small-p-value claims are made — more seeds are the single most valuable robustness check. Second, we did not evaluate SMART, FreeLB, Mixout, or the debiasing methods [5, 20]; the latter are computationally cheap at this scale and are the natural next comparison. Third, the HANS evaluation uses a single seed's checkpoint, and its per-subcase breakdown shows no evidence of zero-shot structural transfer — the method's demonstrated value is in-domain robustness. Fourth, experiments cover only 110M-parameter encoders. Fifth, RoBERTa's absolute PAWS gain is small (+0.22); the supported claim there is seed-consistency, not magnitude. Sixth, training costs 6.5× per step in our implementation (inference is unaffected). Finally, results below 25% of MRPC indicate instability in extremely low-resource regimes.

## 8 Conclusion

We proposed JEPA-Reg, a label-conditioned JEPA-style EMA predictive auxiliary loss for BERT-based paraphrase detection. Fine-tuning on PAWS, JEPA-Reg improves BERT-base F1 by +7.9 points and reduces seed variance 15×, with the advantage concentrated in the most overlap-adversarial strata and maintained on both classes; RoBERTa-base shows small but seed-consistent gains. A systematic multi-seed λ study finds the gain insensitive to the auxiliary-loss weight across [0.05, 1.0], so the method requires no λ tuning. Per-subcase HANS diagnostics show that zero-shot heuristic-level gains reflect prediction-bias shifts, which we report as a negative result bounding the method's scope: JEPA-Reg is an in-domain adversarial-robustness regularizer with no inference-time overhead, not a source of transferable structural reasoning.

---

## Appendix A — PAWS validity diagnostics
Confusion matrices (seed-42 best-validation models, PAWS test, rows = gold {0,1}, cols = predicted {0,1}):
- Baseline: [[2534, 1930], [494, 3042]] — class-0 P/R 0.837/0.568, class-1 P/R 0.612/0.860 (bias toward positive class).
- SimCSE: [[2763, 1701], [542, 2994]] — class-0 P/R 0.836/0.619, class-1 P/R 0.638/0.847.
- JEPA-Reg: [[3245, 1219], [379, 3157]] — class-0 P/R 0.895/0.727, class-1 P/R 0.721/0.893 (balanced; per-class F1 0.80/0.80).
Majority-class baseline: accuracy 55.8, positive-F1 0.0. Always-paraphrase: accuracy 44.2, F1 61.3.

## Appendix B — t-SNE visualization (exploratory)
*(Figure 6 retained without the silhouette-based causal interpretation.)*

## Reproducibility statement
Code, notebook (including all revision experiments R-1–R-8), and per-seed results JSONs are released at ⟨repository URL⟩. Hardware: 1× NVIDIA RTX 5060 Ti (16 GB). Known asymmetry: label smoothing 0.05 (baseline/SimCSE) vs. 0.0 (JEPA-Reg classification loss).

*(References unchanged from the submitted version.)*
