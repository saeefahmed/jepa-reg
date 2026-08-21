# JEPA-Reg: Label-Conditioned Joint-Embedding Predictive Regularization for Paraphrase Detection

Code, analysis notebook, and per-seed result files for the paper
*"JEPA-Reg: … Adversarial Lexical Overlap Robustness"* (ICCA 2026 submission ID #284).

Everything reported in the paper can be regenerated from this repository.
**No dataset files are included** — MRPC, QQP, PAWS and STS-B are downloaded from the
HuggingFace Hub at runtime, and HANS is fetched from its official GitHub release.

---

## 1. Environment

All reported numbers were produced with **Python 3.14.6** on **Windows 11**, on
**1× NVIDIA RTX 5060 Ti (16 GB)**.

```bash
python -m venv jepa-env
jepa-env\Scripts\activate          # Windows;  source jepa-env/bin/activate on Linux

# PyTorch first, from the CUDA 13.0 index (required for Blackwell / sm_120 GPUs)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Key versions: `torch 2.13.0+cu130`, `transformers 5.14.1`, `datasets 5.0.1`,
`numpy 2.5.1`, `scipy 1.18.0`, `scikit-learn 1.9.0`.
`requirements-lock.txt` is a full `pip freeze` of the exact environment used.

> **Note on GPU builds.** A CUDA 12.6 PyTorch build will import successfully but has no
> kernels for sm_120 and will fail at runtime on an RTX 50-series card. Install the
> `cu130` (or later) wheel.

---

## 2. Repository layout

The layout is intentionally **flat**: each stage script calls
`os.chdir(os.path.dirname(__file__))` and then `exec(open("_setup_extracted.py").read())`,
so moving files into subpackages will break the pipeline.

| Path | What it is |
|---|---|
| `_setup_extracted.py` | **Core implementation.** Models (`JepaBertPair`, `SimCSEBertPair`, `HybridJepaSimCSE`), training/eval loops, statistics, dataset loaders, all hyperparameter constants. Executed by every stage script. |
| `_stage1_train_bert.py` | Stage 1 — full BERT-base run: MRPC + low-resource sweep + QQP + PAWS, 3 seeds. Crash-safe via per-dataset checkpointing. |
| `_stage2_evals.py` | Stage 2 — HANS zero-shot transfer, per-subcase breakdown, overlap-quartile stratification, overhead measurement. |
| `_stage3_ablations.py` | Stage 3 — mask ablation (label-conditioned vs. all-pairs) and the full λ sweep. |
| `_stage4_warmup_control.py` | Stage 4 — λ=1.0 **without** λ warmup control (PAWS, 3 seeds). |
| `_smoke_test.py` | Fast sanity check that the environment and model code work. |
| `JEPA_final_optionB_fixed.ipynb` | Analysis notebook — the original end-to-end development record, including all revision experiments. |
| `_stage*_log.txt` | Full stdout logs of the runs that produced the reported numbers. |
| `results/` | Per-seed result JSONs (see §4). |
| `camera_ready/` | LaTeX source (`main.tex`, `references.bib`, `acmart.cls`), final figures, and the figure-regeneration script. |

---

## 3. Reproducing the results

Run the stages in order from the repository root:

```bash
python _smoke_test.py                # optional: ~1 min sanity check
python _stage1_train_bert.py         # ~2.5 h on one RTX 5060 Ti
python _stage2_evals.py
python _stage3_ablations.py
python _stage4_warmup_control.py
```

Stage 1 writes `results/checkpoint_bert.json` and resumes from
`results/checkpoint_bert-base-uncased_partial.json` if interrupted. Stages 2–4 read
`results/checkpoint_bert.json`, so Stage 1 must complete first.

Figures 3–6 are regenerated from the result JSONs alone (no GPU needed):

```bash
python camera_ready/_make_fresh_figs.py
python _draw_fig5_v2.py
```

**Fixed seeds** are `[42, 43, 44]` (`SEEDS` in `_setup_extracted.py`). Exact
bitwise reproduction additionally requires the same GPU and CUDA build; per-seed F1
values are expected to vary in the last decimal on different hardware.

---

## 4. Result files → paper tables

| File | Used for |
|---|---|
| `results/checkpoint_bert.json` | BERT-base per-seed metrics **and per-example predictions** (Tables 4, 6; Appendix A confusion matrices) |
| `results/all_results_fresh_v2.json` | Condensed BERT-base + HANS summary; input to Figures 3–4 |
| `all_results_final.json` | Original full run: BERT-base, RoBERTa-base, HANS, STS-B diagnostic (Table 5) |
| `results/hans_subcase.json` | HANS per-heuristic × gold-label accuracies (Table 6) |
| `results/lambda_ablation_full.json` | λ sweep, mean ± std over 3 seeds (Table 7) |
| `results/lambda1_nowarmup_control.json` | λ=1.0 no-warmup control (Section 6.7) |
| `results/mask_ablation.json` | Label-conditioned vs. all-pairs + collapse check (Section 6.6) |
| `results/qqp_pure_jepa.json` | Pure JEPA-Reg on QQP (Section 7.3) |
| `results/overhead.json` | Training-time and memory overhead (Section 3.5) |

Because per-example predictions are included, **every statistic and figure in the paper
can be recomputed without a GPU** from the JSONs in this repository.

---

## 5. What is deliberately not included

* **Trained model checkpoints** (`results/models/*.pt`, ≈5 GB). These are excluded via
  `.gitignore`. They are not needed to reproduce any reported number — the per-seed
  metrics and per-example predictions in `results/` are sufficient. Rerunning Stage 1
  regenerates them.
* **Dataset files.** All datasets are public and fetched automatically at runtime.

---

## 6. Known limitations

* **Label-smoothing asymmetry.** Baseline and SimCSE use label smoothing 0.05
  (`LABEL_SMOOTHING`); the JEPA-Reg classification loss uses 0.0
  (`JEPA_LABEL_SMOOTHING`). This is an acknowledged confound; **no matched
  label-smoothing control was run.**
* **n = 3 seeds.** The exact paired sign-flip permutation test cannot yield p < 0.25 at
  n = 3 regardless of effect magnitude. Per-seed values and effect sizes are reported
  descriptively; see Section 4.2.
* **Notebook-coupled ablations.** `_stage3_ablations.py` executes notebook cells *by
  index* (cells 49 and 51 of `JEPA_final_optionB_fixed.ipynb`). Editing or reordering the
  notebook will break Stage 3.
* **RoBERTa-base** numbers come from the original run of the same code, not from the
  BERT-base rerun; per-seed values for both are in the released JSONs.

---

## 7. Citation

```bibtex
@inproceedings{jepareg2026,
  title     = {JEPA-Reg: Label-Conditioned Joint-Embedding Predictive Regularization
               for Adversarial Lexical Overlap Robustness},
  author    = {Ahmed, Saeef},
  booktitle = {Proceedings of ICCA 2026},
  year      = {2026}
}
```

## 8. License

Released under the MIT License — see `LICENSE`.
