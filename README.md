# Causal-XAI-Sepsis — Reproducibility Repository

Supporting code for manuscript Section 3.8.1 ("Reproducibility and Code
Availability"). This repository reproduces every reported experiment:
preprocessing (Section 3.7), the BTAE + CNA model (Sections 3.2, 3.4), TSCM
construction and validation (Sections 3.3, 3.3.1), the counterfactual
inference engine (Section 3.5), the semi-synthetic ITE benchmark
(Section 3.7.1), and the training procedure (Section 3.8).

## Repository layout

```
causal_xai_reproducibility/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   └── default_config.yaml        # every hyperparameter reported in the manuscript
└── src/
    ├── seed_utils.py               # deterministic seeding (Sec 3.8.1)
    ├── data_splits.py              # patient-level 70/15/15 temporal-holdout split
    ├── preprocessing.py            # decay imputation + time-delta encoding (Sec 3.7)
    ├── model_btae.py                # Bidirectional Temporal Attention Encoder (Sec 3.2)
    ├── model_cna.py                 # Causal Node Activation (Sec 3.4)
    ├── tscm_construction.py         # PC-Stable + stability + cross-algorithm validation (Sec 3.3/3.3.1)
    ├── counterfactual_engine.py     # abduction-action-prediction do-calculus (Sec 3.5)
    ├── ite_benchmark.py             # semi-synthetic PEHE/ATE-bias/policy-risk benchmark (Sec 3.7.1)
    └── train.py                     # training loop, optimizer/schedule/loss (Sec 3.8)
```

## Quick start

```bash
conda env create -f environment.yml
conda activate causal-xai-sepsis

# 1. Generate the canonical patient-level split (requires local MIMIC-IV v2.2)
python src/data_splits.py --mimic-iv-root /path/to/mimiciv --out splits/mimic_iv_subject_split.json

# 2. Train (one call per seed; manuscript reports mean +/- std over 5 seeds)
for seed in 0 1 2 3 4; do
  python src/train.py --config configs/default_config.yaml --seed $seed \
      --mimic-iv-root /path/to/mimiciv --split splits/mimic_iv_subject_split.json
done

# 3. TSCM construction + validation report
python -c "from src.tscm_construction import build_and_validate_tscm; ..."

# 4. Semi-synthetic ITE benchmark (Table X in the manuscript)
python src/ite_benchmark.py
```

## Known non-determinism

CUDA scatter-add operations inside BTAE's attention aggregation do not have
a fully deterministic kernel in PyTorch 2.1.0. Residual run-to-run variance
on GPU is on the order of 1e-4 AUROC — within the 5-seed mean ± std
reporting convention used throughout the manuscript, but noted here so it
is never mistaken for a bug.

## Data use

No MIMIC-IV or eICU-CRD data is redistributed in this repository. Running
the pipeline requires independent PhysioNet credentialing and completion
of the standard CITI training / Data Use Agreement for both datasets. Only
de-identified `subject_id` partition lists are released (`data_splits.py`
output), which are meaningless without an independent, credentialed
MIMIC-IV extract.

---

## Supplementary Table: Reproducibility Checklist

| Item | Status / Location |
|---|---|
| **Code available** | Yes — this repository, `[public URL]` |
| **License** | MIT |
| **Trained model checkpoints released** | Yes — `checkpoints/best_seed{0-4}.pt`, `[URL]` |
| **Data source and access procedure documented** | Yes — Data Availability Statement; MIMIC-IV v2.2 / eICU-CRD v2.0 via PhysioNet credentialing |
| **Exact patient-level train/val/test split released** | Yes — de-identified `subject_id` lists, `splits/mimic_iv_subject_split.json` |
| **Data leakage safeguard** | Yes — preprocessing statistics fit on train split only (`preprocessing.py: fit_preprocessing_stats`), enforced by construction |
| **Random seeds fixed and reported** | Yes — seeds {0,1,2,3,4}, `seed_utils.set_all_seeds` |
| **Results reported as mean ± std over multiple seeds** | Yes — 5 seeds (Section 3.8) |
| **Deterministic execution documented, including known exceptions** | Yes — `torch.use_deterministic_algorithms`; CUDA scatter-add exception documented in README |
| **Full hyperparameter values (not just search grid) reported** | Yes — `configs/default_config.yaml` |
| **Hyperparameter selection protocol documented** | Yes — Section 3.8.1; grid search on validation split only, curves in Supplementary Table S1 |
| **Software environment pinned (exact versions)** | Yes — `requirements.txt` / `environment.yml` |
| **Hardware specification reported** | Yes — NVIDIA A100 80GB, Section 3.8 |
| **Training / inference compute cost reported** | Yes — 4.2h training, 287ms/patient counterfactual inference (Section 3.8) |
| **Evaluation metrics implementation released** | Yes — `ite_benchmark.py` (PEHE, ATE bias, policy risk); AUPRC/AUROC via scikit-learn standard implementations |
| **Causal structure learning algorithm and parameters specified** | Yes — PC-Stable, G² test, α=0.01 (Section 3.3) |
| **Structure learning stability/robustness reported** | Yes — bootstrap edge-stability filtering + GES/LiNGAM cross-check (Section 3.3.1) |
| **External validation dataset and protocol specified** | Yes — eICU-CRD, no retraining (Section 3.7) |
| **Statistical test methodology for reported comparisons specified** | Yes — paired bootstrap (B=10,000), Section 4.8; linear-weighted Cohen's κ, Section 3.9 |
| **Third-party library versions for baselines pinned** | Partial — TARNet/Dragonnet/causal forest wrapper versions to be added to `requirements.txt` `[list exact packages used]` |
| **Ethics/IRB and data use compliance statement** | Yes — PhysioNet credentialing and DUA referenced in Data Availability Statement |
