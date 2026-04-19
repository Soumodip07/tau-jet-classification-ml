# Notebooks

All analysis and training notebooks/scripts for the project. Run them in the
numbered order for a clean end-to-end workflow.

---

## Root-level notebooks (run first)

### `01_kinematics_analysis.ipynb`
Reads the ROOT files produced by Delphes and computes jet-level kinematic
variables for each process and energy. Produces:
- pT, η, mass, N-subjettiness distributions per sample
- Signal vs background separation power (Wasserstein distance ranking)
- pT CDF plots used to determine the pT ceiling per energy
- Correlation matrices and prong structure visualisations

Outputs saved to `results_analysis/kinematics/`.

### `02_jet_images_dataset.ipynb`
Constructs the jet image datasets from ROOT files. Handles:
- Jet selection (pT window, dR < 0.4 cone matching)
- 3-channel image construction (EFlowTrack / EFlowPhoton / EFlowNeutralHadron)
- Preprocessing pipeline (centering, PCA rotation, energy flip, L2 normalisation)
- Event-wise train/val split (seed=42, no jet leakage)
- Saves `.npz` files to `datasets/cnn_vit/` and `datasets/test/`

### `03_Comparison_CNN_Vs_ViT.ipynb`
Post-evaluation comparison notebook. Loads the summary CSVs from both model
families and produces:
- AUC vs energy overlays (CNN@125, CNN@250, ViT@125, ViT@250)
- F1, recall, precision, event-tagging efficiency vs energy
- Cross-model ROC overlays at selected energies

---

## CNN notebooks (`CNN/`)

### `CNN.ipynb`
Reference notebook for the CNN architecture. Documents the `JetCNN` model
definition, layer shapes, parameter count, and a forward-pass sanity check.

### `CNN_models_training.ipynb`
Full training loop for both CNN models (125 GeV and 250 GeV). Includes:
- Dataset loading and class weight calculation
- Adam optimiser, `ReduceLROnPlateau` scheduler, AUC-based early stopping
- Training curves (loss + AUC per epoch)
- Checkpoint saving in numpy-portable format (no numpy version dependency)

---

## ViT notebooks (`ViT/`)

### `ViT.ipynb`
Reference notebook for the Vision Transformer architecture. Documents the
`JetViT` model (patch embedding, transformer blocks, CLS token head),
patch visualisations, and parameter count.

### `ViT_models_training.ipynb`
Full training loop for both ViT models (125 GeV and 250 GeV). Same structure
as the CNN training notebook but uses AdamW optimiser with weight decay.
ViT loss curves are noticeably smoother than CNN due to weight decay
preventing score overconfidence.

---

## BDT scripts (`BDT/`)

Five scripts forming the complete BDT pipeline. Scripts 01–04 run **locally**
(require ROOT files or saved `.npz` datasets); scripts 05–06 run locally after
all models are trained.

### `01_BDT_dataset_creation.py`
Extracts 10 tabular features from Delphes ROOT files for all training and test
energies. Run once — outputs 8 `.npz` files to `datasets/bdt/`:
- 2 training sets: `bdt_features_125GeV_train.npz`, `bdt_features_250GeV_train.npz`
- 6 test sets: `bdt_features_{100,125,150,200,250,300}GeV_test.npz`

Features extracted: `pt`, `mass`, `ncharged`, `nneutrals`, `ehad`, `chf`, `nef`,
`tau1`, `tau21`, `tau32`. `btag` and `tautag` excluded by design.

### `02_BDT_train_RF.py`
Trains a Random Forest classifier. Set `TRAIN_ENERGY = 125` or `250` and run
once per energy. Uses `class_weight="balanced"`, 600 trees, `sqrt` features.
Training time: ~170s (125 GeV), ~220s (250 GeV). Saves `.pkl` model + scaler.

### `03_BDT_train_XGB.py`
Trains an XGBoost classifier with early stopping. Set `TRAIN_ENERGY` and run
once per energy. Uses `lr=0.02`, `max_depth=4`, `n_estimators=2500` with
`early_stopping_rounds=40`. Training time: ~55s (125 GeV), ~70s (250 GeV).
Saves `.json` model + scaler.

### `04_BDT_train_LGBM.py`
Trains a LightGBM classifier with early stopping. Same hyperparameter
structure as XGBoost. Significantly faster: ~12s (125 GeV), ~18s (250 GeV).
Best overall performance. Saves `.txt` booster + scaler.

### `05_BDT_evaluation.py`
Loads all 6 trained models and evaluates each on all 6 test energies.
Auto-detects which models are available so it can run on a partial set.

**Physics plots per model:**
- Background rejection curve (1/FPR vs signal efficiency, log-y scale)
- Background rejection at 4 working points (30/50/70/90% sig eff) vs √s
- pT-binned τ tagging efficiency + background fake rate (all 6 energies)
- Per-event vs jet-level τ tagging efficiency vs √s
- Per-process rejection curves (τ vs jj, τ vs bb separately)
- AUC (overall, τ vs jj, τ vs bb) vs test energy
- Feature importance (normalised gain)

**Metrics table** (`.txt`) per model:
`Energy | AUC | AUC_jj | AUC_bb | Acc | Prec | Rec | F1 | Jet-eff | Evt-eff | BRej@30/50/70/90%`

`Jet-eff` = fraction of tau jets in the pT window correctly tagged (= recall).
`Evt-eff` = fraction of tau *events* with ≥1 jet correctly tagged — the true
event-selection efficiency relevant for H→ττ physics analyses.

### `06_BDT_results_compare.py`
Cross-algorithm comparison plots for both training groups (125 GeV and 250 GeV).
Requires all 6 models to have been evaluated by `05_BDT_evaluation.py`.

**Comparison plots produced:**
- Background rejection curves — in-distribution test energy (all 3 algorithms overlaid)
- Background rejection curves — all 6 test energies (2×3 grid)
- AUC vs energy — all 3 algorithms per group
- F1 vs energy — all 3 algorithms per group
- Background rejection at 4 WPs vs energy — 2×2 subplot grid
- Feature importance comparison — side-by-side bars, all 3 algorithms
- All-model AUC summary plot (6 models, 2 panels: AUC + F1 vs energy)
- All-model AUC summary table (`.txt`)

Outputs saved to `results_analysis/bdt/comparison/`.

---

## Planned (coming soon)

| Folder | Contents |
|---|---|
| `GNN/` | Graph Neural Network using particle-level inputs |
| `fusion/` | Late-fusion ensemble combining all models |
