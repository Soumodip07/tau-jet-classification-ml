# Notebooks

All analysis and training notebooks for the project. Run them in the
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

## Planned (coming soon)

| Folder | Contents |
|---|---|
| `bdt/` | BDT baseline using 12 tabular jet features |
| `gnn/` | Graph Neural Network using particle-level inputs |
| `fusion/` | Late-fusion ensemble combining all models |
