# Datasets

This folder holds the compressed jet image datasets (`.npz`) used for
training and testing. These files are **not stored in the repository** due
to their size but can be reproduced using the scripts in `../utils/` and
`../notebooks/02_jet_images_dataset.ipynb`.

---

## Folder structure

```
datasets/
├── cnn_vit/
│   ├── jet_images_125GeV_train.npz   ← training set, 125 GeV model
│   └── jet_images_250GeV_train.npz   ← training set, 250 GeV model
└── test/
    ├── jet_images_100GeV_test.npz
    ├── jet_images_125GeV_test.npz
    ├── jet_images_150GeV_test.npz
    ├── jet_images_200GeV_test.npz
    ├── jet_images_250GeV_test.npz
    └── jet_images_300GeV_test.npz
```

---

## Array layout (per `.npz` file)

| Array | Shape | dtype | Description |
|---|---|---|---|
| `images` | (N, 3, 32, 32) | float32 | 3-channel jet images |
| `labels` | (N,) | int8 | 1 = tau (signal), 0 = background |
| `sample_id` | (N,) | int8 | 0 = tau, 1 = jj, 2 = bb |
| `jet_pt` | (N,) | float32 | jet transverse momentum (GeV) |
| `event_id` | (N,) | int32 | globally unique event ID across all samples |

The 3 image channels correspond to: **EFlowTrack**, **EFlowPhoton**,
**EFlowNeutralHadron** — each a 32×32 grid in the η-φ plane (±1.0 range).

---

## Jet counts (125 GeV training set)

| Sample | Jets after selection | Label | sample_id |
|---|---|---|---|
| τ⁺τ⁻ | 178,451 | 1 | 0 |
| jj | 147,961 | 0 | 1 |
| bb̄ | 115,478 | 0 | 2 |
| **Total** | **441,890** | | |

Signal-to-background ratio is approximately 1:1.5 for both the 125 GeV and 250 GeV training datasets.

---

## Preprocessing applied

1. η-φ centering (±1.0 range)
2. PCA rotation — principal axis aligned with η
3. Energy flip — applied after PCA rotation
4. L2 normalisation per image

pT selection window: **15–60 GeV** (125 GeV training), **15–125 GeV** (250 GeV training).
Cone matching: dR < 0.4 between Delphes jet and generator-level tau.
