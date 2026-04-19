# Datasets

This folder holds the compressed datasets (`.npz`) used for training and testing —
both jet image arrays (for CNN/ViT) and tabular feature arrays (for BDT).
These files are **not stored in the repository** due to their size but can be
reproduced using the scripts in `../notebooks/`.

---

## Folder structure

```
datasets/
├── cnn_vit/
│   ├── jet_images_125GeV_train.npz   ← training set, 125 GeV image models
│   └── jet_images_250GeV_train.npz   ← training set, 250 GeV image models
├── bdt/
│   ├── bdt_features_125GeV_train.npz ← training set, 125 GeV BDT models
│   ├── bdt_features_250GeV_train.npz ← training set, 250 GeV BDT models
│   ├── bdt_features_100GeV_test.npz
│   ├── bdt_features_125GeV_test.npz
│   ├── bdt_features_150GeV_test.npz
│   ├── bdt_features_200GeV_test.npz
│   ├── bdt_features_250GeV_test.npz
│   └── bdt_features_300GeV_test.npz
└── test/
    ├── jet_images_100GeV_test.npz
    ├── jet_images_125GeV_test.npz
    ├── jet_images_150GeV_test.npz
    ├── jet_images_200GeV_test.npz
    ├── jet_images_250GeV_test.npz
    └── jet_images_300GeV_test.npz
```

---

## Jet image arrays (`cnn_vit/` and `test/`)

### Array layout (per `.npz` file)

| Array | Shape | dtype | Description |
|---|---|---|---|
| `images` | (N, 3, 32, 32) | float32 | 3-channel jet images |
| `labels` | (N,) | int8 | 1 = tau (signal), 0 = background |
| `sample_id` | (N,) | int8 | 0 = tau, 1 = jj, 2 = bb |
| `jet_pt` | (N,) | float32 | jet transverse momentum (GeV) |
| `event_id` | (N,) | int32 | globally unique event ID across all samples |

The 3 image channels correspond to: **EFlowTrack**, **EFlowPhoton**,
**EFlowNeutralHadron** — each a 32×32 grid in the η-φ plane (±1.0 range).

### Preprocessing applied

1. η-φ centering (±1.0 range)
2. PCA rotation — principal axis aligned with η
3. Energy flip — applied after PCA rotation
4. L2 normalisation per image

---

## Tabular feature arrays (`bdt/`)

### Array layout (per `.npz` file)

| Array | Shape | dtype | Description |
|---|---|---|---|
| `features` | (N, 10) | float32 | 10 jet-level features (see below) |
| `labels` | (N,) | int32 | 1 = tau (signal), 0 = background |
| `sample_id` | (N,) | int32 | 0 = tau, 1 = jj, 2 = bb |
| `jet_pt` | (N,) | float32 | jet transverse momentum (GeV) |
| `event_id` | (N,) | int64 | per-process event index (use with sample_id) |
| `feature_names` | (10,) | str | feature name strings |

### Feature order (columns 0–9)

| Index | Name | Description |
|---|---|---|
| 0 | `pt` | Jet transverse momentum [GeV] |
| 1 | `mass` | Jet invariant mass [GeV] |
| 2 | `ncharged` | Number of charged constituents |
| 3 | `nneutrals` | Number of neutral constituents |
| 4 | `ehad` | EhadOverEem — hadronic-to-EM energy ratio |
| 5 | `chf` | Charged energy fraction = NCharged / (NCharged + NNeutrals) |
| 6 | `nef` | Neutral energy fraction = 1 − CHF |
| 7 | `tau1` | N-subjettiness τ₁ |
| 8 | `tau21` | N-subjettiness ratio τ₂/τ₁ |
| 9 | `tau32` | N-subjettiness ratio τ₃/τ₂ |

`btag` and `tautag` flags are intentionally excluded. `TauTag` is the Delphes built-in
tau ID — including it would make the classifier partially circular. `BTag` has no
physics motivation as a tau discriminant.

### Jet counts — BDT training sets

| Sample | 125 GeV jets | 250 GeV jets |
|---|---|---|
| τ⁺τ⁻ (signal) | 178,451 | 212,613 |
| jj (background) | 147,961 | 184,353 |
| bb̄ (background) | 115,478 | 138,496 |
| **Total** | **441,890** | **535,462** |

Signal:background ratio ≈ 1:1.48 (125 GeV), 1:1.52 (250 GeV).
pT windows: 15–60 GeV (125 GeV), 15–125 GeV (250 GeV).

### Note on `event_id`

The `event_id` field is assigned per-process during extraction (i.e., within tau,
jj, and bb separately). To compute per-event quantities correctly, always combine
`event_id` with `sample_id` as a composite key. For per-event tau tagging efficiency,
filter to `sample_id == 0` first.

---

## Jet counts — image training sets (125 GeV)

| Sample | Jets after selection | Label | sample_id |
|---|---|---|---|
| τ⁺τ⁻ | 178,451 | 1 | 0 |
| jj | 147,961 | 0 | 1 |
| bb̄ | 115,478 | 0 | 2 |
| **Total** | **441,890** | | |

pT window: 15–60 GeV · Cone matching: ΔR < 0.4

## Jet counts — image training sets (250 GeV)

| Sample | Jets after selection | Label | sample_id |
|---|---|---|---|
| τ⁺τ⁻ | 212,613 | 1 | 0 |
| jj | 184,353 | 0 | 1 |
| bb̄ | 138,496 | 0 | 2 |
| **Total** | **535,462** | | |

pT window: 15–125 GeV · Cone matching: ΔR < 0.4
