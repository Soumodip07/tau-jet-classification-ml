# Utils

Core shared modules used across all training and evaluation scripts.

---

## `dataset.py`

Handles dataset loading, splitting, and PyTorch integration.

### `JetDataset`

A PyTorch `Dataset` wrapping the `.npz` jet image files.

```python
JetDataset(npz_path, indices=None)
```

| Argument | Type | Description |
|---|---|---|
| `npz_path` | str | path to the `.npz` file |
| `indices` | array-like or None | subset of jet indices to use; loads all jets if None |

Each `__getitem__` call returns a tuple:
`(image, label, sample_id, jet_pt, event_id)`

- `image` — float32 tensor of shape (3, 32, 32)
- `label` — float32 scalar (1.0 = tau, 0.0 = background)
- `sample_id` — int8 scalar (0 = tau, 1 = jj, 2 = bb)
- `jet_pt` — float32 scalar
- `event_id` — int32 scalar

### `event_wise_split`

Splits jet indices into train and validation sets ensuring **no jet from
the same event appears in both sets** (no leakage).

```python
train_idx, val_idx = event_wise_split(event_ids, val_frac=0.2, seed=42)
```

Internally groups unique event IDs, shuffles them, splits at `val_frac`,
then maps back to jet indices.

### `get_indices_from_events`

Helper that converts a set of event IDs to the corresponding jet indices
in the dataset array.

```python
indices = get_indices_from_events(event_ids_array, target_event_id_set)
```

---

## `modelarch.py`

Defines the two model architectures used in this project.

### `JetCNN`

A 4-block convolutional network for jet image classification.

```
Input: (B, 3, 32, 32)

Conv(3→32, 3×3) → BN → ReLU → MaxPool(2)     → (B, 32, 16, 16)
Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)    → (B, 64, 8, 8)
Conv(64→128, 3×3) → BN → ReLU → MaxPool(2)   → (B, 128, 4, 4)
Conv(128→128, 3×3) → BN → ReLU               → (B, 128, 4, 4)

Flatten → Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→1)

Output: (B, 1)  [raw logit — pass through sigmoid for probability]
```

Total parameters: **766,337**

### `JetViT`

A Vision Transformer adapted for 32×32 3-channel jet images.

```
Patch embedding: patch_size=4 → 64 spatial patches + 1 CLS token = 65 tokens
Embedding dim:   128
Transformer:     depth=4, num_heads=4, mlp_dim=256, dropout=0.1
Head:            CLS token → Linear(128→1)

Output: (B, 1)  [raw logit]
```

Total parameters: **545,281**

---

## Training configuration (both models)

| Setting | CNN | ViT |
|---|---|---|
| Optimiser | Adam | AdamW |
| Learning rate | 3e-4 | 3e-4 |
| Weight decay | — | 1e-4 |
| Scheduler | ReduceLROnPlateau (AUC, factor=0.5, patience=5) | same |
| Early stopping | patience=10, monitors val AUC | same |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 |
| Batch size | 256 | 256 |
| Loss | BCEWithLogitsLoss with pos_weight | same |
| pos_weight | 1.4811 (n_bkg / n_sig) | same |

Checkpoints save only: `epoch` (int), `model_state_dict`, `val_loss` (float),
`val_auc` (float) — this format is portable across numpy versions (no numpy
arrays in the checkpoint).

Load with:
```python
checkpoint = torch.load("model.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
```
