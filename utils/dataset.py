import torch
from torch.utils.data import Dataset
import numpy as np


# =========================
# DATASET CLASS
# =========================
class JetDataset(Dataset):
    def __init__(self, npz_path, indices=None):
        data = np.load(npz_path)

        self.images    = data["images"]
        self.labels    = data["labels"]
        self.jet_pt    = data["jet_pt"]
        self.event_id  = data["event_id"]
        self.sample_id = data["sample_id"]   # 0=tau, 1=jj, 2=bb

        if indices is not None:
            self.images    = self.images[indices]
            self.labels    = self.labels[indices]
            self.jet_pt    = self.jet_pt[indices]
            self.event_id  = self.event_id[indices]
            self.sample_id = self.sample_id[indices]

        # Convert everything to tensors upfront for consistency
        self.images    = torch.tensor(self.images,    dtype=torch.float32)
        self.labels    = torch.tensor(self.labels,    dtype=torch.float32)
        self.jet_pt    = torch.tensor(self.jet_pt,    dtype=torch.float32)
        self.event_id  = torch.tensor(self.event_id,  dtype=torch.int32)
        self.sample_id = torch.tensor(self.sample_id, dtype=torch.int8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image":     self.images[idx],
            "label":     self.labels[idx],
            "jet_pt":    self.jet_pt[idx],
            "event_id":  self.event_id[idx],
            "sample_id": self.sample_id[idx]
        }


# =========================
# EVENT-WISE SPLITTING
# =========================
def event_wise_split(event_ids, val_fraction=0.2, seed=42):
    """
    Splits unique event IDs into train/val sets.
    event_ids are globally unique across samples because of the
    event_offset applied during dataset building.
    """
    rng = np.random.default_rng(seed)

    unique_events = np.unique(event_ids)
    rng.shuffle(unique_events)

    n_val        = int(len(unique_events) * val_fraction)
    val_events   = unique_events[:n_val]
    train_events = unique_events[n_val:]

    return train_events, val_events


# =========================
# EVENT -> INDEX MAPPING
# =========================
def get_indices_from_events(event_ids, selected_events):
    mask = np.isin(event_ids, selected_events)
    return np.where(mask)[0]
