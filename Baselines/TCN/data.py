from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzSequenceDataset(Dataset):
    """Loads (X, y) arrays and yields tensors.

    Supported X shapes:
      - (N, L) integer token ids
      - (N, L, C) float features

    y can be:
      - (N,) int class labels
      - (N,) float labels for BCE
    """

    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):
        self.x = x
        self.y = y

        if self.x.ndim not in (2, 3):
            raise ValueError(f"X must be 2D or 3D, got shape {self.x.shape}")

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        x = self.x[idx]
        if np.issubdtype(x.dtype, np.integer):
            x_t = torch.as_tensor(x, dtype=torch.long)
        else:
            x_t = torch.as_tensor(x, dtype=torch.float32)

        if self.y is None:
            return x_t

        y = self.y[idx]
        if np.issubdtype(self.y.dtype, np.integer):
            y_t = torch.as_tensor(y, dtype=torch.long)
        else:
            y_t = torch.as_tensor(y, dtype=torch.float32)
        return x_t, y_t


def load_npz_splits(npz_path: str):
    npz = np.load(npz_path, allow_pickle=False)

    def _get(key):
        return npz[key] if key in npz.files else None

    splits = {
        "train": (_get("X_train"), _get("y_train")),
        "val": (_get("X_val"), _get("y_val")),
        "test": (_get("X_test"), _get("y_test")),
    }

    if splits["train"][0] is None:
        raise ValueError("npz must contain X_train (and typically y_train)")

    return splits
