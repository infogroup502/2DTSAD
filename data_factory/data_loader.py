import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import torch

def gaf_diff_transform(window_np):
    W, C, L = window_np.shape
    out = np.zeros((W, C, L, L), dtype=np.float32)
    eps = 1e-8

    for w in range(W):
        for c in range(C):
            seq = window_np[w, c].astype(np.float64)

            mn = seq.min()
            mx = seq.max()
            if abs(mx - mn) < eps:
                scaled = np.zeros_like(seq)
            else:
                scaled = 2 * (seq - mn) / (mx - mn) - 1
                scaled = np.clip(scaled, -1.0, 1.0)

            # φ = arccos(x)
            phi = np.arccos(scaled)

            # --- GADF = sin(φ_i - φ_j) ---
            diff = phi[:, None] - phi[None, :]
            gadf = np.abs(np.sin(diff))  # 加绝对值

            # --- GASF = cos(φ_i + φ_j) ---
            summ = phi[:, None] + phi[None, :]
            gasf = np.abs(np.cos(summ))  # 加绝对值

            fused = np.zeros((L, L), dtype=np.float32)

            #  |GADF|
            fused[np.triu_indices(L, k=1)] = gadf[np.triu_indices(L, k=1)]

            #  |GASF|
            fused[np.tril_indices(L, k=-1)] = gasf[np.tril_indices(L, k=-1)]

            # 0.5*(|GADF| + |GASF|)
            diag_vals = 0.5 * (gadf[np.diag_indices(L)] + gasf[np.diag_indices(L)])
            fused[np.diag_indices(L)] = diag_vals

            out[w, c] = fused.astype(np.float32)

    return out

class NeighborSegLoader(Dataset):
    def __init__(self, data_path, prefix, win_size, local_size, global_size, global_stride=None, mode='train'):
        super().__init__()
        self.prefix = prefix
        self.win_size = int(win_size)
        try:
            self.local_size = int(local_size[0])
        except Exception:
            self.local_size = int(local_size)
        try:
            self.global_size = int(global_size[0])
        except Exception:
            self.global_size = int(global_size)
        self.global_stride = int(global_stride) if global_stride is not None else int(self.local_size)
        self.mode = mode

        # load files (train/test)
        train_file = os.path.join(data_path, f"{prefix}_train.npy")
        test_file  = os.path.join(data_path, f"{prefix}_test.npy")
        label_file = os.path.join(data_path, f"{prefix}_test_label.npy")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"{train_file} not found")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"{test_file} not found")

        train = np.load(train_file)
        test = np.load(test_file)
        train = np.nan_to_num(train).astype(np.float32)
        test = np.nan_to_num(test).astype(np.float32)

        # fit scaler on train, apply to train/test
        self.scaler = StandardScaler()
        self.scaler.fit(train)
        self.train = self.scaler.transform(train)
        self.test  = self.scaler.transform(test)

        # load labels if exist (for evaluation)
        if os.path.exists(label_file):
            self.test_labels = np.load(label_file)
        else:
            self.test_labels = None

        # choose split
        if self.mode == 'train':
            self.data = self.train
            self.labels = None
        else:
            self.data = self.test
            self.labels = self.test_labels

        # shape check: data expected (T, C)
        if self.data.ndim != 2:
            if self.data.ndim == 1:
                self.data = self.data[:, None]
            else:
                raise ValueError("data should be 2D array of shape (T, C)")

        self.T, self.C = self.data.shape

        # compute valid center start index to ensure all neighbor indices >= 0:
        start_candidate = max(self.local_size, self.global_size * self.global_stride)
        self.start = int(start_candidate)
        self.num_centers = max(0, self.T - self.start)

        # build neighbor index arrays for centers = start .. T-1
        centers = np.arange(self.start, self.T, dtype=np.int32)
        self.local_idx_big = np.stack([np.arange(c - self.local_size, c, dtype=np.int32) for c in centers], axis=0)
        self.global_idx_big = np.stack([np.array([c - k * self.global_stride for k in range(1, self.global_size + 1)], dtype=np.int32) for c in centers], axis=0)
        self.global_idx_big = self.global_idx_big[:, ::-1]  # reverse order to get closest first

        # number of available windows (non-overlapping)
        self.num_windows = self.num_centers // self.win_size

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_windows:
            raise IndexError

        base = idx * self.win_size
        centers_indices = np.arange(self.start + base, self.start + base + self.win_size, dtype=np.int32)  # length W

        # neighbor indices for these centers
        local_idx_window = self.local_idx_big[base: base + self.win_size]   # (W, L)
        global_idx_window = self.global_idx_big[base: base + self.win_size] # (W, G)

        # fetch actual neighbor values from self.data
        local_window_temp = self.data[local_idx_window]    # (W, L, C)
        global_window_temp = self.data[global_idx_window]  # (W, G, C)

        # transpose to (W, C, L) and (W, C, G)
        local_window = np.transpose(local_window_temp, (0, 2, 1)).astype(np.float32)
        global_window = np.transpose(global_window_temp, (0, 2, 1)).astype(np.float32)

        W = self.win_size
        C = self.C
        L = self.local_size
        G = self.global_size

        # Build local_delay and global_delay
        local_delay = np.zeros((W, C, L, L), dtype=np.float32)
        global_delay = np.zeros((W, C, G, G), dtype=np.float32)

        for wi in range(W):
            neighbor_abs = local_idx_window[wi]  # shape (L,)
            neighbor_rows = (neighbor_abs - self.start).astype(np.int32)
            idx_matrix = self.local_idx_big[neighbor_rows]  # (L, L)
            for ch in range(C):
                local_delay[wi, ch, :, :] = self.data[idx_matrix, ch].astype(np.float32)

            neighbor_abs_g = global_idx_window[wi]  # (G,)
            neighbor_rows_g = (neighbor_abs_g - self.start).astype(np.int32)
            idx_matrix_g = self.global_idx_big[neighbor_rows_g]  # (G, G)
            for ch in range(C):
                global_delay[wi, ch, :, :] = self.data[idx_matrix_g, ch].astype(np.float32)

        # GAF transforms for windows (W, C, L) -> (W, C, L, L) and (W,C,G,G)
        local_gaf = gaf_diff_transform(local_window)
        global_gaf = gaf_diff_transform(global_window)

        # labels for centers in this window (if exist)
        if self.labels is not None:
            labels_window = self.labels[centers_indices].astype(np.int64)
        else:
            labels_window = np.zeros((W,), dtype=np.int64)

        # current center true values (W, C)
        current_values = self.data[centers_indices].astype(np.float32)  # (W, C)

        # convert to torch tensors
        local_window_t = torch.from_numpy(local_window)       # (W, C, L)
        global_window_t = torch.from_numpy(global_window)     # (W, C, G)
        local_delay_t = torch.from_numpy(local_delay)         # (W, C, L, L)
        global_delay_t = torch.from_numpy(global_delay)       # (W, C, G, G)
        local_gaf_t = torch.from_numpy(local_gaf)             # (W, C, L, L)
        global_gaf_t = torch.from_numpy(global_gaf)           # (W, C, G, G)
        labels_t = torch.from_numpy(labels_window)            # (W,)
        current_values_t = torch.from_numpy(current_values)   # (W, C)

        return {
            "local_window": local_window_t,
            "global_window": global_window_t,
            "local_delay": local_delay_t,
            "global_delay": global_delay_t,
            "local_gaf": local_gaf_t,
            "global_gaf": global_gaf_t,
            "labels": labels_t,
            "current_value": current_values_t
        }


def get_loader_segment(index, data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD',
                       local_size=3, global_size=3, global_stride=None, num_workers=4):
    prefix = dataset
    loader = NeighborSegLoader(data_path, prefix,
                               win_size=win_size,
                               local_size=local_size,
                               global_size=global_size,
                               global_stride=global_stride,
                               mode=mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=loader,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             drop_last=False)

    return data_loader
