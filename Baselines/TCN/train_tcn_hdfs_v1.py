import argparse
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import TCNClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits.view(-1))
    preds = (probs > 0.5).to(y.dtype)
    return (preds == y.view(-1)).float().mean().item()


def binary_prf_from_logits(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits.view(-1))
    preds = (probs > float(threshold)).to(torch.long)
    y_true = y.view(-1).to(torch.long)

    tp = int(((preds == 1) & (y_true == 1)).sum().item())
    fp = int(((preds == 1) & (y_true == 0)).sum().item())
    fn = int(((preds == 0) & (y_true == 1)).sum().item())
    tn = int(((preds == 0) & (y_true == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return acc, precision, recall, f1


def pad_or_truncate(seq, seq_len: int):
    # Use 0 as PAD, so shift event ids by +1 before calling this.
    seq = list(seq)
    if len(seq) >= seq_len:
        return seq[-seq_len:]
    return [0] * (seq_len - len(seq)) + seq


class TraceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.x[idx], dtype=torch.long),
            torch.as_tensor(self.y[idx], dtype=torch.long),
        )


def run_epoch(*, model, loader, criterion, optimizer, device: str, train: bool) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_acc = 0.0
    count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb).float()
        loss = criterion(logits, yb.float().view(-1, 1))

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = int(xb.shape[0])
        with torch.no_grad():
            total_acc += binary_accuracy_from_logits(logits, yb) * bs
        count += bs

    return total_acc / max(count, 1)


def eval_epoch_metrics(*, model, loader, device: str, threshold: float):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).float()
            probs = torch.sigmoid(logits.view(-1))
            preds = (probs > float(threshold)).to(torch.long)
            y_true = yb.view(-1).to(torch.long)

            tp += int(((preds == 1) & (y_true == 1)).sum().item())
            fp += int(((preds == 1) & (y_true == 0)).sum().item())
            fn += int(((preds == 0) & (y_true == 1)).sum().item())
            tn += int(((preds == 0) & (y_true == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return acc, precision, recall, f1


def collect_probs_and_labels(*, model, loader, device: str):
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).float().view(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(probs)
            y_all.append(yb.view(-1).cpu().numpy())

    if not probs_all:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64)
    return (
        np.concatenate(probs_all, axis=0).astype(np.float32),
        np.concatenate(y_all, axis=0).astype(np.int64),
    )


def confusion_matrix_at_threshold(probs: np.ndarray, y_true: np.ndarray, threshold: float):
    preds = (probs > float(threshold)).astype(np.int64)
    y_true = y_true.astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    return tn, fp, fn, tp


def pr_curve_and_auc(probs: np.ndarray, y_true: np.ndarray):
    # Precision-Recall curve + area under curve (PR-AUC) and Average Precision (AP).
    probs = probs.astype(np.float64)
    y_true = y_true.astype(np.int64)
    if probs.size == 0:
        return np.asarray([]), np.asarray([]), 0.0, 0.0

    order = np.argsort(-probs)
    y_sorted = y_true[order]
    total_pos = int((y_sorted == 1).sum())
    if total_pos == 0:
        return np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([0.0, 0.0], dtype=np.float32), 0.0, 0.0

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos

    recall0 = np.concatenate([[0.0], recall, [1.0]]).astype(np.float32)
    prec0 = np.concatenate([[1.0], precision, [precision[-1]]]).astype(np.float32)

    pr_auc = float(np.trapz(prec0, recall0))

    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall, precision):
        if r > prev_r:
            ap += (r - prev_r) * p
            prev_r = r
    ap = float(ap)

    return recall0, prec0, pr_auc, ap


def load_hdfs_v1_traces(hdfs_npz_path: str, *, seq_len: int, max_traces: int | None, seed: int):
    # HDFS_v1 preprocessed NPZ contains:
    # - x_data: object array, each element is a per-BlockId event-id sequence
    # - y_data: int labels per BlockId (0 normal / 1 anomaly)
    npz = np.load(hdfs_npz_path, allow_pickle=True)
    if "x_data" not in npz.files or "y_data" not in npz.files:
        raise ValueError("Expected HDFS_v1 preprocessed NPZ to contain x_data and y_data")

    x_data = npz["x_data"]
    y_data = npz["y_data"].astype(np.int64)
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("x_data and y_data length mismatch")

    n = int(x_data.shape[0])
    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    if max_traces is not None:
        idx = idx[: int(max_traces)]

    max_eid = 0
    fixed = np.zeros((len(idx), int(seq_len)), dtype=np.int64)
    labels = np.zeros((len(idx),), dtype=np.int64)

    e_pat = re.compile(r"[Ee](\d+)")

    def to_event_int(e):
        # HDFS_v1 x_data often uses strings like "E22".
        if isinstance(e, (int, np.integer)):
            return int(e)
        if isinstance(e, (bytes, bytearray)):
            e = e.decode("utf-8", errors="ignore")
        if isinstance(e, str):
            s = e.strip()
            m = e_pat.search(s)
            if m:
                return int(m.group(1))
            return int(s)
        return int(e)

    for out_i, i in enumerate(idx):
        seq = x_data[int(i)]
        # Reserve PAD=0 by shifting ids by +1.
        seq_shifted = [to_event_int(e) + 1 for e in seq]
        if seq_shifted:
            max_eid = max(max_eid, max(seq_shifted))
        fixed[out_i] = np.asarray(pad_or_truncate(seq_shifted, int(seq_len)), dtype=np.int64)
        labels[out_i] = 1 if int(y_data[int(i)]) != 0 else 0

    num_tokens = int(max_eid) + 1  # includes PAD=0
    return fixed, labels, num_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCN baseline: HDFS_v1 BlockId trace anomaly classification")
    parser.add_argument(
        "--hdfs_npz",
        default=os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Dataset", "HDFS_v1", "preprocessed", "HDFS.npz")
        ),
        help="Path to Dataset/HDFS_v1/preprocessed/HDFS.npz",
    )
    parser.add_argument("--seq_len", type=int, default=64, help="Pad/truncate each BlockId trace to this length")
    parser.add_argument("--max_traces", type=int, default=50000, help="Cap number of traces (for speed)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--channels", type=str, default="64,64,64")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for anomaly=1 metrics")
    parser.add_argument("--save_pr_curve", action="store_true", help="Save test Precision-Recall curve (PNG) and points (NPZ)")

    args = parser.parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    X, y, num_tokens = load_hdfs_v1_traces(
        args.hdfs_npz, seq_len=args.seq_len, max_traces=args.max_traces, seed=args.seed
    )
    n = int(X.shape[0])
    n_test = int(n * float(args.test_frac))
    n_val = int(n * float(args.val_frac))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split produced empty train set; reduce val/test fractions or increase max_traces")

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = (X[n_train : n_train + n_val], y[n_train : n_train + n_val]) if n_val else (None, None)
    X_test, y_test = (X[n_train + n_val :], y[n_train + n_val :]) if n_test else (None, None)

    train_loader = DataLoader(TraceDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(TraceDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False, drop_last=False)
        if X_val is not None
        else None
    )
    test_loader = (
        DataLoader(TraceDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, drop_last=False)
        if X_test is not None
        else None
    )

    channels = [int(x) for x in args.channels.split(",") if x.strip()]
    model = TCNClassifier(
        num_classes=1,
        num_channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_embedding=True,
        num_tokens=num_tokens,
        embedding_dim=args.embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    out_dir = os.path.join(os.path.dirname(__file__), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "tcn_hdfs_v1_best.pt")
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        _ = run_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, train=True)
        tr_acc, tr_p, tr_r, tr_f1 = eval_epoch_metrics(model=model, loader=train_loader, device=device, threshold=args.threshold)
        msg = f"epoch={epoch} train_acc={tr_acc:.6f} train_p={tr_p:.6f} train_r={tr_r:.6f} train_f1={tr_f1:.6f}"

        if val_loader is not None:
            va_acc, va_p, va_r, va_f1 = eval_epoch_metrics(model=model, loader=val_loader, device=device, threshold=args.threshold)
            msg += f" val_acc={va_acc:.6f} val_p={va_p:.6f} val_r={va_r:.6f} val_f1={va_f1:.6f}"
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save({"model": model.state_dict(), "args": vars(args), "num_tokens": num_tokens}, best_path)
        else:
            torch.save({"model": model.state_dict(), "args": vars(args), "num_tokens": num_tokens}, best_path)

        print(msg)

    if test_loader is not None and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        te_acc, te_p, te_r, te_f1 = eval_epoch_metrics(model=model, loader=test_loader, device=device, threshold=args.threshold)
        probs, y_true = collect_probs_and_labels(model=model, loader=test_loader, device=device)
        tn, fp, fn, tp = confusion_matrix_at_threshold(probs, y_true, threshold=args.threshold)
        recall_pts, prec_pts, pr_auc, ap = pr_curve_and_auc(probs, y_true)

        print(f"test_acc={te_acc:.6f} test_p={te_p:.6f} test_r={te_r:.6f} test_f1={te_f1:.6f} pr_auc={pr_auc:.6f} ap={ap:.6f}")
        print(f"test_cm=[[{tn}, {fp}], [{fn}, {tp}]] threshold={args.threshold}")

        if args.save_pr_curve:
            out_dir = os.path.join(os.path.dirname(__file__), args.out_dir)
            os.makedirs(out_dir, exist_ok=True)
            pts_path = os.path.join(out_dir, "test_pr_curve_points.npz")
            np.savez_compressed(pts_path, recall=recall_pts, precision=prec_pts, pr_auc=np.asarray([pr_auc]), ap=np.asarray([ap]))

            try:
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(6, 4))
                plt.plot(recall_pts, prec_pts, linewidth=2)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Test PR Curve (AP={ap:.4f}, PR-AUC={pr_auc:.4f})")
                plt.grid(True, alpha=0.3)
                img_path = os.path.join(out_dir, "test_pr_curve.png")
                plt.tight_layout()
                plt.savefig(img_path, dpi=160)
                plt.close(fig)
                print(f"Saved PR curve: {img_path}")
                print(f"Saved PR points: {pts_path}")
            except Exception as e:
                print(f"Could not save PR curve plot (matplotlib missing?): {e}")

    print(f"Saved: {best_path}")
