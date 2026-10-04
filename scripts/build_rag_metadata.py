"""Build rag_metadata.npz + event_stats.json from the training dataset (run once).

Why: hdfs_faiss_index.index only stores L2-normalised feature vectors. The training notebook
collected them through a *shuffled* DataLoader, so FAISS ids cannot be mapped back to dataset rows
directly. We recover the mapping by recomputing the backbone features of every unique (truncated)
training sequence and matching each FAISS vector to its nearest recomputed vector (distance ~0).
That gives, for each FAISS id: its event sequence, its raw (un-normalised) feature - which the
RF was trained on - and label statistics for that pattern ("similar logs" evidence for RCA).

Usage:  python scripts/build_rag_metadata.py [--eval 20000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_agents import config  # noqa: E402
from log_agents.ml import AnomalyPipeline, encode_sequences, extract_features, load_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=20000,
                    help="evaluate the pipeline on N rows of the held-out fold (0 = skip)")
    args = ap.parse_args()
    import faiss

    t0 = time.time()
    npz = np.load(config.DATASET_DIR / "HDFS.npz", allow_pickle=True)
    x_raw, y = npz["x_data"], npz["y_data"].astype(np.int64)
    blocks = pd.read_csv(config.DATASET_DIR / "anomaly_label.csv", usecols=["BlockId"])["BlockId"].to_numpy()
    assert len(blocks) == len(x_raw), "anomaly_label.csv rows must align with HDFS.npz"
    print(f"[data] {len(x_raw):,} sessions, {int(y.sum()):,} anomalies ({time.time() - t0:.0f}s)")

    model, event2id, _, _ = load_backbone("cpu")
    seqs = [[str(e).strip() for e in s] for s in x_raw]
    enc = encode_sequences(seqs, event2id)

    # ── unique truncated sequences + label statistics ─────────────────────
    uniq, first, inverse = np.unique(enc, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = np.bincount(inverse)
    anom = np.bincount(inverse, weights=y).astype(np.int64)
    print(f"[uniq] {len(uniq):,} unique sequences")

    feats = extract_features(model, uniq, "cpu")
    normed = feats.copy()
    faiss.normalize_L2(normed)

    # ── map every FAISS vector to its unique sequence ────────────────────
    index = faiss.read_index(str(config.FAISS_PATH))
    stored = index.reconstruct_n(0, index.ntotal)
    lookup = faiss.IndexFlatL2(normed.shape[1])
    lookup.add(normed)
    d, nn_idx = lookup.search(stored, 1)
    d = d.ravel()
    print(f"[match] FAISS→sequence distance: median={np.median(d):.2e} p99={np.percentile(d, 99):.2e} max={d.max():.2e}")
    if np.percentile(d, 99) > 1e-3:
        sys.exit("FAISS vectors do not match recomputed features - backbone/index mismatch.")

    np.savez_compressed(
        config.RAG_METADATA_PATH,
        faiss2uniq=nn_idx.ravel().astype(np.int32),
        uniq_seqs=uniq.astype(np.int8),
        uniq_feats=feats.astype(np.float32),
        uniq_count=count.astype(np.int32),
        uniq_anom=anom.astype(np.int32),
        uniq_block=blocks[first].astype("S32"),
    )
    print(f"[out] {config.RAG_METADATA_PATH}")

    # ── per-event statistics: P(anomaly | event present), typical counts ──
    stats = {"n_sessions": int(len(seqs)), "anomaly_rate": float(y.mean()), "events": {}}
    lengths = np.array([len(s) for s in seqs])
    stats["normal_length"] = {"median": float(np.median(lengths[y == 0])),
                              "p05": float(np.percentile(lengths[y == 0], 5)),
                              "p95": float(np.percentile(lengths[y == 0], 95))}
    for ev in sorted(event2id, key=lambda e: int(e[1:])):
        has = np.fromiter((ev in s for s in seqs), bool, len(seqs))
        n = int(has.sum())
        stats["events"][ev] = {
            "sessions": n,
            "anomaly_rate_when_present": float(y[has].mean()) if n else None,
            "presence_in_normal": float(has[y == 0].mean()),
            "presence_in_anomaly": float(has[y == 1].mean()),
        }
    config.EVENT_STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(f"[out] {config.EVENT_STATS_PATH}")

    # ── sanity evaluation on the fold the RF never trained on ─────────────
    if args.eval:
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.model_selection import StratifiedKFold
        *_, (_, val_idx) = StratifiedKFold(5, shuffle=True, random_state=42).split(enc, y)
        rng = np.random.default_rng(0)
        val_idx = rng.choice(val_idx, size=min(args.eval, len(val_idx)), replace=False)
        pipe = AnomalyPipeline(device="cpu")
        proba = pipe.predict([seqs[i] for i in val_idx])["anomaly_proba"]
        pred = (proba >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y[val_idx], pred, average="binary", zero_division=0)
        acc = (pred == y[val_idx]).mean()
        print(f"[eval] held-out fold sample n={len(val_idx):,}: acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f:.4f}")
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
