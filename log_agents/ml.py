"""The existing ML pipeline: Transformer encoder → FAISS vector DB → RAG augmentation → Random Forest.

Architecture and feature construction mirror OtherTests/RAG__Trans_RF.ipynb exactly (HDFS) and
scripts/train_dataset.py (BGL, OpenStack), which reuses the same classes:
  * sequences encoded with event2id, truncated to the LAST `max_len` events (60 for HDFS), right-padded with 0
  * feature = masked mean of the transformer encoder output (64-d)
  * RAG feature = concat(feature, mean of the k=5 nearest training features[, numeric session features])
    NOTE: training used the *raw* (un-normalised) neighbour features for the mean, while the FAISS
    index stores L2-normalised vectors. rag_metadata.npz restores the raw neighbour features so
    inference matches training (the old Streamlit app averaged the normalised vectors).
"""
import json
import math
import warnings

import joblib
import numpy as np
import torch
import torch.nn as nn

from . import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, num_layers, dim_feedforward, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                               dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, num_classes)

    def extract_features(self, x):
        pad_mask = (x == 0)
        e = self.pos_enc(self.emb(x))
        out = self.encoder(e, src_key_padding_mask=pad_mask)
        mask_f = (~pad_mask).unsqueeze(-1).float()
        return (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

    def forward(self, x):
        return self.fc(self.extract_features(x))


def load_backbone(paths, device="cpu"):
    """paths: datasets.base.ArtifactPaths of one dataset."""
    event2id = joblib.load(paths.event2id)
    id2event = joblib.load(paths.id2event)
    cfg = joblib.load(paths.model_config)
    model = TransformerModel(**{k: cfg[k] for k in
                                ["vocab_size", "emb_dim", "nhead", "num_layers", "dim_feedforward", "num_classes"]})
    model.load_state_dict(torch.load(paths.backbone, map_location=device))
    model.to(device).eval()
    return model, event2id, id2event, cfg


def encode_sequences(seqs, event2id, max_len=config.MAX_SEQ_LEN):
    """List of event lists -> (N, max_len) int64 array, last max_len events, right padded."""
    arr = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        ids = [event2id[e] for e in seq if e in event2id][-max_len:]
        arr[i, :len(ids)] = ids
    return arr


@torch.no_grad()
def extract_features(model, encoded, device="cpu", batch_size=2048):
    feats = []
    for s in range(0, len(encoded), batch_size):
        x = torch.as_tensor(encoded[s:s + batch_size], device=device)
        feats.append(model.extract_features(x).float().cpu().numpy())
    return np.vstack(feats) if feats else np.zeros((0, model.emb.embedding_dim), np.float32)


class AnomalyPipeline:
    """Loads all artifacts of one dataset once; `predict` scores a batch of event sequences."""

    def __init__(self, profile, device=None):
        import faiss  # imported lazily: heavy
        self.faiss = faiss
        self.profile = profile
        paths = profile.paths
        missing = paths.missing()
        if missing:
            raise FileNotFoundError(
                f"{profile.title} model artifacts not found in {paths.directory} (missing: {', '.join(missing)}). "
                f"Train them with: python scripts/train_dataset.py --dataset {profile.name}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = profile.max_seq_len
        self.model, self.event2id, self.id2event, self.model_config = load_backbone(paths, self.device)
        with warnings.catch_warnings():
            # The HDFS RF was pickled with sklearn 1.7.0 and loads under newer versions with a warning.
            warnings.simplefilter("ignore")
            self.rf = joblib.load(paths.rf)
        self.index = faiss.read_index(str(paths.faiss))

        self.meta = None
        if paths.rag_meta.exists():
            m = np.load(paths.rag_meta, allow_pickle=False)
            self.meta = {k: m[k] for k in m.files}
        self.event_stats = {}
        if paths.event_stats.exists():
            self.event_stats = json.loads(paths.event_stats.read_text())

    @property
    def has_metadata(self):
        return self.meta is not None

    def status(self):
        return {
            "device": self.device,
            "vocab_size": self.model_config["vocab_size"],
            "faiss_vectors": int(self.index.ntotal),
            "rf_estimators": int(self.rf.n_estimators),
            "rag_metadata": self.has_metadata,
        }

    def _neighbour_raw_features(self, idx):
        """(N, k) FAISS ids -> (N, k, d) features in the space the RF was trained on."""
        if self.meta is not None:
            return self.meta["uniq_feats"][self.meta["faiss2uniq"][idx]]
        # Fallback without metadata: normalised vectors (slight train/serve skew)
        flat = np.vstack([self.index.reconstruct(int(i)) for i in idx.ravel()])
        return flat.reshape(idx.shape[0], idx.shape[1], -1)

    def predict(self, sequences, k=config.RAG_K, extras=None):
        """sequences: list of event lists; extras: optional (N, m) numeric session features (NaN = missing).
        Returns a dict of arrays aligned with the input."""
        encoded = encode_sequences(sequences, self.event2id, self.max_len)
        if extras is not None:
            extras = np.nan_to_num(np.asarray(extras, dtype=np.float32), nan=-1.0)
            key = np.hstack([encoded.astype(np.float32), extras])
        else:
            key = encoded
        # Identical (truncated) sequences with identical numeric features give identical predictions
        _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        uniq = encoded[first]
        feats = extract_features(self.model, uniq, self.device)
        q = feats.astype(np.float32).copy()
        self.faiss.normalize_L2(q)
        dist, idx = self.index.search(q, k)
        nb_mean = self._neighbour_raw_features(idx).mean(axis=1)
        if self.model_config.get("rf_inputs", "full") == "extras":
            aug = extras[first]
        else:
            aug = np.concatenate([feats, nb_mean] + ([extras[first]] if extras is not None else []), axis=1)
        proba = self.rf.predict_proba(aug)[:, list(self.rf.classes_).index(1)]
        empty = (uniq != 0).sum(axis=1) == 0
        proba[empty] = np.nan
        return {
            "anomaly_proba": proba[inverse],
            "neighbour_ids": idx[inverse],
            "neighbour_dist": dist[inverse],
        }

    def neighbour_details(self, faiss_ids, dists):
        """Describe retrieved historical sequences (needs rag_metadata.npz for sequences/labels)."""
        out = []
        for fid, d in zip(faiss_ids, dists):
            item = {"faiss_id": int(fid), "distance": float(d)}
            if self.meta is not None:
                u = int(self.meta["faiss2uniq"][fid])
                seq = [self.id2event[int(t)] for t in self.meta["uniq_seqs"][u] if t != 0]
                count, anom = int(self.meta["uniq_count"][u]), int(self.meta["uniq_anom"][u])
                item.update({
                    "sequence": seq,
                    "block_id": self.meta["uniq_block"][u].decode() if isinstance(self.meta["uniq_block"][u], bytes)
                    else str(self.meta["uniq_block"][u]),
                    "occurrences": count,
                    "anomaly_rate": anom / count if count else 0.0,
                    "label": "Anomaly" if anom * 2 > count else "Normal",
                })
            out.append(item)
        return out
