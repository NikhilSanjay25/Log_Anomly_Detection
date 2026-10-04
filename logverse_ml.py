"""
LogVerse AI Platform — ML Anomaly Detection Engine
==================================================
Loads trained PyTorch Transformer backbone (`transformer_backbone.pth`),
`event2id.joblib`, and `id2event.joblib` to perform log sequence anomaly detection,
probability scoring, feature vector extraction, and key error event pinpointing.
"""

import os
import math
import joblib
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# 1. PYTORCH TRANSFORMER BACKBONE ARCHITECTURE
# ─────────────────────────────────────────────────────────────
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
    def __init__(self, vocab_size=31, emb_dim=64, nhead=4, num_layers=2, dim_feedforward=256, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, num_classes)

    def extract_features(self, x):
        pad_mask = (x == 0)
        e = self.pos_enc(self.emb(x))
        out = self.encoder(e, src_key_padding_mask=pad_mask)
        mask_f = (~pad_mask).unsqueeze(-1).float()
        return (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.fc(features)
        return logits


class MLAnomalyDetector:
    """
    ML Anomaly Detector loaded directly from the workspace pretrained weights.
    """

    def __init__(self, artifacts_dir="."):
        self.artifacts_dir = artifacts_dir
        self.event2id = {}
        self.id2event = {}
        self.model = None
        self.loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            event2id_path = os.path.join(self.artifacts_dir, "event2id.joblib")
            id2event_path = os.path.join(self.artifacts_dir, "id2event.joblib")
            model_path = os.path.join(self.artifacts_dir, "transformer_backbone.pth")

            if os.path.exists(event2id_path):
                self.event2id = joblib.load(event2id_path)
            else:
                self.event2id = {f"E{i}": i for i in range(1, 30)}

            if os.path.exists(id2event_path):
                self.id2event = joblib.load(id2event_path)
            else:
                self.id2event = {v: k for k, v in self.event2id.items()}

            vocab_size = len(self.event2id) + 2  # default 31
            self.model = TransformerModel(vocab_size=vocab_size).to(DEVICE)

            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.loaded = True
            else:
                print(f"[WARN] {model_path} not found. Running in heuristic fallback mode.")
        except Exception as e:
            print(f"[ERROR] Failed to load ML model artifacts: {e}")

    def parse_sequence_to_ids(self, event_list, max_len=60):
        """Converts a list of event strings (e.g. ['E5', 'E22', 'E4']) into padded integer IDs."""
        ids = []
        for tok in event_list:
            tok_str = str(tok).strip().upper()
            if tok_str.isdigit():
                tok_str = f"E{tok_str}"
            if tok_str in self.event2id:
                ids.append(self.event2id[tok_str])

        if not ids:
            ids = [1]  # fallback index

        ids = ids[-max_len:]
        padded = np.zeros((1, max_len), dtype=np.int64)
        padded[0, :len(ids)] = ids
        return padded

    def predict_session(self, event_list):
        """
        Runs transformer inference on a single session's event sequence.
        Returns dict with anomaly prediction (bool), anomaly_probability,
        confidence_score, feature_vector, and pinpointed_suspicious_events.
        """
        seq_tensor_np = self.parse_sequence_to_ids(event_list)

        # Pinpoint error events
        error_events_in_seq = [e for e in event_list if e in {"E4", "E7", "E8", "E10", "E11", "E12", "E14", "E17", "E20", "E24", "E29"}]

        if self.loaded and self.model is not None:
            with torch.no_grad():
                x = torch.tensor(seq_tensor_np, dtype=torch.long).to(DEVICE)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                feat = self.model.extract_features(x).cpu().numpy()[0]

            is_anomaly = bool(probs[1] > 0.5 or len(error_events_in_seq) > 0)
            anomaly_prob = float(probs[1])
            if len(error_events_in_seq) > 0 and anomaly_prob < 0.8:
                anomaly_prob = min(0.98, anomaly_prob + 0.45)
                is_anomaly = True

            confidence = float(probs[1] if is_anomaly else probs[0])
        else:
            is_anomaly = len(error_events_in_seq) > 0
            anomaly_prob = 0.95 if is_anomaly else 0.05
            confidence = 0.90
            feat = np.zeros(64)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_probability": round(anomaly_prob, 4),
            "confidence": round(confidence, 4),
            "status_label": "ANOMALY DETECTED" if is_anomaly else "NORMAL PATTERN",
            "feature_vector": feat.tolist(),
            "suspicious_events": error_events_in_seq,
            "total_events_evaluated": len(event_list)
        }
