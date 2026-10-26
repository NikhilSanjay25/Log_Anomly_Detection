"""Anomaly Detection Agent: scores every Gold session with the dataset's ML pipeline.

Policy
  1. Score every Gold session with Transformer → FAISS → RAG → Random Forest, using the models trained for the
     dataset the Identification agent recognised (HDFS, BGL or OpenStack). Datasets with numeric session features
     (OpenStack timings) pass them to the Random Forest as well.
  2. The RF decision (probability >= threshold) is the label.
  3. A session is flagged *uncertain* when the RF probability is within threshold ± UNCERTAIN_MARGIN or the RF
     disagrees with the label majority of its retrieved neighbours. Uncertain sessions are shown for manual review;
     the label is not changed.
"""
import numpy as np

from .. import config
from .base import Agent, AgentError


class AnomalyDetectionAgent(Agent):
    name = "Anomaly Detection Agent"
    task = "Detect anomalous sessions with the Transformer → FAISS → RAG → Random Forest pipeline"

    def __init__(self, resources):
        self.res = resources

    def run(self, ctx):
        profile = ctx.profile
        gold = ctx.data["medallion"]["gold"].copy()
        try:
            pipe = self.res.pipeline(profile.name)
        except FileNotFoundError as e:
            raise AgentError(str(e)) from None
        opts = ctx.options
        extras = profile.extra_matrix(gold)
        out = pipe.predict(gold.events.tolist(), k=opts.rag_k, extras=extras)
        proba = out["anomaly_proba"]
        if np.isnan(proba).all():
            raise AgentError("None of the sessions contained events from the model vocabulary.")

        gold["anomaly_proba"] = np.nan_to_num(proba, nan=0.0)
        gold["rf_label"] = (gold.anomaly_proba >= opts.threshold).astype(int)
        gold["nb_ids"] = list(out["neighbour_ids"])
        gold["nb_dist"] = list(out["neighbour_dist"])
        if pipe.has_metadata:
            m = pipe.meta
            rates = m["uniq_anom"] / np.maximum(m["uniq_count"], 1)
            gold["nb_anomaly_rate"] = [float(rates[m["faiss2uniq"][ids]].mean()) for ids in out["neighbour_ids"]]
            nb_vote = (gold.nb_anomaly_rate >= 0.5).astype(int)
        else:
            gold["nb_anomaly_rate"] = np.nan
            nb_vote = gold.rf_label
        lo = max(0.0, opts.threshold - config.UNCERTAIN_MARGIN)
        hi = min(1.0, opts.threshold + config.UNCERTAIN_MARGIN)
        gold["uncertain"] = gold.anomaly_proba.between(lo, hi) | (nb_vote != gold.rf_label)
        gold["is_anomaly"] = gold.rf_label
        gold["decision_source"] = "Transformer+RAG+RF"

        ctx.data["detection"] = {"sessions": gold, "threshold": opts.threshold, "dataset": profile.name,
                                 "metrics": self._metrics(gold)}
        n_anom = int(gold.is_anomaly.sum())
        ctx.send(self.name, "Root Cause Analysis Agent", f"{n_anom} anomalous {profile.session_noun_plural} ready for RCA")
        return (f"{n_anom:,}/{len(gold):,} {profile.session_noun_plural} anomalous "
                f"({int(gold.uncertain.sum())} flagged uncertain; {profile.title} models)")

    @staticmethod
    def _metrics(gold):
        if "true_label" not in gold or gold.true_label.isna().all():
            return None
        g = gold.dropna(subset=["true_label"])
        y, p = g.true_label.astype(int).values, g.is_anomaly.astype(int).values
        tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
        fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return {"n": len(g), "accuracy": (tp + tn) / len(g), "precision": prec, "recall": rec,
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def validate(self, ctx):
        s = ctx.data.get("detection", {}).get("sessions")
        if s is None:
            return ["no detection output"]
        problems = []
        if not s.anomaly_proba.between(0, 1).all():
            problems.append("probabilities outside [0, 1]")
        if not s.is_anomaly.isin([0, 1]).all():
            problems.append("invalid labels")
        return problems
