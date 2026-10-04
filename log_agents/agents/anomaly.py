"""Anomaly Detection Agent: invokes the existing ML pipeline and decides when to ask the SLM.

Policy
  1. Score every Gold session with Transformer → FAISS → RAG → Random Forest.
  2. Flag a session as *uncertain* when the RF probability is within threshold ± UNCERTAIN_MARGIN or the RF
     disagrees with the label majority of its retrieved neighbours.
  3. If the fine-tuned LoRA SLM is available, ask it for a second opinion on the uncertain unique
     sequences (lora_mode=auto) or on every unique sequence (lora_mode=all), capped for latency.
  4. The RF decision is always the final label. The SLM vote is advisory: it is shown as evidence
     and disagreements are flagged for review. Measured on the 62 uncertain sessions of the
     held-out CV fold, the RF made 6 errors and the LoRA 27 (all false positives), so letting
     the SLM override the RF made results worse.
"""
import numpy as np

from .. import config
from .base import Agent, AgentError


class AnomalyDetectionAgent(Agent):
    name = "Anomaly Detection Agent"
    task = "Detect anomalous block sessions with the ML pipeline (+ SLM second opinion)"

    def __init__(self, resources):
        self.res = resources

    def run(self, ctx):
        gold = ctx.data["medallion"]["gold"].copy()
        pipe = self.res.pipeline()
        opts = ctx.options
        out = pipe.predict(gold.events.tolist(), k=opts.rag_k)
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

        gold["slm_label"] = None
        gold["slm_disagrees"] = False
        gold["is_anomaly"] = gold.rf_label
        gold["decision_source"] = "Transformer+RAG+RF"
        slm_note = self._second_opinion(ctx, gold)

        ctx.data["detection"] = {"sessions": gold, "slm_note": slm_note, "threshold": opts.threshold}
        n_anom = int(gold.is_anomaly.sum())
        ctx.data["detection"]["metrics"] = self._metrics(gold)
        ctx.send(self.name, "Root Cause Analysis Agent", f"{n_anom} anomalous sessions ready for RCA")
        return (f"{n_anom:,}/{len(gold):,} sessions anomalous "
                f"({int(gold.uncertain.sum())} uncertain; {slm_note})")

    def _second_opinion(self, ctx, gold):
        mode = ctx.options.lora_mode
        lora = self.res.lora()
        if mode == "off":
            return "SLM second opinion disabled"
        if lora is None:
            return "LoRA adapter not found - SLM second opinion skipped"
        cand = gold if mode == "all" else gold[gold.uncertain]
        if cand.empty:
            return "no uncertain sessions - SLM not needed"
        uniq = cand.drop_duplicates("sequence")
        if len(uniq) > config.LORA_MAX_SEQUENCES:
            uniq = uniq.sort_values("anomaly_proba", key=lambda s: (s - 0.5).abs()).head(config.LORA_MAX_SEQUENCES)
            ctx.warn(f"SLM second opinion limited to the {config.LORA_MAX_SEQUENCES} most uncertain patterns.")
        try:
            votes = dict(zip(uniq.sequence, lora.classify(uniq.events.tolist())))
        except Exception as e:  # SLM failure must not break detection
            ctx.warn(f"LoRA SLM failed: {e}")
            return f"LoRA SLM failed ({type(e).__name__}); RF decision kept"
        gold["slm_label"] = gold.sequence.map(votes)
        asked = gold.slm_label.notna()
        disagree = asked & ((gold.slm_label == "Anomaly").astype(int) != gold.rf_label)
        gold["slm_disagrees"] = disagree
        n_asked = int(asked.sum())
        return (f"LoRA SLM (advisory) consulted on {len(votes)} pattern(s) / {n_asked} session(s); "
                f"disagrees with RF on {int(disagree.sum())}")

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
