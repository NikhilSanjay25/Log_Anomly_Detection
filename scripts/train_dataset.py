"""Train the Transformer → FAISS → RAG → Random Forest pipeline for BGL or OpenStack.

    python scripts/train_dataset.py --dataset bgl
    python scripts/train_dataset.py --dataset openstack

Everything is built with the same code that runs at inference time (profile.clean → profile.build_silver →
build_gold) and the saved artifacts are evaluated through the real `AnomalyPipeline`, so training and serving
cannot drift apart. Output goes to artifacts/<dataset>/:

  drain_templates.json / events.json    learned log templates and what each event means
  transformer_backbone.pth, model_config.joblib, event2id.joblib, id2event.joblib
  faiss.index, rag_metadata.npz         vector index of the training sessions + what each vector is
  rf.joblib                             Random Forest on [embedding, mean of k neighbours (excluding itself), extras]
  event_stats.json                      per-event anomaly rates, normal length / timing statistics (used by RCA)
  alert_signatures.json                 (BGL) which events identify each labelled alert category
  metrics.json                          evaluation

BGL has no session id: sessions are windows of config.BGL_WINDOW consecutive messages, anomalous if they contain
an alert line, and the evaluation split is chronological (first 80% train, last 20% test).
OpenStack has only 4 labelled anomalies (injected delays), so it is evaluated with leave-one-anomaly-out
cross-validation (4 folds, 1 anomaly each) and the deployed model is trained on everything.
"""
import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from log_agents import config  # noqa: E402
from log_agents.datasets import get_profile  # noqa: E402
from log_agents.datasets.base import ArtifactPaths  # noqa: E402
from log_agents.datasets.mined import LEVEL_KIND  # noqa: E402
from log_agents.drain import DrainParser  # noqa: E402
from log_agents.medallion import build_gold  # noqa: E402
from log_agents.ml import AnomalyPipeline, TransformerModel, encode_sequences, extract_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
K = config.RAG_K
SEED = 42
BGL_CHUNK = 500_000                 # lines per chunk (a multiple of the window size)
BGL_RF = {"n_estimators": 300, "min_samples_leaf": 2, "class_weight": "balanced_subsample"}
# OpenStack has 4 positives: every tree must see every sample and every feature (the timings are 6 of ~134 columns)
OPENSTACK_RF = {"n_estimators": 100, "bootstrap": False, "max_features": None, "class_weight": "balanced"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════ template mining ═══════════════════════════
def make_events(parser, counts):
    """events.json: one entry per Drain cluster. counts: cid -> {'n', 'levels': Counter, 'components': Counter}."""
    events = {}
    for cid, tokens in sorted(parser.templates.items()):
        c = counts.get(cid)
        if not c or not c["n"]:
            continue
        level = c["levels"].most_common(1)[0][0]
        events[f"E{cid}"] = {"template": " ".join(tokens), "component": c["components"].most_common(1)[0][0],
                             "level": level, "kind": LEVEL_KIND.get(level, "normal"), "count": int(c["n"])}
    return events


def new_counts():
    return defaultdict(lambda: {"n": 0, "levels": Counter(), "components": Counter()})


# ═══════════════════════════ models ═══════════════════════════
def train_transformer(encoded, y, vocab_size, epochs, device, pos_weight):
    cfg = {"vocab_size": vocab_size, "emb_dim": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 256,
           "num_classes": 2}
    torch.manual_seed(SEED)
    model = TransformerModel(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))
    X, Y = torch.as_tensor(encoded), torch.as_tensor(y, dtype=torch.float32)
    gen = torch.Generator().manual_seed(SEED)
    model.train()
    for ep in range(epochs):
        perm, total, nb = torch.randperm(len(X), generator=gen), 0.0, 0
        for s in range(0, len(X), 256):
            idx = perm[s:s + 256]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb)[:, 1], yb)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        log(f"    epoch {ep + 1}/{epochs}  loss {total / max(nb, 1):.4f}")
    model.eval()
    return model, cfg


def neighbours_excluding_self(index, q, k):
    """k nearest training vectors of every training vector, leaving out the vector itself.

    The RF is trained on the mean of its neighbours' features. If a training session retrieved itself, the RF
    would learn from a neighbour that never exists for a new session, so each row's own entry is dropped."""
    _, idx = index.search(q, k + 1)
    is_self = idx == np.arange(len(q))[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")           # non-self first, original order kept
    return np.take_along_axis(idx, order, axis=1)[:, :k]


def fit_and_save(out_dir, profile_name, title, max_len, event2id, seqs, y, session_ids, extras, epochs,
                 device, pos_weight, rf_params, rf_inputs="full"):
    """Train on (seqs, y[, extras]) and write the standard artifacts into out_dir. Returns a small info dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths.standard(out_dir)
    encoded = encode_sequences(seqs, event2id, max_len)
    model, cfg = train_transformer(encoded, y, len(event2id) + 1, epochs, device, pos_weight)
    feats = extract_features(model, encoded, device)
    normed = feats.astype(np.float32).copy()
    faiss.normalize_L2(normed)
    index = faiss.IndexFlatL2(normed.shape[1])
    index.add(normed)
    nb = neighbours_excluding_self(index, normed, K)
    nb_mean = feats[nb].mean(axis=1)
    extras = np.nan_to_num(extras, nan=-1.0) if extras is not None else None
    rf_X = extras if rf_inputs == "extras" else np.concatenate([feats, nb_mean] + ([extras] if extras is not None else []), axis=1)
    rf = RandomForestClassifier(n_jobs=-1, random_state=SEED, **{"n_estimators": 100, **rf_params})
    rf.fit(rf_X, y)

    # metadata: what each FAISS vector is (same schema as scripts/build_rag_metadata.py)
    uniq, first, inverse = np.unique(encoded, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    np.savez_compressed(
        paths.rag_meta, faiss2uniq=inverse.astype(np.int32), uniq_seqs=uniq.astype(np.int16),
        uniq_feats=feats[first].astype(np.float32), uniq_count=np.bincount(inverse).astype(np.int32),
        uniq_anom=np.bincount(inverse, weights=y).astype(np.int32),
        uniq_block=np.array([session_ids[i] for i in first]).astype("S64"))
    torch.save(model.state_dict(), paths.backbone)
    joblib.dump({**cfg, "rf_inputs": rf_inputs}, paths.model_config)
    joblib.dump(event2id, paths.event2id)
    joblib.dump({v: k for k, v in event2id.items()}, paths.id2event)
    faiss.write_index(index, str(paths.faiss))
    joblib.dump(rf, paths.rf)
    return {"paths": paths, "view": SimpleNamespace(paths=paths, max_seq_len=max_len, title=title, name=profile_name)}


def event_stats(gold, y, event2id, extras_cols=()):
    """Per-event statistics, in the same format as the HDFS event_stats.json (used by the RCA agent)."""
    n = len(gold)
    stats = {"n_sessions": int(n), "anomaly_rate": float(np.mean(y)), "events": {}}
    lengths = gold.n_events.to_numpy()
    norm = y == 0
    stats["normal_length"] = {"median": float(np.median(lengths[norm])), "p05": float(np.percentile(lengths[norm], 5)),
                              "p95": float(np.percentile(lengths[norm], 95))}
    sets = [set(e) for e in gold.events]
    for ev in sorted(event2id, key=lambda e: int(e[1:])):
        has = np.fromiter((ev in s for s in sets), bool, n)
        cnt = int(has.sum())
        stats["events"][ev] = {
            "sessions": cnt,
            "anomaly_rate_when_present": float(y[has].mean()) if cnt else None,
            "presence_in_normal": float(has[norm].mean()) if norm.any() else 0.0,
            "presence_in_anomaly": float(has[~norm].mean()) if (~norm).any() else 0.0,
        }
    numeric = {}
    for col in extras_cols:
        if col not in gold or not col.startswith("took_"):
            continue
        v = gold.loc[norm, col].dropna().to_numpy()
        if len(v):
            numeric[col] = {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                            "max_normal": float(v.max()), "threshold": float(v.max())}
    if numeric:
        stats["numeric"] = numeric
    return stats


def prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "tp": int(((pred == 1) & (y == 1)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()), "tn": int(((pred == 0) & (y == 0)).sum())}


def score(view, gold, extras):
    """Probabilities from the real inference path (AnomalyPipeline.predict); extras: (N, m) matrix or None."""
    pipe = AnomalyPipeline(view)
    return pipe.predict(gold.events.tolist(), k=K, extras=extras)["anomaly_proba"]


# ═══════════════════════════ BGL ═══════════════════════════
def iter_chunks(path, size):
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            lines = [ln for _, ln in zip(range(size), f)]
            if not lines:
                return
            yield "".join(lines)


def train_bgl(args, device):
    profile = get_profile("bgl")
    src = config.BGL_DATASET_DIR / "BGL.log"
    out = config.ARTIFACTS_ROOT / "bgl"
    out.mkdir(parents=True, exist_ok=True)
    limit = args.max_lines or 10 ** 12

    # pass 1: learn the templates
    log(f"BGL pass 1: mining templates from {src.name}")
    parser, counts, n_lines = DrainParser(extra_masks=profile.drain_masks), new_counts(), 0
    for text in iter_chunks(src, BGL_CHUNK):
        for ln in text.splitlines():
            parts = ln.split(None, 9)
            if len(parts) < 10:
                continue
            cid = parser.add(parts[9])
            c = counts[cid]
            c["n"] += 1
            c["levels"][parts[8].upper()] += 1
            c["components"][parts[7]] += 1
            n_lines += 1
            if n_lines >= limit:
                break
        if n_lines >= limit:
            break
    events = make_events(parser, counts)
    log(f"  {n_lines:,} lines → {len(events)} templates")
    profile.install(parser, events)
    parser.save(out / "drain_templates.json")
    (out / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")

    # pass 2: sessions (windows) through the production cleaning / Silver / Gold code
    log("BGL pass 2: windows")
    golds, tag_event, event_lines, offset, seen = [], Counter(), Counter(), 0, 0
    for text in iter_chunks(src, BGL_CHUNK):
        bronze, stats = profile.clean(text, "bgl_raw", window_offset=offset)
        silver, _ = profile.build_silver(bronze, "bgl_raw")
        gold = build_gold(silver, bronze, "bgl_raw", profile)
        tags = bronze[bronze.label != "-"].groupby("block_id").label.agg(lambda t: sorted(set(t)))
        gold["alert_tags"] = [tags.get(b, []) for b in gold.block_id]
        offset += stats["windows"]
        golds.append(gold)
        m = silver[["line_no", "event_id"]].merge(bronze[["line_no", "label"]], on="line_no")
        for (tag, ev), n in m.groupby(["label", "event_id"]).size().items():
            tag_event[(tag, ev)] += int(n)
            event_lines[ev] += int(n)
        seen += stats["input_lines"]
        log(f"  {seen:,} lines → {offset:,} windows")
        if seen >= limit:
            break
    gold = pd.concat(golds, ignore_index=True)
    gold = gold[gold.n_events >= config.BGL_WINDOW // 2].reset_index(drop=True)      # drop chunk-tail fragments
    y = gold.true_label.astype(int).to_numpy()
    log(f"  {len(gold):,} windows, {int(y.sum()):,} anomalous ({y.mean():.1%})")

    event2id = {e: i + 1 for i, e in enumerate(sorted(events, key=lambda e: int(e[1:])))}
    sig = alert_signatures(tag_event, event_lines, events)
    (out / "alert_signatures.json").write_text(json.dumps(sig, indent=1), encoding="utf-8")
    log(f"  {len(sig)} alert categories with signatures")

    # chronological evaluation: forward chaining. Train on the first part of the log, test on the next 20%.
    # The last fold (train 80% / test 20%) is the deployed model. Hyperparameters were fixed beforehand.
    n = len(gold)

    def fit_eval(frac_train, frac_test_end, out_dir, keep_for_baselines=False):
        a, b = int(n * frac_train), int(n * frac_test_end)
        tr, te = gold.iloc[:a].reset_index(drop=True), gold.iloc[a:b].reset_index(drop=True)
        ytr, yte = y[:a], y[a:b]
        log(f"fold train {len(tr):,} windows ({ytr.mean():.1%} anomalous) · test {len(te):,} ({yte.mean():.1%} anomalous)")
        profile.fit_word_vocab(tr)                         # vocabulary from the training windows only
        pos_weight = float(min(50.0, (ytr == 0).sum() / max(ytr.sum(), 1)))
        info = fit_and_save(out_dir, "bgl", profile.title, profile.max_seq_len, event2id, tr.events.tolist(), ytr,
                            tr.block_id.tolist(), profile.extra_matrix(tr), args.epochs, device, pos_weight, BGL_RF,
                            rf_inputs=profile.rf_inputs)
        proba = score(info["view"], te, profile.extra_matrix(te))
        return tr, te, ytr, yte, proba

    folds = []
    for frac in (0.4, 0.6):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, yte, proba = fit_eval(frac, frac + 0.2, tmp)
        folds.append({"train_fraction": frac, "test_range": [frac, round(frac + 0.2, 1)],
                      **prf(yte, (proba >= 0.5).astype(int))})
        log(f"  fold F1 {folds[-1]['f1']:.3f}")
    tr, te, ytr, yte, proba = fit_eval(0.8, 1.0, out)       # the deployed model
    profile.save_word_vocab(out)
    (out / "event_stats.json").write_text(json.dumps(event_stats(tr, ytr, event2id)), encoding="utf-8")
    pred = (proba >= 0.5).astype(int)
    main = prf(yte, pred)
    folds.append({"train_fraction": 0.8, "test_range": [0.8, 1.0], **main})
    if args.dump_gold:
        pd.to_pickle({"train": tr, "test": te, "proba": proba, "events": events}, args.dump_gold)
    res = {"pipeline": main, "forward_chaining_folds": folds,
           "forward_chaining_mean_f1": float(np.mean([f["f1"] for f in folds]))}
    # context: models without order, embedding or retrieval
    cnt = lambda g: np.array([np.bincount([event2id[e] for e in ev], minlength=len(event2id) + 1) for ev in g.events])
    Xtr, Xte = cnt(tr), cnt(te)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    res["baseline_event_count_rf"] = prf(yte, rf.predict(Xte))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.log1p(Xtr), ytr)
    res["baseline_event_count_logreg"] = prf(yte, lr.predict(np.log1p(Xte)))
    n_base = len(profile.extra_features)
    words_tr, words_te = profile.extra_matrix(tr)[:, n_base:], profile.extra_matrix(te)[:, n_base:]
    rf = RandomForestClassifier(n_jobs=-1, random_state=SEED, **BGL_RF).fit(words_tr, ytr)
    res["baseline_word_features_only_rf"] = prf(yte, rf.predict(words_te))
    metrics = {"dataset": "bgl", "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "lines_used": n_lines,
               "window_size": config.BGL_WINDOW, "templates": len(events), "word_vocabulary": len(profile.word_vocab),
               "split": "forward chaining by window (train on the first 40/60/80%, test on the next 20%)",
               "train_windows": len(tr), "test_windows": len(te), "test_anomaly_rate": float(yte.mean()),
               "epochs": args.epochs, "random_forest": BGL_RF, "evaluation": res}
    save_metrics(out, metrics)
    return metrics


def alert_signatures(tag_event, event_lines, events):
    """Which events identify each labelled alert category: the share of an event's lines carrying that tag."""
    by_tag = defaultdict(dict)
    for (tag, ev), n in tag_event.items():
        if tag != "-":
            by_tag[tag][ev] = n
    levels = {e: v["level"] for e, v in events.items()}
    sig = {}
    for tag, evs in by_tag.items():
        total = sum(evs.values())
        if total < 3:
            continue
        pure = {e: n for e, n in evs.items() if n >= 3 and n / event_lines[e] >= 0.5}
        if pure:
            chosen, weight = pure, min(1.0, 0.6 + 0.4 * sum(pure.values()) / total)
        else:
            chosen, weight = dict(Counter(evs).most_common(2)), 0.5
        top = max(chosen, key=chosen.get)
        lv = {levels.get(e) for e in chosen}
        severity = "high" if lv & {"FATAL", "FAILURE", "SEVERE"} else "medium" if "ERROR" in lv else "low"
        sig[tag] = {"events": sorted(chosen, key=lambda e: int(e[1:])), "lines": total, "weight": round(weight, 3),
                    "severity": severity, "top_event": top, "top_template": events[top]["template"]}
    return sig


# ═══════════════════════════ OpenStack ═══════════════════════════
def train_openstack(args, device):
    profile = get_profile("openstack")
    src = config.OPENSTACK_DATASET_DIR
    out = config.ARTIFACTS_ROOT / "openstack"
    out.mkdir(parents=True, exist_ok=True)
    bad = {ln.strip() for ln in (src / "anomaly_labels.txt").read_text().splitlines()
           if len(ln.strip()) == 36 and ln.count("-") == 4}
    log(f"OpenStack: {len(bad)} labelled anomalous instances")
    text = "\n".join((src / f"openstack_{n}.log").read_text(encoding="utf-8", errors="replace")
                     for n in ("normal1", "normal2", "abnormal"))
    bronze, stats = profile.clean(text, "openstack_raw")
    log(f"  Bronze {len(bronze):,} lines, {stats['instances']:,} instances "
        f"({stats['lines_without_instance']:,} lines carry no instance id)")

    parser, counts = DrainParser(extra_masks=profile.drain_masks), new_counts()
    for _, r in bronze[bronze.block_id.notna()].iterrows():
        cid = parser.add(r.content)
        c = counts[cid]
        c["n"] += 1
        c["levels"][r.level] += 1
        c["components"][r.component] += 1
    events = make_events(parser, counts)
    log(f"  {len(events)} templates")
    profile.install(parser, events)
    parser.save(out / "drain_templates.json")
    (out / "events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")

    silver, unmatched = profile.build_silver(bronze, "openstack_raw")
    gold = build_gold(silver, bronze, "openstack_raw", profile)
    gold = gold.sort_values("start").reset_index(drop=True)
    y = gold.block_id.isin(bad).astype(int).to_numpy()
    cols = list(profile.extra_features)
    log(f"  {len(gold):,} instances, {int(y.sum())} anomalous, {unmatched} unmatched lines")
    event2id = {e: i + 1 for i, e in enumerate(sorted(events, key=lambda e: int(e[1:])))}
    X_extra = profile.extra_matrix(gold)
    seqs, ids = gold.events.tolist(), gold.block_id.tolist()
    pos_weight = 50.0

    def run_cv(use_extras):
        probas = np.zeros(len(gold))
        for fold, (tr, va) in enumerate(StratifiedKFold(4, shuffle=True, random_state=SEED).split(seqs, y)):
            with tempfile.TemporaryDirectory() as tmp:
                log(f"  CV fold {fold + 1}/4 ({'with' if use_extras else 'without'} timing features): "
                    f"{int(y[va].sum())} held-out anomaly, {int((y[va] == 0).sum())} held-out normal")
                info = fit_and_save(tmp, "openstack", profile.title, profile.max_seq_len, event2id,
                                    [seqs[i] for i in tr], y[tr], [ids[i] for i in tr],
                                    X_extra[tr] if use_extras else None, args.epochs, device, pos_weight, OPENSTACK_RF)
                sub = gold.iloc[va]
                probas[va] = score(info["view"], sub, X_extra[va] if use_extras else None)
        pred = (probas >= 0.5).astype(int)
        return {**prf(y, pred), "anomaly_probabilities": {gold.block_id[i][:8]: round(float(probas[i]), 3)
                                                          for i in np.where(y == 1)[0]}}

    cv_with, cv_without = run_cv(True), run_cv(False)
    log("final model on all instances")
    info = fit_and_save(out, "openstack", profile.title, profile.max_seq_len, event2id, seqs, y, ids, X_extra,
                        args.epochs, device, pos_weight, OPENSTACK_RF)
    (out / "event_stats.json").write_text(json.dumps(event_stats(gold, y, event2id, cols)), encoding="utf-8")
    metrics = {"dataset": "openstack", "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "instances": len(gold),
               "anomalous_instances": int(y.sum()), "templates": len(events), "epochs": args.epochs,
               "random_forest": OPENSTACK_RF, "evaluation_protocol": "leave-one-anomaly-out: 4 folds, each holds out 1 of the 4 labelled anomalies "
                                      "and a quarter of the normal instances; the deployed model is trained on all",
               "evaluation": {"pipeline_with_timing_features": cv_with, "pipeline_event_sequence_only": cv_without}}
    save_metrics(out, metrics)
    return metrics


def save_metrics(out, metrics):
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    log("metrics: " + json.dumps(metrics["evaluation"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bgl", "openstack"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-lines", type=int, default=0, help="BGL: use only the first N lines (quick test)")
    ap.add_argument("--dump-gold", help="BGL: also pickle the Gold windows and test probabilities here (analysis)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    {"bgl": train_bgl, "openstack": train_openstack}[args.dataset](args, device)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
