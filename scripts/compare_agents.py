"""Compare the language-model investigator (Qwen3-1.7B / 4B / 8B) with the rule-based root-cause analysis.

    python scripts/compare_agents.py                       # all models, 150 BGL cases + the 4 OpenStack anomalies
    python scripts/compare_agents.py --cases 20 --models Qwen/Qwen3-1.7B     # quick pilot

Ground truth for "was the root cause right?":
  * BGL: every window of the held-out last ~18% of the log that holds alert lines carries the alert tags of those
    lines (APPSEV, KERNTERM, ...). The cause chosen for a window is correct if it is one of its tags. The BGL
    root-cause signatures were learned from the TRAINING windows only, so the test windows' labels were never used.
    The label column is replaced by "-" in the text the system analyses, so no tool or model can read the answer.
  * OpenStack: the 4 labelled instances are injected delays visible in "Took N seconds to spawn"; the slow-spawn cause
    is correct. (The slow-build cause says the spawn was NOT the slow part, so it is wrong here.)
Only anomalies the detector flagged (and that are truly anomalous) are scored: this compares root-cause analysis,
not detection, and every configuration sees the identical detector output.

Configurations
  rules                    rule-based root-cause analysis (no language model)
  retrieval                the knowledge-base text retrieval alone (top-1), the non-LLM way to cover unseen templates
  random                   a random pick from the investigator's shortlist
  <model>, no evidence     the model picks from the shortlist without gathering any evidence
  <model>, investigation   the full loop: the model chooses its own evidence steps, then the cause
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from log_agents import config  # noqa: E402
from log_agents.agents.coordinator import CoordinatorAgent  # noqa: E402
from log_agents.context import RunOptions  # noqa: E402
from log_agents.datasets import get_profile  # noqa: E402
from log_agents.ingestion import TextSource  # noqa: E402
from log_agents.resources import Resources  # noqa: E402
from log_agents.slm import model_available  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
SEED = 23                 # the evaluation seed (design work used other seeds and a separate development set)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════ evaluation inputs ═══════════════════════════
def bgl_input(n_anom, n_norm, region="test", seed=SEED):
    """Whole 100-message windows from the held-out part of BGL, labels hidden. Returns (text, {window: tags})."""
    cache = EVAL_DIR / f"bgl_{region}_{n_anom}_{n_norm}_{seed}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["text"], d["truth"]
    profile = get_profile("bgl")
    lines = (config.BGL_DATASET_DIR / "BGL.log").read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    part = lines[int(n * 0.82):] if region == "test" else lines[int(n * 0.30):int(n * 0.70)]     # dev = training period
    bronze, _ = profile.clean("\n".join(part), "bgl_raw")
    full = bronze.groupby("block_id").size()
    full = set(full[full == config.BGL_WINDOW].index)
    tags = bronze[bronze.label != "-"].groupby("block_id").label.agg(lambda s: sorted(set(s)))
    anomalous = sorted(w for w in full if w in tags.index)
    normal = sorted(w for w in full if w not in tags.index)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(anomalous, min(n_anom, len(anomalous))) + rng.sample(normal, n_norm))
    out = bronze[bronze.block_id.isin(chosen)].sort_values("line_no")
    hidden = ["- " + r.split(" ", 1)[1] for r in out.raw]                      # the alert tag is not shown
    truth = {f"win_{i:07d}": tags[w] if w in tags.index else [] for i, w in enumerate(chosen)}
    text = "\n".join(hidden)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"text": text, "truth": truth}), encoding="utf-8")
    return text, truth


def openstack_input(n_norm=100):
    """All 4 labelled-anomalous instances and n_norm normal ones (every tagged line) plus untagged lines."""
    profile = get_profile("openstack")
    src = config.OPENSTACK_DATASET_DIR
    bad = {ln.strip() for ln in (src / "anomaly_labels.txt").read_text().splitlines() if len(ln.strip()) == 36}
    rng = random.Random(SEED)
    abn, nor = [(src / f"openstack_{n}.log").read_text(encoding="utf-8", errors="replace") for n in ("abnormal", "normal2")]
    b_n, _ = profile.clean(nor, "openstack_raw")
    keep_ids = bad | set(rng.sample(sorted(b_n.block_id.dropna().unique()), n_norm))
    frames = []
    for text in (abn, nor):
        b, _ = profile.clean(text, "openstack_raw")
        frames += [b[b.block_id.isin(keep_ids)], b[b.block_id.isna()].sample(80, random_state=SEED)]
    import pandas as pd
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    return "\n".join(out.raw), bad


# ═══════════════════════════ scoring ═══════════════════════════
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def run(coord, text, model, steps, cases, decision=config.AGENT_DECISION):
    return coord.run(_batch(text), RunOptions(persist=False, agent_model=model, agent_max_steps=steps,
                                              agent_max_cases=cases, agent_decision=decision))


def _batch(text):
    return TextSource(text, name="eval").read()


def case_table(ctx, is_correct, truth_ids):
    """One row per truly-anomalous window the detector flagged: the chosen cause and what the rule engine said."""
    rows = {}
    for p in ctx.data["rca"]["sessions"]:
        if p["block_id"] in truth_ids:
            rows[p["block_id"]] = {"choice": p["primary_cause"], "rule_top": p["hypotheses"][0]["cause"],
                                   "decided_by": p["decided_by"], "ok": bool(is_correct(p["block_id"], p["primary_cause"]))}
    return rows


def summarize(rows, ctx, seconds, base=None):
    n = len(rows)
    k = sum(r["ok"] for r in rows.values())
    lo, hi = wilson(k, n)
    out = {"n": n, "correct": k, "accuracy": k / n if n else 0.0, "ci95": [round(lo, 3), round(hi, 3)],
           "seconds": round(seconds, 1)}
    if base is not None:                                           # paired against the rule engine
        wins = sum(1 for b, r in rows.items() if r["ok"] and not base[b]["ok"])
        losses = sum(1 for b, r in rows.items() if not r["ok"] and base[b]["ok"])
        from scipy.stats import binomtest
        out.update(wins_vs_rules=wins, losses_vs_rules=losses,
                   sign_test_p=round(binomtest(wins, wins + losses, 0.5).pvalue, 4) if wins + losses else 1.0)
    inv = ctx.data.get("investigation") or {}
    eps = inv.get("episodes") or []
    if eps:
        steps = Counter(s["action"] for ep in eps for s in ep["steps"])
        out.update(independent_cases=inv["cases"], model_calls=inv["model_calls"], mean_steps=round(inv["mean_steps"], 2),
                   seconds_per_case=round(inv["seconds"] / max(len(eps), 1), 2),
                   consulted_rule_ranking=round(inv["rule_ranking_used"] / len(eps), 2),
                   agrees_with_rule_top=round(inv["agrees_with_rule_top"] / len(eps), 2),
                   mean_final_confidence=round(float(np.mean([max(ep["choice_probs"].values()) for ep in eps])), 3),
                   action_counts=dict(steps.most_common()),
                   final_letter_counts=dict(Counter(
                       chr(65 + ep["candidates"].index(ep["choice"])) for ep in eps if len(ep["candidates"]) > 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.LLM_CHOICES) + ["Qwen/Qwen3-1.7B:nf4"])
    ap.add_argument("--cases", type=int, default=150, help="anomalous BGL windows to analyse")
    ap.add_argument("--normal", type=int, default=1350, help="normal BGL windows mixed in (keeps the base rate ~10%)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "agent_comparison.json"))
    ap.add_argument("--region", choices=["test", "dev"], default="test", help="dev = windows from the training period")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--decisions", nargs="+", default=[config.AGENT_DECISION],
                    help="pointwise listwise listwise_debiased")
    ap.add_argument("--datasets", nargs="+", default=["bgl", "openstack"])
    ap.add_argument("--per-case", action="store_true", help="also store each case's truth, choices and retrieval pick")
    args = ap.parse_args()

    res = Resources()
    coord = CoordinatorAgent(res)
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "gpu": __import__("torch").cuda.get_device_name(0),
              "settings": {"max_steps": config.AGENT_MAX_STEPS, "shortlist": config.AGENT_SHORTLIST, "seed": args.seed, "region": args.region},
              "datasets": {}}

    for dname in args.datasets:
        profile = get_profile(dname)
        if dname == "bgl":
            text, truth = bgl_input(args.cases, args.normal, args.region, args.seed)
            truth_ids = {w for w, t in truth.items() if t}
            is_correct = lambda b, c: c in truth.get(b, [])
        else:
            text, bad = openstack_input()
            truth_ids = None                                                     # set after the first run
            is_correct = lambda b, c: c == "slow_spawn"          # the delay is in the spawn; a slow-build cause says it is not
        log(f"=== {dname}: building the rule baseline")
        t0 = time.time()
        base_ctx = run(coord, text, config.RULES, 0, 10 ** 6)
        if dname == "openstack":
            ids = base_ctx.data["detection"]["sessions"].block_id
            truth_ids = {b for b in ids if b in bad}
        rules_rows = case_table(base_ctx, is_correct, truth_ids)
        log(f"  {len(rules_rows)} truly-anomalous cases flagged by the detector")
        d = {"cases": len(rules_rows), "configs": {}}
        d["configs"]["rules"] = summarize(rules_rows, base_ctx, time.time() - t0)

        # non-LLM baselines on the same cases
        coord.rca._prepare(profile)
        sess = base_ctx.data["detection"]["sessions"].set_index("block_id")
        per = {p["block_id"]: p for p in base_ctx.data["rca"]["sessions"]}
        rng = random.Random(args.seed)
        retr, rand, short_hit = {}, {}, 0
        for b in rules_rows:
            row, entry = sess.loc[b], per[b]
            tr = coord.rca.text_ranking(row.events)
            retr[b] = {"ok": bool(tr and is_correct(b, tr[0][0]))}
            sl = coord.investigator.shortlist(entry, row)
            rand[b] = {"ok": is_correct(b, rng.choice(sl))}
            short_hit += any(is_correct(b, c) for c in sl)
        if args.per_case:
            d["per_case"] = {b: {"truth": truth.get(b, []) if dname == "bgl" else ["slow_spawn"],
                                 "rules": rules_rows[b]["choice"],
                                 "retrieval": (coord.rca.text_ranking(sess.loc[b].events) or [[None]])[0][0]}
                             for b in rules_rows}
        for nm, rows in (("retrieval", retr), ("random_shortlist", rand)):
            d["configs"][nm] = summarize(rows, base_ctx, 0.0, rules_rows)
        d["shortlist_contains_truth"] = round(short_hit / max(len(rules_rows), 1), 3)

        for model in args.models:
            if not model_available(model):
                log(f"  skip {model}: weights not found (scripts/download_models.py)")
                d["configs"][model] = {"skipped": "weights not downloaded"}
                continue
            for decision, (mode, steps) in [(dc, m) for dc in args.decisions
                                            for m in (("no_evidence", 0), ("investigation", config.AGENT_MAX_STEPS))]:
                key = f"{model} | {decision} | {mode}"
                log(f"  {key}")
                t0 = time.time()
                ctx = run(coord, text, model, steps, 10 ** 6, decision)
                rows = case_table(ctx, is_correct, truth_ids)
                d["configs"][key] = c = summarize(rows, ctx, time.time() - t0, rules_rows)
                if args.per_case:
                    c["choices"] = {b: r["choice"] for b, r in rows.items()}
                log(f"    accuracy {c['accuracy']:.3f} ({c['correct']}/{c['n']}) in {c['seconds']:.0f}s; "
                    f"vs rules +{c['wins_vs_rules']}/-{c['losses_vs_rules']}; letters {c.get('final_letter_counts')}")
                Path(args.out).write_text(json.dumps({**report, "datasets": {**report["datasets"], dname: d}}, indent=1), encoding="utf-8")
        report["datasets"][dname] = d
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"written {args.out}")


if __name__ == "__main__":
    main()
