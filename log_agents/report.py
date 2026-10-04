"""Builds the final JSON report and a Markdown rendering of it."""
import time
from dataclasses import asdict
from datetime import datetime


def build_report(ctx):
    d = ctx.data
    rep = {
        "run_id": ctx.run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": ctx.source_name,
        "origin": ctx.origin,
        "status": d.get("status"),
        "error": d.get("error"),
        "duration_s": round(time.time() - ctx.started, 2),
        "identification": d.get("identification"),
        "plan": d.get("plan"),
        "preprocessing": {k: v for k, v in (d.get("preprocessing") or {}).items()},
        "trace": [asdict(s) for s in ctx.trace],
        "messages": [{"from": m.sender, "to": m.recipient, "content": m.content} for m in ctx.messages],
        "warnings": ctx.warnings,
    }
    det = d.get("detection")
    if det is not None:
        s = det["sessions"]
        rep["anomaly_summary"] = {
            "sessions": len(s), "anomalies": int(s.is_anomaly.sum()),
            "anomaly_rate": float(s.is_anomaly.mean()),
            "uncertain": int(s.uncertain.sum()), "slm_second_opinion": det["slm_note"],
            "evaluation": det.get("metrics"),
        }
        cols = ["block_id", "n_events", "anomaly_proba", "is_anomaly", "slm_label", "slm_disagrees", "sequence"]
        rep["anomalous_sessions"] = (s[s.is_anomaly == 1].sort_values("anomaly_proba", ascending=False)[cols]
                                     .head(200).to_dict("records"))
    if d.get("medallion"):
        rep["medallion_path"] = d["medallion"].get("path")
    if d.get("rca"):
        rep["root_causes"] = d["rca"]["causes"]
        rep["correlations"] = d["rca"]["correlations"]
        rep["rca_sessions"] = d["rca"]["sessions"][:50]
    if d.get("insight"):
        rep["insight"] = {k: v for k, v in d["insight"].items() if k not in ("facts", "slm_raw")}
    summ = rep.get("anomaly_summary", {})
    rep["summary"] = {"run_id": ctx.run_id, "source": ctx.source_name, "status": rep["status"],
                      "sessions": summ.get("sessions"), "anomalies": summ.get("anomalies"),
                      "top_cause": (rep.get("root_causes") or [{}])[0].get("title"),
                      "generated_at": rep["generated_at"]}
    return rep


def to_markdown(rep):
    L = [f"# HDFS Log Anomaly Report", "",
         f"- **Run:** `{rep['run_id']}`  ", f"- **Source:** {rep['source']} ({rep['origin']})  ",
         f"- **Generated:** {rep['generated_at']}  ", f"- **Status:** {rep['status']}", ""]
    if rep.get("error"):
        L += [f"> **Error:** {rep['error']}", ""]
    ident = rep.get("identification") or {}
    if ident.get("format"):
        L += [f"**Log family:** {ident['family']} · format `{ident['format']}` · confidence {ident['confidence']:.0%}", ""]
    a = rep.get("anomaly_summary")
    if a:
        L += ["## Anomaly Summary", "",
              f"| Sessions | Anomalies | Rate | Uncertain |", "|---|---|---|---|",
              f"| {a['sessions']:,} | {a['anomalies']:,} | {a['anomaly_rate']:.2%} | {a['uncertain']:,} |", "",
              f"SLM second opinion: {a['slm_second_opinion']}", ""]
        if a.get("evaluation"):
            e = a["evaluation"]
            L += [f"Ground-truth evaluation (n={e['n']:,}): accuracy {e['accuracy']:.4f}, precision {e['precision']:.4f}, "
                  f"recall {e['recall']:.4f}, F1 {e['f1']:.4f}", ""]
    ins = rep.get("insight")
    if ins:
        L += ["## AI Insights", f"_Source: {ins.get('source')}_", "", f"**Summary.** {ins['summary']}", "",
              f"**Explanation.** {ins['explanation']}", ""]
        if ins.get("normal_explanation"):
            L += [f"**Why the other sessions look normal.** {ins['normal_explanation']}", ""]
        imp = ins.get("impact", {})
        L += [f"**Impact ({imp.get('severity', '?').upper()}).** {imp.get('description', '')}", ""]
        if imp.get("affected"):
            L += ["Affected: " + ", ".join(map(str, imp["affected"])), ""]
        L += ["## Recommendations", ""] + [f"- {r}" for r in ins.get("recommendations", [])] + [""]
        L += ["## Next Best Actions", ""] + [f"{i}. {r}" for i, r in enumerate(ins.get("next_best_actions", []), 1)] + [""]
    if rep.get("root_causes"):
        L += ["## Root Cause Analysis", ""]
        for c in rep["root_causes"]:
            L += [f"### {c['title']} — {c['count']} session(s), severity {c['severity']}", "",
                  c["explanation"], "", f"*Impact:* {c['impact']}", ""]
            if c["key_events"]:
                L += ["Key events: " + "; ".join(c["key_events"]), ""]
            L += ["Example blocks: " + ", ".join(f"`{b}`" for b in c["example_blocks"]), ""]
        corr = rep.get("correlations", {})
        if any(corr.values()):
            L += ["### Correlations", ""]
            L += [f"- DataNode `{n['node']}`: {n['anomalous_blocks']}/{n['total_blocks']} blocks anomalous" for n in corr["nodes"]]
            L += [f"- Burst: {b['anomalies']} anomalies at {b['minute']}" for b in corr["time_bursts"]]
            L += [f"- Pattern repeated in {p['blocks']} blocks: `{p['sequence'][:120]}`" for p in corr["patterns"]]
            L += [""]
    if rep.get("rca_sessions"):
        L += ["## Evidence (top anomalous sessions)", ""]
        for p in rep["rca_sessions"][:10]:
            L += [f"### `{p['block_id']}` — {p['hypotheses'][0]['title']} (rule-score share {p['hypotheses'][0]['share']:.0%})", "",
                  f"Sequence: `{p['sequence'][:300]}`", ""] + [f"- {e}" for e in p["evidence"]] + [""]
    L += ["## Agent Trace", "", "| Agent | Status | Time (s) | Summary |", "|---|---|---|---|"]
    L += [f"| {t['agent']} | {t['status']} | {t['duration_s']:.2f} | {t['summary']} |" for t in rep["trace"]]
    if rep.get("warnings"):
        L += ["", "**Warnings**", ""] + [f"- {w}" for w in rep["warnings"]]
    return "\n".join(L) + "\n"
