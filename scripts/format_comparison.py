"""Print the agent-comparison results (artifacts/agent_comparison.json) as Markdown tables for the README.

    python scripts/format_comparison.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def label(key):
    if "|" not in key:
        return {"rules": "Rule-based RCA (the old engine)", "retrieval": "Text retrieval over the knowledge base (no LLM)",
                "random_shortlist": "Random pick from the shortlist"}.get(key, key)
    model, _, mode = [p.strip() for p in key.split("|")]
    name = model.replace("Qwen/", "").replace(":nf4", " (4-bit)")
    return f"{name}, " + ("investigation" if mode == "investigation" else "no evidence")


def table(ds):
    rows = ["| Configuration | Correct | Accuracy (95% CI) | vs rules (wins / losses, sign test p) | Seconds per case | "
            "Steps per case | Asked the rule engine | Same cause as rules |", "|---|---|---|---|---|---|---|---|"]
    for key, c in ds["configs"].items():
        if "skipped" in c:
            rows.append(f"| {label(key)} | not run: {c['skipped']} | | | | | | |")
            continue
        vs = f"{c['wins_vs_rules']} / {c['losses_vs_rules']}, p = {c['sign_test_p']:.3g}" if "wins_vs_rules" in c else "–"
        agent = "investigation" in key
        rows.append(f"| {label(key)} | {c['correct']}/{c['n']} | **{c['accuracy']:.2f}** ({c['ci95'][0]:.2f}–{c['ci95'][1]:.2f}) | {vs} | "
                    + (f"{c['seconds_per_case']:.1f}" if "seconds_per_case" in c else "–") + " | "
                    + (f"{c['mean_steps']:.1f}" if agent else "–") + " | "
                    + (f"{c['consulted_rule_ranking']:.0%}" if agent else "–") + " | "
                    + (f"{c['agrees_with_rule_top']:.0%}" if "agrees_with_rule_top" in c else "–") + " |")
    return "\n".join(rows)


def main():
    d = json.loads((ROOT / "artifacts" / "agent_comparison.json").read_text(encoding="utf-8"))
    print(f"_Run {d['date']} on {d['gpu']}; settings {d['settings']}_\n")
    for name, ds in d["datasets"].items():
        print(f"### {name.upper()} ({ds['cases']} truly-anomalous windows/instances flagged by the detector; the final menu always "
              f"contains the true cause in {ds['shortlist_contains_truth']:.0%} of cases)\n")
        print(table(ds) + "\n")
        for key, c in ds["configs"].items():
            if "investigation" in key and "action_counts" in c:
                print(f"- `{label(key)}` tool usage: " + ", ".join(f"{k} {v}" for k, v in c["action_counts"].items())
                      + f"; independent cases {c['independent_cases']}; mean final confidence {c['mean_final_confidence']:.2f}; "
                        f"final-menu position chosen: {c['final_letter_counts']}")
        print()


if __name__ == "__main__":
    main()
