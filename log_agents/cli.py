"""Command-line entry point.

    python -m log_agents.cli samples/hdfs_raw_sample.log
    python -m log_agents.cli my.log --no-slm --lora off --out report.md
    python -m log_agents.cli my.log --chat          # then ask follow-up questions about the analysis
"""
import argparse
import json
import sys
from pathlib import Path

from .agents.coordinator import CoordinatorAgent
from .context import RunOptions
from .ingestion import UploadSource
from .report import to_markdown
from .resources import Resources


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agentic HDFS log anomaly detection")
    ap.add_argument("log_file")
    ap.add_argument("--no-slm", action="store_true", help="skip the Qwen3 insight SLM (knowledge-base insight)")
    ap.add_argument("--lora", choices=["auto", "all", "off"], default="auto", help="LoRA SLM second opinion")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", help="write the Markdown report here (.json for JSON)")
    ap.add_argument("--no-persist", action="store_true", help="do not write medallion layers / run memory")
    ap.add_argument("--chat", action="store_true", help="after the report, ask follow-up questions about it")
    args = ap.parse_args(argv)

    path = Path(args.log_file)
    batch = UploadSource(path.name, path.read_bytes()).read()
    opts = RunOptions(use_slm_insight=not args.no_slm, lora_mode=args.lora, threshold=args.threshold,
                      persist=not args.no_persist)

    def progress(step):
        print(f"  [{step.status:>8}] {step.agent:<34} {step.duration_s:6.1f}s  {step.summary}", file=sys.stderr)

    print(f"Analysing {path} ...", file=sys.stderr)
    coordinator = CoordinatorAgent(Resources())
    ctx = coordinator.run(batch, opts, on_step=progress)
    report = ctx.data["report"]
    text = json.dumps(report, indent=1, default=str) if args.out and args.out.endswith(".json") else to_markdown(report)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    if args.chat and report["status"] == "completed":
        chat_loop(ctx, coordinator)
    return 0 if report["status"] == "completed" else 1


def chat_loop(ctx, coordinator):
    from .chat import RunChat
    sys.stdout.reconfigure(encoding="utf-8")
    chat = RunChat(ctx, coordinator.rca)
    slm = coordinator.res.insight_slm()
    print("\nAsk follow-up questions about this analysis. Name a block (blk_...) for its details. "
          "Empty line or 'exit' quits.", file=sys.stderr)
    while True:
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        msgs, content = chat.messages_for(question)
        print("\nbot> ", end="", flush=True)
        pieces = []
        for piece in slm.stream_chat(msgs):
            print(piece, end="", flush=True)
            pieces.append(piece)
        print()
        _, issues = chat.finish(question, content, "".join(pieces))
        for issue in issues:
            print(f"  ! not backed by the facts: {issue}")


if __name__ == "__main__":
    sys.exit(main())
