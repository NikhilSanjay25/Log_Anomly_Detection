"""Coordinator Agent (orchestration layer).

Plans the workflow, assigns tasks to agents, manages shared context + memory, handles
inter-agent communication, validates every agent's output (retrying once) and guarantees
end-to-end execution with graceful fallbacks.
"""
import time

from ..context import RunContext, RunMemory, RunOptions, TraceStep
from ..medallion import MedallionProcessor
from ..report import build_report
from .anomaly import AnomalyDetectionAgent
from .base import AgentError
from .identification import LogIdentificationAgent
from .insight import InsightAgent, deterministic_insight
from .preprocessing import PreprocessingAgent
from .rca import RootCauseAnalysisAgent

NAME = "Coordinator Agent"


class CoordinatorAgent:
    name = NAME

    def __init__(self, resources, memory=None):
        self.res = resources
        self.memory = memory or RunMemory()
        self.identification = LogIdentificationAgent()
        self.preprocessing = PreprocessingAgent()
        self.medallion = MedallionProcessor()
        self.detection = AnomalyDetectionAgent(resources)
        self.rca = RootCauseAnalysisAgent(resources)
        self.insight = InsightAgent(resources)

    def run(self, batch, options=None, on_step=None):
        """batch: ingestion.LogBatch. on_step(trace_step) is called after every step (UI progress)."""
        ctx = RunContext(batch.name, batch.text, options or RunOptions(), origin=batch.origin)
        ctx.data["memory"] = self.memory.recent(3)
        self._on_step = on_step or (lambda step: None)
        ctx.data["plan"] = ["Log Identification Agent"]
        try:
            self._execute(ctx, self.identification, critical=True)
            ctx.data["plan"] = self._plan(ctx)
            ctx.send(NAME, "all agents", "Workflow plan: " + " → ".join(ctx.data["plan"]))
            self._execute(ctx, self.preprocessing, critical=True)
            self._execute(ctx, self.medallion, critical=True)
            self._execute(ctx, self.detection, critical=True)
            self._execute(ctx, self.rca, critical=False)
            if "rca" not in ctx.data:
                ctx.data["rca"] = {"sessions": [], "causes": [], "normal_examples": [],
                                   "correlations": {"nodes": [], "time_bursts": [], "patterns": []}}
            self._execute(ctx, self.insight, critical=False,
                          fallback=lambda: ctx.data.__setitem__(
                              "insight", {**deterministic_insight(ctx), "source": "knowledge base (fallback)"}))
            if "insight" not in ctx.data:
                ctx.data["insight"] = {"summary": "Insight generation failed; see warnings.", "explanation": "",
                                       "impact": {"severity": "medium", "description": "Unknown", "affected": []},
                                       "recommendations": [], "next_best_actions": [], "source": "none (failed)"}
            ctx.data["status"] = "completed"
        except AgentError as e:
            ctx.data["status"] = "failed"
            ctx.data["error"] = str(e)
        except Exception as e:  # bug outside an agent (e.g. in a fallback): fail the run, never the caller
            ctx.data["status"] = "failed"
            ctx.data["error"] = f"Coordinator error: {type(e).__name__}: {e}"
        report = build_report(ctx)
        ctx.data["report"] = report
        if ctx.options.persist and ctx.data["status"] == "completed":
            self.memory.save(ctx.run_id, report)
        return ctx

    def _plan(self, ctx):
        ident = ctx.data["identification"]
        plan = ["Preprocessing & Cleaning Agent", "Medallion Data Processing (Bronze → Silver → Gold)",
                "Anomaly Detection Agent (Transformer → FAISS → RAG → Random Forest"
                + ("" if ctx.options.lora_mode == "off" else " + advisory LoRA SLM second opinion") + ")",
                "Root Cause Analysis Agent",
                "Insight Agent (" + ("SLM" if ctx.options.use_slm_insight else "knowledge base") + ")"]
        ctx.send(NAME, "Log Identification Agent",
                 f"Identified {ident['family']} / {ident['format']} → workflow: {ident['workflow']}")
        return plan

    def _execute(self, ctx, agent, critical, retries=1, fallback=None):
        step = TraceStep(agent.name, agent.task, status="running")
        ctx.trace.append(step)
        ctx.send(NAME, agent.name, f"Assigned task: {agent.task}")
        t0 = time.time()
        error = None
        for attempt in range(1 + retries):
            step.attempts = attempt + 1
            try:
                step.summary = agent.run(ctx) or ""
                problems = agent.validate(ctx)
                if not problems:
                    step.status = "ok"
                    break
                step.issues = problems
                error = AgentError(f"{agent.name} output failed validation: {'; '.join(problems)}")
                ctx.send(NAME, agent.name, f"Validation failed ({'; '.join(problems)}) - retrying")
            except AgentError as e:
                error = e
                break                      # domain errors are deterministic: retrying will not help
            except Exception as e:         # unexpected failure: retry once
                error = AgentError(f"{agent.name} crashed: {type(e).__name__}: {e}")
                step.issues.append(str(error))
        else:
            step.status = "failed"
        if step.status != "ok":
            step.duration_s = time.time() - t0
            if critical:
                step.status = "failed"
                self._on_step(step)
                raise error
            ctx.warn(str(error))
            if fallback:
                try:
                    fallback()
                    step.status = "fallback"
                except Exception as e:
                    step.status = "failed"
                    step.issues.append(f"fallback failed: {type(e).__name__}: {e}")
                    ctx.warn(f"{agent.name} fallback failed: {e}")
            else:
                step.status = "failed"
        step.duration_s = time.time() - t0
        ctx.send(agent.name, NAME, f"[{step.status}] {step.summary}")
        self._on_step(step)
