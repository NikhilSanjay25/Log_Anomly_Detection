"""OpenStack profile (LogHub / DeepLog OpenStack logs).

A session is one VM instance: the nova log lines tagged `[instance: <uuid>]`, ordered by time. The dataset's
labelled anomalies (anomaly_labels.txt) are 4 instances with *injected delays*: their event order is identical to a
normal instance, only the "Took N seconds to spawn/build ..." timings are 1.5-2.5x longer. A model that only sees
the event sequence cannot tell them apart, so the Gold layer extracts those timings as numeric session features
that are appended to the Random Forest input (like DeepLog's parameter-value model).
"""
import re

import numpy as np
import pandas as pd

from .base import read_json
from .mined import MinedProfile, to_datetime

WS_RE = re.compile(r"\s+")
UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
LINE_RE = re.compile(r"^(?:(?P<service>\S+\.log\S*) )?(?P<date>\d{4}-\d\d-\d\d) (?P<time>\d\d:\d\d:\d\d(?:\.\d+)?) "
                     r"(?P<pid>\d+) (?P<level>[A-Z]+) (?P<component>[\w.\-]+) (?P<rest>.*)$")
OPENSTACK_COMPONENT = re.compile(r"^(?:nova|neutron|glance|cinder|keystone|oslo)[\w.]*$")
INSTANCE_RE = re.compile(r"\[instance: (" + UUID + r")\]")
TOOK_RE = re.compile(r"^Took ([\d.]+) seconds to (\w+)")
METRICS = ("spawn", "build", "destroy", "deallocate")
TIMING_FEATURES = tuple(f"took_{m}" for m in METRICS)

CAUSES = {
    "slow_spawn": {
        "title": "Slow instance spawn (hypervisor / image / network)",
        "events": set(), "weight": 1.0, "severity": "medium",
        "explanation": "The instance took much longer than any normal instance to spawn on the hypervisor (libvirt "
                       "domain creation, image preparation and network plug-in). Its event order is normal; only the "
                       "timing is abnormal.",
        "impact": "Slow VM provisioning: users wait longer for instances, and build timeouts become likely. Often a "
                  "symptom of an overloaded compute host, a slow image or disk backend, or a delayed network plug-in.",
        "actions": ["Check CPU, memory and disk I/O load on the compute host at that time",
                    "Check the image backend and the _base image cache for slow transfers or a cold cache",
                    "Check Neutron / Open vSwitch agent logs for a delayed VIF plug (network-vif-plugged)",
                    "Compare with other instances created in the same minute to see if the host or only this VM is affected"],
    },
    "slow_build": {
        "title": "Slow instance build (scheduling / claim / network allocation)",
        "events": set(), "weight": 0.8, "severity": "medium",
        "explanation": "The overall build of the instance took longer than any normal instance, although the "
                       "hypervisor spawn itself was not the slow part: the delay is in scheduling, the resource claim, "
                       "image fetch or network allocation.",
        "impact": "Slow VM provisioning and a higher risk of build timeouts under load.",
        "actions": ["Check nova-scheduler and conductor latency", "Check Neutron port creation time for the instance",
                    "Check image download time from Glance"],
    },
    "slow_teardown": {
        "title": "Slow instance teardown",
        "events": set(), "weight": 0.7, "severity": "low",
        "explanation": "Destroying the instance or deallocating its network took longer than any normal instance.",
        "impact": "Resources (compute, ports, IPs) are held longer than needed.",
        "actions": ["Check disk I/O on the compute host during deletion", "Check Neutron port deallocation time"],
    },
    "incomplete_spawn": {
        "title": "Instance did not finish spawning",
        "events": set(), "weight": 0.9, "severity": "high",
        "explanation": "The instance started creating its image but never logged 'Instance spawned successfully'.",
        "impact": "The VM may be stuck in BUILD or have failed to start.",
        "actions": ["Look for errors from nova-compute and libvirt for this instance",
                    "Check whether the compute host or the build request was interrupted"],
    },
    "abrupt_termination": {
        "title": "Session ended abruptly",
        "events": set(), "weight": 0.7, "severity": "medium",
        "explanation": "The instance has far fewer log messages than normal instances.",
        "impact": "Log collection may be incomplete, or the instance lifecycle was cut short.",
        "actions": ["Verify log collection covered the whole time window", "Check what happened to the instance after the last message"],
    },
    "error_messages": {
        "title": "Error-level messages in the instance lifecycle",
        "events": set(), "weight": 1.0, "severity": "high",
        "explanation": "Error-level messages were logged for this instance.",
        "impact": "The instance operation (build, delete, migrate...) probably failed.",
        "actions": ["Read the error messages and the surrounding nova-compute log lines",
                    "Check libvirt, Neutron and Glance for the matching failure"],
    },
    "warning_messages": {
        "title": "Warning-level messages in the instance lifecycle",
        "events": set(), "weight": 0.5, "severity": "low",
        "explanation": "Warning-level messages were logged for this instance.",
        "impact": "Possible degradation; often harmless on its own.",
        "actions": ["Review the warning messages for this instance"],
    },
    "rare_pattern": {
        "title": "Unusual event pattern (model-detected)",
        "events": set(), "weight": 0.4, "severity": "medium",
        "explanation": "No timing or error evidence explains it, but the ML model and the retrieved similar historical "
                       "instances indicate this pattern is characteristic of anomalous instances.",
        "impact": "Unknown - needs manual review of the instance timeline.",
        "actions": ["Review the full timeline of the instance", "Compare with the retrieved similar anomalous instances"],
    },
}


class OpenStackProfile(MinedProfile):
    name = "openstack"
    title = "OpenStack"
    persona = "OpenStack (Nova) cloud operations engineer"
    session_noun = "instance"
    session_noun_plural = "instances"
    node_noun = "host"
    max_seq_len = 40
    # Only the service-reported timings. Session duration and the longest gap depend on how long a VM was kept
    # alive (set by the user or test script), so they are shown in the Gold layer but not given to the model.
    extra_features = TIMING_FEATURES
    session_re = re.compile(r"\b" + UUID + r"\b")
    formats = {"openstack_raw": "Raw OpenStack (nova) log lines grouped into VM-instance sessions by [instance: <uuid>]"}
    glossary = ("A session is one VM instance (UUID): the nova log messages tagged [instance: <uuid>], in time order. "
                "Events are message templates learned from the logs (E<n>). The anomalies in the labelled dataset "
                "are timing anomalies: an instance whose 'Took N seconds to spawn/build' is far above every normal "
                "instance while its event order is normal. Timings are numeric features the detector also uses.")
    normal_description = ("A normal OpenStack instance claims resources, creates its image, spawns, runs, and is "
                          "terminated and cleaned up, with spawn and build times close to the usual ~20 seconds.")
    sample_files = ("openstack_raw_sample.log",)

    # ── identification ────────────────────────────────────────────────
    def detect(self, lines):
        hits = 0
        for ln in lines:
            m = LINE_RE.match(ln.strip())
            if m and (OPENSTACK_COMPONENT.match(m["component"]) or "[req-" in ln):
                hits += 1
        return {"openstack_raw": hits / len(lines)}

    # ── Bronze ────────────────────────────────────────────────────────
    def clean(self, text, fmt, **kw):
        lines = pd.Series(text.splitlines(), dtype="object")
        n_in = len(lines)
        df = pd.DataFrame({"line_no": range(1, n_in + 1),
                           "raw": lines.str.replace(WS_RE, " ", regex=True).str.strip()})
        blank = df.raw.eq("") | df.raw.str.startswith("#")
        df = df[~blank]
        dup = df.raw.duplicated()
        n_dup = int(dup.sum())
        df = df[~dup]

        parts = df.raw.str.extract(LINE_RE)
        ok = parts.date.notna()
        noise_examples = df.loc[~ok, "raw"].head(5).tolist()
        n_noise = int((~ok).sum())
        df, parts = df[ok].copy(), parts[ok]

        rest = parts.rest.str.replace(r"^\[req-[^\]]*\]\s*", "", regex=True)
        df["block_id"] = rest.str.extract(INSTANCE_RE)[0]
        content = (rest.str.replace(INSTANCE_RE, "", regex=True).str.strip()
                   .str.replace(r"^\[-\]\s*", "", regex=True))
        df["service"] = parts.service.fillna("").str.extract(r"^([\w\-]+?)\.log")[0].fillna("unknown")
        df["timestamp"] = to_datetime(parts.date + " " + parts.time.where(parts.time.str.contains(r"\."),
                                                                          parts.time + ".0"), "%Y-%m-%d %H:%M:%S.%f")
        df["pid"] = parts.pid
        df["level"] = parts.level
        df["component"] = parts.component
        df["content"] = content
        unassigned = int(df.block_id.isna().sum())
        df = df[["line_no", "raw", "service", "timestamp", "pid", "level", "component", "block_id", "content"]]
        return df.reset_index(drop=True), {
            "input_lines": n_in, "blank_or_comment": int(blank.sum()), "duplicates_removed": n_dup,
            "noise_removed": n_noise, "noise_examples": noise_examples, "clean_records": len(df),
            "lines_without_instance": unassigned, "instances": int(df.block_id.nunique()),
            "steps": ["whitespace normalisation", "blank/comment removal", "exact-duplicate removal",
                      "field parsing (service, time, pid, level, component, request id, instance id, message)",
                      "timestamp normalisation", "noise removal (lines that are not OpenStack log records)",
                      "instance id extraction (lines without [instance: <uuid>] stay in Bronze only)"],
        }

    # ── Silver ────────────────────────────────────────────────────────
    def build_silver(self, clean, fmt):
        df = clean[clean.block_id.notna()].copy()
        df["event_id"] = self.event_for(df.content)
        unmatched = int(df.event_id.isna().sum())
        df = df.dropna(subset=["event_id"])
        df = df.sort_values(["timestamp", "line_no"], kind="stable")
        df["position"] = df.groupby("block_id").cumcount()
        took = df.content.str.extract(TOOK_RE)
        df["value"] = pd.to_numeric(took[0], errors="coerce")
        df["metric"] = took[1].str.lower()
        df["ips"] = [[] for _ in range(len(df))]
        silver = df[["block_id", "position", "line_no", "timestamp", "level", "component", "event_id", "ips",
                     "metric", "value"]].reset_index(drop=True)
        return self.enrich_events(silver), unmatched

    def add_session_features(self, gold, silver, clean, fmt):
        took = silver.dropna(subset=["metric"])
        piv = (took.pivot_table(index="block_id", columns="metric", values="value", aggfunc="max")
               if len(took) else pd.DataFrame())
        for m in METRICS:
            gold[f"took_{m}"] = piv[m].reindex(gold.block_id).values if m in piv else np.nan
        return gold

    # ── RCA ───────────────────────────────────────────────────────────
    def build_root_causes(self):
        causes = {k: {**v} for k, v in CAUSES.items()}
        causes["error_messages"]["events"] = self.error_events
        causes["warning_messages"]["events"] = self.warning_events
        return causes

    def structural_scores(self, counts, present, row, stats):
        scores, evidence = {}, []
        num = stats.get("numeric", {})
        for metric, cause, label in (("spawn", "slow_spawn", "Spawn"), ("build", "slow_build", "Build"),
                                     ("destroy", "slow_teardown", "Destroy"),
                                     ("deallocate", "slow_teardown", "Network deallocation")):
            value = row.get(f"took_{metric}", np.nan)
            ref = num.get(f"took_{metric}")
            if ref is None or value is None or pd.isna(value):
                continue
            if value > ref["threshold"]:
                strength = min(1.0, 0.5 + 0.25 * (value / ref["median"] - 1))
                if cause == "slow_build" and "slow_spawn" in scores:
                    continue                       # the spawn is the slow part; do not blame the wider build
                scores[cause] = max(scores.get(cause, 0), CAUSES[cause]["weight"] * (0.5 + 0.5 * strength))
                evidence.append(f"{label} took {value:.1f} s; normal training instances: median {ref['median']:.1f} s, "
                                f"maximum {ref['max_normal']:.1f} s (flagged above {ref['threshold']:.1f} s)")
        created = self.events_matching("creating image")
        spawned = self.events_matching("spawned successfully")
        if created & present and spawned and not spawned & present:
            scores["incomplete_spawn"] = CAUSES["incomplete_spawn"]["weight"]
            evidence.append("Image creation started but 'Instance spawned successfully' was never logged")
        p05 = stats.get("normal_length", {}).get("p05")
        if p05 and row.n_events < 0.5 * p05:
            scores["abrupt_termination"] = CAUSES["abrupt_termination"]["weight"] * 0.9
            evidence.append(f"Only {row.n_events} events vs ≥{int(p05)} in 95% of normal instances")
        return scores, evidence

    def normal_structural_evidence(self, counts, row, stats):
        out = []
        num = stats.get("numeric", {})
        for metric, label in (("spawn", "Spawn"), ("build", "Build")):
            value, ref = row.get(f"took_{metric}", np.nan), num.get(f"took_{metric}")
            if ref is not None and value is not None and not pd.isna(value):
                out.append(f"{label} took {value:.1f} s, within the normal range (median {ref['median']:.1f} s, "
                           f"maximum {ref['max_normal']:.1f} s)")
        return out
