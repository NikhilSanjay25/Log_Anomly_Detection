"""
HDFS Log Processing Script
===========================
Converts raw HDFS log file (HDFS.txt) into:
  - anomaly_label.csv
  - Event_occurrence_matrix.csv
  - Event_traces.csv          (full sequence, no deduplication)
  - HDFS.log_templates.csv
  - HDFS.npz                  (full padded sequences, not just count matrix)

Usage:
    python process_hdfs_logs.py --log_file "C:/path/to/HDFS.txt" --output_dir "C:/path/to/output"

Dependencies:
    pip install pandas numpy
"""
import time
"""
HDFS Log Processing Script
===========================
Converts raw HDFS log file (HDFS.txt) into:
  - anomaly_label.csv
  - Event_occurrence_matrix.csv
  - Event_traces.csv          (BlockId, Label, Type, Features, TimeInterval, Latency)
  - HDFS.log_templates.csv
  - HDFS.npz

Usage:
    python process_hdfs_logs.py --log_file "C:/path/to/HDFS.txt" --output_dir "C:/path/to/output"
    python process_hdfs_logs.py --log_file "C:/path/to/HDFS.txt" --output_dir "C:/path/to/output" --label_file "C:/path/to/anomaly_label.csv"

Dependencies:
    pip install pandas numpy
"""

import re
"""
HDFS Log Processing Script
===========================
Converts raw HDFS log file (HDFS.txt) into:
  - anomaly_label.csv
  - Event_occurrence_matrix.csv
  - Event_traces.csv          (BlockId, Label, Type, Features, TimeInterval, Latency)
  - HDFS.log_templates.csv
  - HDFS.npz

Usage:
    python process_hdfs_logs.py --log_file "C:/path/to/HDFS.txt" --output_dir "C:/path/to/output"
    python process_hdfs_logs.py --log_file "C:/path/to/HDFS.txt" --output_dir "C:/path/to/output" --label_file "C:/path/to/anomaly_label.csv"

Dependencies:
    pip install pandas numpy
"""

import re
import os
import csv
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────
# 1.  EXACT EVENT TEMPLATES  (E1–E29)
# ─────────────────────────────────────────────
HDFS_TEMPLATES = [
    ("E1",  r"Adding an already existing block",                                                                                       "Adding an already existing block[*]"),
    ("E2",  r"Verification succeeded for",                                                                                             "Verification succeeded for[*]"),
    ("E3",  r"Served block.+to",                                                                                                       "Served block[*]to[*]"),
    ("E4",  r"Got exception while serving.+to",                                                                                        "Got exception while serving[*]to[*]"),
    ("E5",  r"Receiving block.+src:.+dest:",                                                                                           "Receiving block[*]src:[*]dest:[*]"),
    ("E6",  r"Received block.+src:.+dest:.+of size",                                                                                   "Received block[*]src:[*]dest:[*]of size[*]"),
    ("E7",  r"writeBlock.+received exception",                                                                                         "writeBlock[*]received exception[*]"),
    ("E8",  r"PacketResponder.+for block.+Interrupted",                                                                                "PacketResponder[*]for block[*]Interrupted[*]"),
    ("E9",  r"Received block.+of size.+from",                                                                                          "Received block[*]of size[*]from[*]"),
    ("E10", r"PacketResponder.+Exception",                                                                                             "PacketResponder[*]Exception[*]"),
    ("E11", r"PacketResponder.+for block.+terminating",                                                                                "PacketResponder[*]for block[*]terminating[*]"),
    ("E12", r":Exception writing block.+to mirror",                                                                                    "[*]:Exception writing block[*]to mirror[*]"),
    ("E13", r"Receiving empty packet for block",                                                                                       "Receiving empty packet for block[*]"),
    ("E14", r"Exception in receiveBlock for block",                                                                                    "Exception in receiveBlock for block[*]"),
    ("E15", r"Changing block file offset of block.+from.+to.+meta file offset to",                                                    "Changing block file offset of block[*]from[*]to[*]meta file offset to[*]"),
    ("E16", r":Transmitted block.+to",                                                                                                 "[*]:Transmitted block[*]to[*]"),
    ("E17", r":Failed to transfer.+to.+got",                                                                                          "[*]:Failed to transfer[*]to[*]got[*]"),
    ("E18", r"Starting thread to transfer block.+to",                                                                                  "Starting thread to transfer block[*]to[*]"),
    ("E19", r"Reopen Block",                                                                                                           "Reopen Block[*]"),
    ("E20", r"Unexpected error trying to delete block.+BlockInfo not found in volumeMap",                                              "Unexpected error trying to delete block[*]BlockInfo not found in volumeMap[*]"),
    ("E21", r"Deleting block.+file",                                                                                                   "Deleting block[*]file[*]"),
    ("E22", r"BLOCK\* NameSystem.+allocateBlock:",                                                                                     "BLOCK* NameSystem[*]allocateBlock:[*]"),
    ("E23", r"BLOCK\* NameSystem.+delete:.+is added to invalidSet of",                                                                "BLOCK* NameSystem[*]delete:[*]is added to invalidSet of[*]"),
    ("E24", r"BLOCK\* Removing block.+from neededReplications as it does not belong to any file",                                     "BLOCK* Removing block[*]from neededReplications as it does not belong to any file[*]"),
    ("E25", r"BLOCK\* ask.+to replicate.+to",                                                                                         "BLOCK* ask[*]to replicate[*]to[*]"),
    ("E26", r"BLOCK\* NameSystem.+addStoredBlock: blockMap updated:.+is added to.+size",                                              "BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]"),
    ("E27", r"BLOCK\* NameSystem.+addStoredBlock: Redundant addStoredBlock request received for.+on.+size",                           "BLOCK* NameSystem[*]addStoredBlock: Redundant addStoredBlock request received for[*]on[*]size[*]"),
    ("E28", r"BLOCK\* NameSystem.+addStoredBlock: addStoredBlock request received for.+on.+size.+But it does not belong to any file", "BLOCK* NameSystem[*]addStoredBlock: addStoredBlock request received for[*]on[*]size[*]But it does not belong to any file[*]"),
    ("E29", r"PendingReplicationMonitor timed out block",                                                                              "PendingReplicationMonitor timed out block[*]"),
]

# Error/anomaly-related event IDs (used for Type count)
ERROR_EVENTS = {"E4", "E7", "E8", "E10", "E11", "E12", "E14", "E17", "E20", "E24", "E29"}

COMPILED_TEMPLATES = [(eid, re.compile(pat, re.IGNORECASE), tmpl) for eid, pat, tmpl in HDFS_TEMPLATES]
BLOCK_RE    = re.compile(r"blk_[-\d]+")
LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<component>[\w.$]+):\s+(?P<content>.+)$"
)

# ─────────────────────────────────────────────
# 2.  TEMPLATE MATCHING
# ─────────────────────────────────────────────

def match_template(content):
    for eid, pattern, tmpl in COMPILED_TEMPLATES:
        if pattern.search(content):
            return eid, tmpl
    return "E0", content[:80]

# ─────────────────────────────────────────────
# 3.  PARSING
# ─────────────────────────────────────────────

def parse_timestamp(date_str, time_str):
    """Convert HDFS date/time strings to seconds since epoch."""
    try:
        dt = datetime.strptime(date_str + time_str, "%y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None

def parse_logs(log_path):
    records = []
    print(f"[INFO] Reading: {log_path}")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            m = LOG_LINE_RE.match(line)
            if m:
                content   = m.group("content")
                date_str  = m.group("date")
                time_str  = m.group("time")
                timestamp = parse_timestamp(date_str, time_str)
            else:
                content   = line
                date_str  = ""
                time_str  = ""
                timestamp = None
            records.append({
                "LineId":    lineno,
                "Date":      date_str,
                "Time":      time_str,
                "Timestamp": timestamp,
                "Pid":       m.group("pid")       if m else "",
                "Level":     m.group("level")     if m else "",
                "Component": m.group("component") if m else "",
                "Content":   content,
            })
    print(f"[INFO] Parsed {len(records):,} log lines.")
    return records

def assign_events(records):
    template_counts = defaultdict(int)
    template_map    = {}
    for r in records:
        eid, tmpl = match_template(r["Content"])
        r["EventId"]       = eid
        r["EventTemplate"] = tmpl
        template_counts[eid] += 1
        template_map[eid]    = tmpl
    unmatched = template_counts.get("E0", 0)
    if unmatched:
        print(f"[WARN] {unmatched:,} lines did not match any template (assigned E0).")
    templates = {
        eid: {"EventId": eid, "EventTemplate": tmpl, "Occurrences": template_counts[eid]}
        for eid, tmpl in template_map.items()
    }
    return records, templates

# ─────────────────────────────────────────────
# 4.  SESSION BUILDING  — full sequences + timestamps
# ─────────────────────────────────────────────

def build_session_traces(records):
    """
    Group log lines by Block ID.
    Each session stores: list of (EventId, timestamp) tuples.
    """
    session_data = defaultdict(list)   # blk -> [(eid, ts), ...]
    for r in records:
        for blk in BLOCK_RE.findall(r["Content"]):
            session_data[blk].append((r["EventId"], r["Timestamp"]))
    print(f"[INFO] Found {len(session_data):,} unique block sessions.")
    return session_data

# ─────────────────────────────────────────────
# 5.  FEATURE EXTRACTION per session
# ─────────────────────────────────────────────

def compute_session_features(session_data, label_map):
    """
    Returns a list of dicts, one per session, with columns:
    BlockId, Label, Type, Features, TimeInterval, Latency
    """
    rows = []
    for blk, events in sorted(session_data.items()):
        event_ids  = [e for e, _ in events]
        timestamps = [t for _, t in events]

        # --- Label (Success / Fail) ---
        label_int = label_map.get(blk, 0)
        label_str = "Fail" if label_int == 1 else "Success"

        # --- Type: count of error/anomaly events in session ---
        error_count = sum(1 for e in event_ids if e in ERROR_EVENTS)
        type_val    = error_count if error_count > 0 else ""

        # --- Features: event sequence as [E5,E22,E5,...] ---
        features = "[" + ",".join(event_ids) + "]"

        # --- TimeInterval: gaps (in seconds) between consecutive events ---
        valid_ts = [t for t in timestamps if t is not None]
        if len(valid_ts) >= 2:
            gaps = []
            for i in range(1, len(timestamps)):
                t_prev = timestamps[i - 1]
                t_curr = timestamps[i]
                if t_prev is not None and t_curr is not None:
                    gap = round(t_curr - t_prev, 1)
                    gaps.append(gap)
                else:
                    gaps.append(0.0)
            time_interval = "[" + ", ".join(str(g) for g in gaps) + "]"
            latency = round(valid_ts[-1] - valid_ts[0])
        else:
            time_interval = "[0.0]"
            latency       = 0

        rows.append({
            "BlockId":      blk,
            "Label":        label_str,
            "Type":         type_val,
            "Features":     features,
            "TimeInterval": time_interval,
            "Latency":      latency,
        })

    return rows

# ─────────────────────────────────────────────
# 6.  OCCURRENCE MATRIX
# ─────────────────────────────────────────────

def build_occurrence_matrix(session_data, all_event_ids):
    def sort_key(e):
        num = e[1:]
        return int(num) if num.isdigit() else 9999

    event_list   = sorted(all_event_ids, key=sort_key)
    session_list = sorted(session_data.keys())
    event_index  = {e: i for i, e in enumerate(event_list)}
    matrix       = np.zeros((len(session_list), len(event_list)), dtype=np.int32)

    for row_idx, blk in enumerate(session_list):
        for eid, _ in session_data[blk]:
            col = event_index.get(eid)
            if col is not None:
                matrix[row_idx, col] += 1

    return session_list, event_list, matrix

# ─────────────────────────────────────────────
# 7.  NPZ — padded full sequences
# ─────────────────────────────────────────────

def build_sequences(session_data, session_list, event_list):
    """
    Build x_data and y_data matching the expected NPZ format:
      x_data : (n_sessions,) object array — each element is a list of int event IDs
      y_data : (n_sessions,) int64        — 0=Normal, 1=Anomaly (filled later)
    """
    event_to_int = {e: i + 1 for i, e in enumerate(event_list)}  # 1-indexed, 0 unused

    x_data = np.empty(len(session_list), dtype=object)
    for row_idx, blk in enumerate(session_list):
        x_data[row_idx] = [event_to_int.get(eid, 0) for eid, _ in session_data[blk]]

    print(f"[INFO] NPZ x_data shape: {x_data.shape}  (object array of variable-length lists)")
    return x_data

# ─────────────────────────────────────────────
# 8.  OUTPUT WRITERS
# ─────────────────────────────────────────────

def write_log_templates(templates, out_dir):
    path = os.path.join(out_dir, "HDFS.log_templates.csv")
    rows = sorted(templates.values(), key=lambda x: int(x["EventId"][1:]) if x["EventId"][1:].isdigit() else 9999)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["EventId", "EventTemplate", "Occurrences"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {path}  ({len(rows)} templates)")

def write_event_traces(trace_rows, out_dir):
    path = os.path.join(out_dir, "Event_traces.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["BlockId", "Label", "Type", "Features", "TimeInterval", "Latency"])
        w.writeheader()
        w.writerows(trace_rows)
    size_kb = os.path.getsize(path) / 1024
    print(f"[OUT] {path}  ({len(trace_rows):,} sessions, {size_kb:.0f} KB)")

def write_occurrence_matrix(session_list, event_list, matrix, out_dir):
    path = os.path.join(out_dir, "Event_occurrence_matrix.csv")
    df   = pd.DataFrame(matrix, index=session_list, columns=event_list)
    df.index.name = "BlockId"
    df.to_csv(path)
    print(f"[OUT] {path}  (shape {matrix.shape})")

def write_npz(session_list, event_list, x_data, label_map, out_dir):
    """
    Save in the exact expected format:
      x_data : (n_sessions,) object  -- variable-length list of int event IDs per session
      y_data : (n_sessions,) int64   -- 0=Normal, 1=Anomaly
    """
    path   = os.path.join(out_dir, "HDFS.npz")
    y_data = np.array([label_map.get(blk, 0) for blk in session_list], dtype=np.int64)
    np.savez(path, x_data=x_data, y_data=y_data)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[OUT] {path}  ({size_mb:.1f} MB)  x_data={x_data.shape} y_data={y_data.shape}")

def write_anomaly_label(session_list, label_map, out_dir):
    path = os.path.join(out_dir, "anomaly_label.csv")
    rows = [{"BlockId": blk, "Label": label_map.get(blk, 0)} for blk in session_list]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["BlockId", "Label"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {path}  ({len(rows):,} sessions)")

# ─────────────────────────────────────────────
# 9.  LABEL LOADER
# ─────────────────────────────────────────────

def load_label_map(label_file):
    """Returns dict: block_id -> 0 (Normal) or 1 (Anomaly)"""
    if not label_file or not os.path.isfile(label_file):
        print("[WARN] No label file provided – all sessions will be labelled Normal (0).")
        return {}
    gt = pd.read_csv(label_file)
    gt.columns = [c.strip().lower() for c in gt.columns]
    id_col    = next(c for c in gt.columns if "block" in c or "id" in c)
    label_col = next(c for c in gt.columns if "label" in c or "anomal" in c)
    label_map = dict(zip(gt[id_col].astype(str), gt[label_col].astype(int)))
    n_anomaly = sum(1 for v in label_map.values() if v == 1)
    print(f"[INFO] Loaded {len(label_map):,} labels  ({n_anomaly:,} anomalies).")
    return label_map

# ─────────────────────────────────────────────
# 10.  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process raw HDFS logs into structured dataset files.")
    parser.add_argument("--log_file",   required=True, help="Path to raw HDFS log file  e.g. C:/data/HDFS.txt")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files  e.g. C:/data/output")
    parser.add_argument("--label_file", default="",   help="(Optional) Path to ground-truth anomaly labels CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse & assign events
    records                = parse_logs(args.log_file)
    records, templates     = assign_events(records)

    # Load labels
    label_map = load_label_map(args.label_file)

    # Build sessions
    session_data  = build_session_traces(records)
    all_event_ids = set(r["EventId"] for r in records)

    # Compute per-session features (for Event_traces)
    trace_rows = compute_session_features(session_data, label_map)

    # Build matrix & sequences
    session_list, event_list, matrix = build_occurrence_matrix(session_data, all_event_ids)
    x_data = build_sequences(session_data, session_list, event_list)

    # Write all outputs
    write_log_templates    (templates,                                    args.output_dir)
    write_event_traces     (trace_rows,                                   args.output_dir)
    write_occurrence_matrix(session_list, event_list, matrix,             args.output_dir)
    write_npz              (session_list, event_list, x_data, label_map,  args.output_dir)
    write_anomaly_label    (session_list, label_map,                      args.output_dir)

    print("\n Done! Output saved to:", os.path.abspath(args.output_dir))
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("[INFO] Total processing time: seconds"+str(end - start))