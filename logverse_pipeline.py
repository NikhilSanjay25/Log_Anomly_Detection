"""
LogVerse AI Platform — Core Data & Medallion Pipeline Engine
============================================================
Implements Medallion Lakehouse Architecture (Bronze -> Silver -> Gold)
and AI Catalog Metadata Management for Enterprise Log Data (HDFS / K8s / Docker).
"""

import os
import re
import csv
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 1. HDFS TEMPLATE DEFINITIONS & ERROR EVENT REGEX
# ─────────────────────────────────────────────────────────────
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

ERROR_EVENTS = {"E4", "E7", "E8", "E10", "E11", "E12", "E14", "E17", "E20", "E24", "E29"}
COMPILED_TEMPLATES = [(eid, re.compile(pat, re.IGNORECASE), tmpl) for eid, pat, tmpl in HDFS_TEMPLATES]
BLOCK_RE = re.compile(r"blk_[-\d]+")
LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<component>[\w.$]+):\s+(?P<content>.+)$"
)


class MedallionPipeline:
    """
    Manages log processing through Bronze, Silver, and Gold tiers.
    Also builds line-level metadata for the AI Catalog.
    """

    def __init__(self, source_type="HDFS"):
        self.source_type = source_type
        self.bronze_data = []
        self.silver_data = pd.DataFrame()
        self.gold_data = pd.DataFrame()
        self.ai_catalog = {}

    def process_raw_logs(self, log_content: str, source_name="upload.log"):
        """
        Executes the Medallion Pipeline on input text log data.
        """
        start_time = time.time()
        lines = log_content.strip().splitlines()

        # -------------------------------------------------------------
        # 1. BRONZE LAYER: Raw Data Ingestion (Immutable Record)
        # -------------------------------------------------------------
        self.bronze_data = []
        for idx, line in enumerate(lines, 1):
            if not line.strip():
                continue
            self.bronze_data.append({
                "LineId": idx,
                "Source": source_name,
                "RawText": line,
                "IngestedAt": datetime.now().isoformat()
            })

        # -------------------------------------------------------------
        # 2. SILVER LAYER: Parsing, Normalization, Event Extraction
        # -------------------------------------------------------------
        parsed_records = []
        block_sessions = defaultdict(list)
        template_counts = defaultdict(int)

        for rec in self.bronze_data:
            line = rec["RawText"]
            m = LOG_LINE_RE.match(line)
            if m:
                date_str = m.group("date")
                time_str = m.group("time")
                pid = m.group("pid")
                level = m.group("level")
                component = m.group("component")
                content = m.group("content")
            else:
                date_str, time_str, pid, level, component = "", "", "", "INFO", "Unknown"
                content = line

            # Match event template
            event_id = "E0"
            event_template = content[:80]
            for eid, pat, tmpl in COMPILED_TEMPLATES:
                if pat.search(content):
                    event_id = eid
                    event_template = tmpl
                    break

            template_counts[event_id] += 1
            blocks = BLOCK_RE.findall(content)
            blk_id = blocks[0] if blocks else "blk_global"

            parsed_record = {
                "LineId": rec["LineId"],
                "Date": date_str,
                "Time": time_str,
                "Pid": pid,
                "Level": level,
                "Component": component,
                "Content": content,
                "EventId": event_id,
                "EventTemplate": event_template,
                "BlockId": blk_id,
                "IsError": event_id in ERROR_EVENTS or level in ["ERROR", "FATAL", "WARN"]
            }
            parsed_records.append(parsed_record)
            block_sessions[blk_id].append(parsed_record)

        self.silver_data = pd.DataFrame(parsed_records)

        # -------------------------------------------------------------
        # 3. GOLD LAYER: Session Aggregations & AI Feature Sets
        # -------------------------------------------------------------
        gold_rows = []
        for blk, recs in block_sessions.items():
            if blk == "blk_global" and len(block_sessions) > 1:
                continue

            event_ids = [r["EventId"] for r in recs]
            error_count = sum(1 for e in event_ids if e in ERROR_EVENTS)
            levels = [r["Level"] for r in recs]

            gold_rows.append({
                "BlockId": blk,
                "EventSequence": " → ".join(event_ids),
                "EventList": event_ids,
                "TotalEvents": len(event_ids),
                "ErrorCount": error_count,
                "HasErrorEvent": error_count > 0,
                "Components": list(set(r["Component"] for r in recs)),
                "SampleLogLine": recs[0]["Content"]
            })

        self.gold_data = pd.DataFrame(gold_rows)

        # -------------------------------------------------------------
        # 4. AI CATALOG METADATA REGISTRATION
        # -------------------------------------------------------------
        self.ai_catalog = {
            "source": source_name,
            "source_type": self.source_type,
            "bronze_records_count": len(self.bronze_data),
            "silver_records_count": len(self.silver_data),
            "gold_sessions_count": len(self.gold_data),
            "total_anomalous_blocks": int((self.gold_data["ErrorCount"] > 0).sum()) if not self.gold_data.empty else 0,
            "unique_events_found": sorted(list(template_counts.keys())),
            "processing_time_sec": round(time.time() - start_time, 4),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return self.bronze_data, self.silver_data, self.gold_data, self.ai_catalog


def generate_sample_hdfs_log():
    """Generates sample HDFS log lines for testing/demonstration."""
    return """081109 203518 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.99:54106 dest: /10.250.19.99:50010
081109 203519 148 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/randtxt/_temporary/_task_200811092035_0001_m_000000_0/part-00000. blk_-1608999687919862906
081109 203520 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
081109 203521 148 WARN dfs.DataNode$DataXceiver: Got exception while serving blk_-1608999687919862906 to /10.250.19.99:54106
081109 203522 148 ERROR dfs.DataNode$DataXceiver: writeBlock blk_-1608999687919862906 received exception java.io.IOException: Connection reset by peer
081109 203523 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 Exception java.io.EOFException
081109 203524 148 WARN dfs.FSNamesystem: PendingReplicationMonitor timed out block blk_-1608999687919862906
081109 203600 200 INFO dfs.DataNode$DataXceiver: Receiving block blk_9876543210123456789 src: /10.250.10.12:50010 dest: /10.250.10.13:50010
081109 203601 200 INFO dfs.DataNode$DataXceiver: Received block blk_9876543210123456789 of size 67108864 from /10.250.10.12
081109 203602 200 INFO dfs.FSNamesystem: Verification succeeded for blk_9876543210123456789"""
