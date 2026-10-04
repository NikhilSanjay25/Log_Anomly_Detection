"""Root-cause knowledge base for HDFS block sessions.

Each cause has: a trigger (events / structural checks evaluated by the RCA agent), an explanation,
impact, severity and recommended actions. The Insight agent uses this as grounding for the SLM
and as the deterministic fallback when the SLM is unavailable.
"""

ROOT_CAUSES = {
    "write_pipeline_failure": {
        "title": "Write pipeline failure",
        "events": {"E7", "E8", "E10", "E12", "E14"},
        "weight": 1.0,
        "explanation": "A DataNode in the replication write pipeline raised an exception (writeBlock / "
                       "PacketResponder / receiveBlock / mirror write), so the block was not written to all replicas.",
        "impact": "Block may be under-replicated or the client write failed; repeated failures risk data loss "
                  "if remaining replicas are lost.",
        "severity": "high",
        "actions": ["Check DataNode logs on the nodes involved for IOExceptions, disk errors or 'Connection reset by peer'",
                    "Verify network health between the pipeline DataNodes (packet loss, NIC errors)",
                    "Run `hdfs fsck <path> -blocks -locations` to confirm replication of affected files",
                    "Check disk health / free space on the involved DataNodes (dfs.datanode.du.reserved)"],
    },
    "replication_failure": {
        "title": "Replication failure / timeout",
        "events": {"E17", "E29"},
        "weight": 1.0,
        "explanation": "The NameNode tried to re-replicate the block but the transfer failed or the pending "
                       "replication timed out.",
        "impact": "Block stays under-replicated; durability is reduced until replication succeeds.",
        "severity": "high",
        "actions": ["Inspect the target DataNodes for availability and transfer errors",
                    "Check `dfs.namenode.replication.pending.timeout-sec` and NameNode replication queue size",
                    "Run `hdfs dfsadmin -report` to find dead or decommissioning DataNodes",
                    "Trigger re-replication with `hdfs fsck / -list-corruptfileblocks` follow-up"],
    },
    "delete_inconsistency": {
        "title": "Delete of unknown block (metadata inconsistency)",
        "events": {"E20"},
        "weight": 0.9,
        "explanation": "A DataNode was asked to delete a block that is not in its volume map - NameNode and "
                       "DataNode metadata disagree about where replicas live.",
        "impact": "Usually harmless for data, but indicates stale block reports or a disk that was replaced/remounted.",
        "severity": "medium",
        "actions": ["Force a fresh block report from the DataNode (`hdfs dfsadmin -triggerBlockReport <dn>`)",
                    "Check whether a data directory was recently unmounted, replaced or failed",
                    "Compare NameNode block locations with the DataNode's actual block files"],
    },
    "orphan_block": {
        "title": "Orphan block (belongs to no file)",
        "events": {"E28", "E24"},
        "weight": 0.9,
        "explanation": "Replicas were reported for a block that no longer belongs to any file - typically the "
                       "file was deleted or its lease expired while the block was still being written.",
        "impact": "Wasted storage and extra block-report churn; can indicate aborted client writes.",
        "severity": "medium",
        "actions": ["Correlate with client/job logs: was the file deleted or the job killed during the write?",
                    "Check lease recovery events on the NameNode for the affected path",
                    "Let the NameNode invalidate the replicas; monitor for recurrence on the same clients"],
    },
    "duplicate_report": {
        "title": "Duplicate / redundant block reports",
        "events": {"E1", "E27"},
        "weight": 0.5,
        "explanation": "The same replica was reported more than once to the NameNode.",
        "impact": "Low direct impact; may hint at DataNode restarts or retrying block reports.",
        "severity": "low",
        "actions": ["Check for DataNode restarts or re-registrations around the session time",
                    "Review block report intervals and NameNode RPC load"],
    },
    "incomplete_write": {
        "title": "Incomplete write lifecycle",
        "events": set(),
        "weight": 0.8,
        "explanation": "The block was allocated / started receiving but not every replica confirmed completion "
                       "(fewer 'Received block' / 'PacketResponder terminating' / 'addStoredBlock' events than expected).",
        "impact": "Write may have been aborted mid-way; the block can end up under-replicated or orphaned.",
        "severity": "medium",
        "actions": ["Check client logs for aborted writes or timeouts",
                    "Check whether a pipeline DataNode went down during the write",
                    "Run `hdfs fsck` on the file to confirm its final replication"],
    },
    "abrupt_termination": {
        "title": "Session ended abruptly",
        "events": set(),
        "weight": 0.7,
        "explanation": "The block session is much shorter than normal sessions and stops right after it started.",
        "impact": "Indicates a write that never progressed - client crash, NameNode allocation without data, or lost logs.",
        "severity": "medium",
        "actions": ["Check whether the client process crashed right after block allocation",
                    "Verify log collection was complete for the time window (missing DataNode logs?)"],
    },
    "re_replication": {
        "title": "Re-replication of an under-replicated block",
        "events": {"E25", "E18", "E16", "E6"},
        "weight": 0.7,
        "explanation": "The NameNode asked DataNodes to copy the block again (ask to replicate → transfer → "
                       "received replica), which happens when a replica was lost or a DataNode went away. "
                       "In training data 66% of sessions with these events were anomalous.",
        "impact": "Durability was temporarily reduced; frequent re-replication points to unstable DataNodes or disks.",
        "severity": "medium",
        "actions": ["Run `hdfs dfsadmin -report` and check for dead, stale or decommissioning DataNodes",
                    "Check the source DataNodes of the replication for disk failures or restarts",
                    "Watch the NameNode under-replicated blocks metric for a growing backlog"],
    },
    "write_recovery": {
        "title": "Write recovery / block re-open",
        "events": {"E13", "E15", "E19"},
        "weight": 0.9,
        "explanation": "The write went through recovery: empty packets, a block file offset change or a re-opened "
                       "block. Every training session with these events was labelled anomalous.",
        "impact": "The client write was interrupted and resumed; the block may be inconsistent until recovery finishes.",
        "severity": "medium",
        "actions": ["Check client logs for write timeouts or retries on the affected files",
                    "Check pipeline DataNodes for slow disks or network stalls around the session time",
                    "Run `hdfs fsck <path> -openforwrite` to find files still stuck in recovery"],
    },
    "rare_pattern": {
        "title": "Rare event ordering (model-detected)",
        "events": set(),
        "weight": 0.4,
        "explanation": "No single error event explains it, but the ML model and the retrieved similar historical "
                       "sessions indicate this event ordering is characteristic of anomalous blocks.",
        "impact": "Unknown - needs manual review of the full block timeline.",
        "severity": "medium",
        "actions": ["Review the full timeline of the block across NameNode and DataNode logs",
                    "Compare with the retrieved similar anomalous sessions for a common pattern"],
    },
}
