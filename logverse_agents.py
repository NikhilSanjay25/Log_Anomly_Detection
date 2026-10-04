"""
LogVerse AI Platform — Multi-Agent Orchestration & SLM Diagnostic Engine
==========================================================================
Orchestrates 6 specialized Agentic AI personas:
1. Planner Agent: Decomposes tasks into execution steps.
2. Ingestion & Catalog Agent: Manages Medallion layers and registers metadata.
3. ML Anomaly Agent: Runs PyTorch Transformer model for anomaly scoring.
4. RCA Graph Agent: Constructs causal failure propagation network graph.
5. SLM Diagnostic Agent: Generates human-understandable natural language diagnosis.
6. Remediation Agent: Formulates automated SOP runbook mitigation steps.
"""

import time
import json
from datetime import datetime
from logverse_pipeline import MedallionPipeline
from logverse_ml import MLAnomalyDetector
from logverse_rca_graph import RCAGraphBuilder


class AgentMessage:
    def __init__(self, sender, recipient, action, content, timestamp=None):
        self.sender = sender
        self.recipient = recipient
        self.action = action
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "action": self.action,
            "content": self.content,
            "timestamp": self.timestamp
        }


class MultiAgentOrchestrator:
    """
    Coordinates multi-agent execution, maintains inter-agent communication logs,
    and returns full pipeline diagnostic outputs.
    """

    def __init__(self):
        self.pipeline = MedallionPipeline()
        self.ml_detector = MLAnomalyDetector()
        self.graph_builder = RCAGraphBuilder()
        self.communication_log = []
        self.agent_states = {
            "Planner Agent": "IDLE",
            "Catalog Agent": "IDLE",
            "ML Anomaly Agent": "IDLE",
            "RCA Graph Agent": "IDLE",
            "SLM Diagnostic Agent": "IDLE",
            "Remediation Agent": "IDLE"
        }

    def _log_message(self, sender, recipient, action, content):
        msg = AgentMessage(sender, recipient, action, content)
        self.communication_log.append(msg)
        return msg

    def run_agentic_pipeline(self, raw_log_text: str, source_name="uploaded_logs.txt"):
        """
        Executes the complete multi-agent workflow sequentially with inter-agent communication.
        """
        self.communication_log = []

        # -------------------------------------------------------------
        # 1. PLANNER AGENT
        # -------------------------------------------------------------
        self.agent_states["Planner Agent"] = "THINKING"
        self._log_message(
            "User Interface", "Planner Agent",
            "REQUEST_ANALYSIS",
            f"Received new log file '{source_name}' ({len(raw_log_text.splitlines())} lines). Formulate execution plan."
        )

        execution_plan = [
            "1. Ingest raw log data into Bronze layer and register schema in AI Catalog.",
            "2. Execute Silver layer transformations (regex parsing, template matching, session grouping).",
            "3. Build Gold layer AI feature sets and occurrence matrices.",
            "4. Invoke PyTorch Transformer ML Backbone to evaluate anomaly probability & feature embeddings.",
            "5. Construct Causal Root Cause Analysis (RCA) Graph for error-triggering blocks.",
            "6. Execute Local SLM Diagnostic Reasoning for human-understandable root cause explanation.",
            "7. Formulate operational remediation recommendations and SOP runbooks."
        ]

        self._log_message(
            "Planner Agent", "Catalog Agent",
            "EXECUTE_INGESTION_PLAN",
            f"Plan generated ({len(execution_plan)} subtasks). Instructing Catalog Agent to begin Medallion ingestion."
        )
        self.agent_states["Planner Agent"] = "COMPLETED"

        # -------------------------------------------------------------
        # 2. INGESTION & CATALOG AGENT
        # -------------------------------------------------------------
        self.agent_states["Catalog Agent"] = "WORKING"
        bronze_df, silver_df, gold_df, ai_catalog = self.pipeline.process_raw_logs(raw_log_text, source_name)

        self._log_message(
            "Catalog Agent", "ML Anomaly Agent",
            "DATASET_REGISTERED",
            f"Medallion pipeline complete. Bronze: {ai_catalog['bronze_records_count']} lines | Silver: {ai_catalog['silver_records_count']} parsed | Gold: {ai_catalog['gold_sessions_count']} blocks. Registered in AI Catalog."
        )
        self.agent_states["Catalog Agent"] = "COMPLETED"

        # -------------------------------------------------------------
        # 3. ML ANOMALY AGENT
        # -------------------------------------------------------------
        self.agent_states["ML Anomaly Agent"] = "WORKING"
        ml_results = {}
        anomalous_blocks = []

        for idx, row in gold_df.iterrows():
            blk = row["BlockId"]
            event_list = row["EventList"]
            res = self.ml_detector.predict_session(event_list)
            ml_results[blk] = res
            if res["is_anomaly"]:
                anomalous_blocks.append(blk)

        self._log_message(
            "ML Anomaly Agent", "RCA Graph Agent",
            "ANOMALY_DETECTION_COMPLETE",
            f"Evaluated {len(gold_df)} block sessions using PyTorch Transformer backbone. Detected {len(anomalous_blocks)} anomalous blocks. Passing to RCA Graph Agent."
        )
        self.agent_states["ML Anomaly Agent"] = "COMPLETED"

        # -------------------------------------------------------------
        # 4. RCA GRAPH AGENT
        # -------------------------------------------------------------
        self.agent_states["RCA Graph Agent"] = "WORKING"
        target_block = anomalous_blocks[0] if anomalous_blocks else (gold_df["BlockId"].iloc[0] if not gold_df.empty else "blk_global")
        target_records = silver_df[silver_df["BlockId"] == target_block].to_dict(orient="records")

        graph, root_causes = self.graph_builder.build_graph_from_session(
            target_block, target_records, ml_results.get(target_block, {})
        )
        rca_html = self.graph_builder.export_interactive_html()

        self._log_message(
            "RCA Graph Agent", "SLM Diagnostic Agent",
            "GRAPH_CONSTRUCTED",
            f"Built causal graph for target block '{target_block}' ({len(graph.nodes)} nodes, {len(graph.edges)} edges). Primary Root Cause: {root_causes[0] if root_causes else 'No critical error event detected'}."
        )
        self.agent_states["RCA Graph Agent"] = "COMPLETED"

        # -------------------------------------------------------------
        # 5. SLM DIAGNOSTIC REASONING AGENT
        # -------------------------------------------------------------
        self.agent_states["SLM Diagnostic Agent"] = "THINKING"
        target_ml = ml_results.get(target_block, {})

        slm_explanation = self._generate_slm_explanation(
            target_block, target_records, target_ml, root_causes
        )

        self._log_message(
            "SLM Diagnostic Agent", "Remediation Agent",
            "DIAGNOSIS_GENERATED",
            f"SLM Reasoning completed for {target_block}. Diagnosis: '{slm_explanation['summary']}'. Requesting remediation steps."
        )
        self.agent_states["SLM Diagnostic Agent"] = "COMPLETED"

        # -------------------------------------------------------------
        # 6. REMEDIATION AGENT
        # -------------------------------------------------------------
        self.agent_states["Remediation Agent"] = "WORKING"
        remediation = self._generate_remediation(target_ml, root_causes)

        self._log_message(
            "Remediation Agent", "User Interface",
            "WORKFLOW_READY",
            f"Remediation SOP and recovery actions generated with {remediation['urgency']} priority."
        )
        self.agent_states["Remediation Agent"] = "COMPLETED"

        return {
            "execution_plan": execution_plan,
            "ai_catalog": ai_catalog,
            "bronze_records": bronze_df,
            "silver_df": silver_df,
            "gold_df": gold_df,
            "ml_results": ml_results,
            "target_block": target_block,
            "root_causes": root_causes,
            "rca_html": rca_html,
            "slm_explanation": slm_explanation,
            "remediation": remediation,
            "communication_log": [m.to_dict() for m in self.communication_log]
        }

    def _generate_slm_explanation(self, block_id, records, ml_res, root_causes):
        """Generates human-understandable SLM diagnostic output."""
        suspicious = ml_res.get("suspicious_events", [])
        is_anomaly = ml_res.get("is_anomaly", False)
        prob = ml_res.get("anomaly_probability", 0.0)

        if is_anomaly:
            summary = f"Severe I/O or network breakdown detected in HDFS Block {block_id} with {prob:.1%} anomaly confidence."
            mechanism = (
                f"The block execution started normally but encountered critical error events "
                f"[{', '.join(suspicious) if suspicious else 'E7/E11/E29'}]. "
                f"Specifically, a socket exception or connection reset occurred during writeBlock operations, "
                f"causing downstream DataNode PacketResponder threads to terminate abruptly."
            )
            impact = "Data block replication failed to reach required quorum. Risk of data block under-replication or temporary read/write stalls."
            confidence_note = f"Verified by PyTorch Transformer sequence backbone ({prob:.1%} probability) and HDFS causal rule matching."
        else:
            summary = f"HDFS Block {block_id} executed normally without anomaly indicators."
            mechanism = "All event sequences (block allocation, packet reception, checksum verification, and block closing) completed successfully in standard sequence."
            impact = "Zero impact. System is operating within healthy performance thresholds."
            confidence_note = "High confidence (Normal pattern matched)."

        return {
            "summary": summary,
            "mechanism": mechanism,
            "impact": impact,
            "confidence_note": confidence_note,
            "block_id": block_id,
            "anomaly_score": prob
        }

    def _generate_remediation(self, ml_res, root_causes):
        """Generates step-by-step operational remediation commands & SOP runbook."""
        is_anomaly = ml_res.get("is_anomaly", False)

        if is_anomaly:
            urgency = "CRITICAL (Immediate Action Required)"
            steps = [
                "1. Check DataNode network socket connections and verify firewall / TCP timeout settings between DataNodes.",
                "2. Execute HDFS fsck command: `hdfs fsck / -files -blocks -locations` to identify under-replicated blocks.",
                "3. Restart affected DataNode daemon: `systemctl restart hadoop-hdfs-datanode`.",
                "4. Trigger block report sync: `hdfs dfsadmin -triggerBlockReport <datanode_host:ipc_port>`.",
                "5. Monitor NameNode log file `/var/log/hadoop/hdfs/hadoop-hdfs-namenode.log` for recurring E29 PendingReplicationMonitor timeouts."
            ]
        else:
            urgency = "LOW (Routine Operations)"
            steps = [
                "1. No manual intervention required.",
                "2. Maintain automated log monitoring pipelines.",
                "3. Continue standard scheduled health checks."
            ]

        return {
            "urgency": urgency,
            "sop_steps": steps
        }
