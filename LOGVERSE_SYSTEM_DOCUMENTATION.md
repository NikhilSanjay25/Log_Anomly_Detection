# LogVerse AI Platform — Complete System Architecture & Technical Documentation

**System Title:** LogVerse AI Platform  
**Subtitle:** An Agentic AI Data Platform for Enterprise Log Intelligence using Medallion Architecture, Multi-Agent Orchestration, PyTorch Deep Learning, and Causal Root Cause Analysis  
**Repository Location:** `c:\1Data\Ved_Data\AmritaCSE\FinalYearProject\C_ProjectPhase_2(23CSE498)\Log_Anomly_Detection`  

---

## 1. Executive Overview

**LogVerse AI Platform** is an enterprise-grade Agentic AI Data Platform designed for autonomous log intelligence, anomaly detection, root cause analysis (RCA), and operational remediation. 

Traditional log management tools (such as Splunk or Kibana) require engineers to manually query raw logs, write complex search expressions, and diagnose root causes under high pressure. **LogVerse AI Platform** automates this entire lifecycle by combining:
1. **Medallion Lakehouse Architecture** ($\text{Bronze} \rightarrow \text{Silver} \rightarrow \text{Gold}$) for scalable log data engineering.
2. **Pretrained PyTorch Transformer Backbone** for deep sequence pattern evaluation and anomaly probability scoring.
3. **Multi-Agent Orchestration Engine** with 6 specialized AI personas collaborating in real-time.
4. **Causal Root Cause Analysis (RCA) Graph Engine** visualizing fault propagation across system components.
5. **Local Small Language Model (SLM) Diagnostic Engine** translating raw log events into plain-English root cause explanations and step-by-step SOP runbooks.

---

## 2. System Architecture & Component Diagram

```
                        +-----------------------------------------+
                        |        AGENTIC USER EXPERIENCE          |
                        |      (Desktop UI / Streamlit App)       |
                        +-----------------------------------------+
                                             |
        +------------------------------------+------------------------------------+
        |                                    |                                    |
+---------------+                    +---------------+                    +---------------+
|  Interactive  |                    |  Multi-Agent  |                    | Medallion Data|
|   RCA Graph   |                    |   Canvas &    |                    |  Pipeline &   |
|   (Vis.js)    |                    | Timeline Stream|                   |  AI Catalog   |
+---------------+                    +---------------+                    +---------------+
        |                                    |                                    |
        +------------------------------------+------------------------------------+
                                             |
                        +-----------------------------------------+
                        |       MULTI-AGENT ORCHESTRATOR          |
                        |         (logverse_agents.py)            |
                        +-----------------------------------------+
                                             |
   +-----------------------+-----------------+-----------------------+-----------------------+
   |                       |                 |                       |                       |
   v                       v                 v                       v                       v
[Planner Agent]     [Catalog Agent]   [ML Anomaly Agent]      [RCA Graph Agent]     [SLM Diagnostic Agent]
                                             |                       |                       |
                                             v                       v                       v
                                    +-----------------+     +-----------------+     +-----------------+
                                    | PyTorch Model   |     | NetworkX /      |     | Local SLM /     |
                                    | (Transformer)   |     | Vis.js Causal   |     | Qwen Reasoning  |
                                    | Backbone        |     | Graph Engine    |     | Engine          |
                                    +-----------------+     +-----------------+     +-----------------+
                                             |
                                             v
                        +-----------------------------------------+
                        |         MEDALLION DATA LAKEHOUSE        |
                        |        (logverse_pipeline.py)           |
                        +-----------------------------------------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
              v                              v                              v
     +-----------------+            +-----------------+            +-----------------+
     |  BRONZE LAYER   |    --->    |  SILVER LAYER   |    --->    |   GOLD LAYER    |
     | Raw Immutable   |            | Regex Parsed &  |            | AI Feature Sets |
     |  Log Store      |            |  Sessionized    |            | & Aggregations  |
     +-----------------+            +-----------------+            +-----------------+
```

---

## 3. Comprehensive File-by-File Breakdown

### 3.1 New Pipeline & Core System Files

#### 📄 `logverse_pipeline.py`
- **Purpose:** Implements the Medallion Data Architecture ($\text{Bronze} \rightarrow \text{Silver} \rightarrow \text{Gold}$) and the Enterprise AI Metadata Catalog.
- **Key Classes & Methods:**
  - `MedallionPipeline`: The main data orchestration class.
  - `process_raw_logs(log_content, source_name)`: Reads raw log text, parses records into Bronze, Silver, and Gold DataFrames, and constructs the AI Catalog dictionary.
  - `generate_sample_hdfs_log()`: Helper function providing realistic sample HDFS log sequences for execution and demonstration.
- **Data Layers Handled:**
  - **Bronze Layer:** Stores immutable raw log lines with Line IDs, ingest timestamps, and source file metadata.
  - **Silver Layer:** Applies HDFS template matching regex (`E1` through `E29`), normalizes timestamps, extracts component names (`dfs.DataNode$DataXceiver`, `dfs.FSNamesystem`), flags log severity levels, and sessionizes logs by Block ID (`blk_-...`).
  - **Gold Layer:** Aggregates block session metrics, computes event frequency distributions, flags error presence, and constructs clean event sequence lists for AI/ML inference.
  - **AI Metadata Catalog:** Indexes line counts, unique event types found, processing latency, and anomaly totals.

#### 📄 `logverse_ml.py`
- **Purpose:** Encapsulates the PyTorch Transformer backbone anomaly detection model (`transformer_backbone.pth`) and vocabulary mappings (`event2id.joblib`, `id2event.joblib`).
- **Key Classes & Methods:**
  - `PositionalEncoding(nn.Module)`: PyTorch module injecting sinusoidal positional embeddings into log event sequences.
  - `TransformerModel(nn.Module)`: Deep learning model architecture with embedding layer ($d=64$), positional encoder, 2-layer TransformerEncoder ($nhead=4, d_{ff}=256$), and a linear classification head ($num\_classes=2$).
  - `MLAnomalyDetector`: High-level wrapper class that loads pretrained weights and executes session inference.
  - `parse_sequence_to_ids(event_list, max_len=60)`: Converts event ID string tokens into zero-padded 1D integer tensors.
  - `predict_session(event_list)`: Executes model forward pass, extracts 64-dimensional sequence embeddings, calculates anomaly softmax probabilities, and pinpoints suspicious error event IDs.

#### 📄 `logverse_rca_graph.py`
- **Purpose:** Constructs directed causal dependency graphs for Root Cause Analysis (RCA) and renders interactive network graph visualizations.
- **Key Classes & Methods:**
  - `EVENT_CAUSALITY_MAP`: Expert knowledge dictionary mapping HDFS event IDs (`E1`–`E29`) to operation categories, severity levels, and structural root causes (e.g., `E7` $\rightarrow$ `Connection reset by peer or I/O write fault`).
  - `RCAGraphBuilder`: Graph construction engine using `NetworkX`.
  - `build_graph_from_session(block_id, parsed_records, ml_analysis)`: Connects System Node $\rightarrow$ Component Nodes $\rightarrow$ Event Sequence Step Nodes $\rightarrow$ Primary Root Cause Node.
  - `export_interactive_html(height="500px")`: Generates a standalone, interactive JavaScript network graph (`Vis.js`) with custom node colors, hierarchical physics, and hover tooltips.

#### 📄 `logverse_agents.py`
- **Purpose:** Multi-Agent System Orchestrator coordinating 6 specialized AI Agents and executing local SLM diagnostic reasoning.
- **Key Personas & Classes:**
  - `AgentMessage`: Data structure for inter-agent messages containing sender, recipient, action, content payload, and timestamp.
  - `MultiAgentOrchestrator`: Coordinates agent workflows and maintains communication logs.
  - **The 6 Specialized AI Agents:**
    1. **Planner Agent:** Analyzes incoming log files and generates a 7-step execution workflow plan.
    2. **Catalog Agent:** Triggers Medallion pipeline transformations and registers data schemas in the AI Catalog.
    3. **ML Anomaly Agent:** Invokes the PyTorch Transformer backbone to compute block anomaly probabilities.
    4. **RCA Graph Agent:** Directs `RCAGraphBuilder` to construct causal failure networks and pinpoint root cause nodes.
    5. **SLM Diagnostic Agent:** Generates plain-English natural language diagnosis, mechanism breakdown, and impact analysis.
    6. **Remediation Agent:** Formulates step-by-step operational SOP runbooks and bash commands for operators.

#### 📄 `logverse_desktop_app.py`
- **Purpose:** Aesthetic, dark glassmorphism desktop product interface built with Streamlit.
- **Key UI Sections:**
  - **Header & System KPI Bar:** Visualizes system health badges, total raw lines, parsed events, session block counts, and active agent statuses.
  - **Sidebar Controls:** Allows file uploads (`.txt`, `.log`), sample HDFS log selection, live Kubernetes/Docker log stream simulation, and SLM backbone configuration.
  - **Tab 1: 🤖 Agent Workflow Canvas:** Shows real-time agent status cards and streams live inter-agent communication messages.
  - **Tab 2: 🕸️ Causal RCA Graph:** Embeds interactive Vis.js network graph highlighting the primary red root cause node and affected block ID.
  - **Tab 3: 🏅 Medallion Data Pipeline:** Interactive data tables exploring Bronze, Silver, Gold layers, and AI Catalog metadata JSON.
  - **Tab 4: 🧠 PyTorch ML Backbone:** Displays model confidence metrics, suspicious event lists, and 64-dimensional deep feature vector line charts.
  - **Tab 5: 💡 Local SLM Diagnosis & Runbook:** Presents human-understandable executive summaries, mechanism analysis, operational impacts, and runnable bash SOP steps.

#### 📄 `launch_desktop.py`
- **Purpose:** Entry point runner script to launch the desktop application on `http://localhost:8501`.
- **Key Mechanics:** Configures UTF-8 console output encoding on Windows environments and executes `streamlit run logverse_desktop_app.py`.

#### 📄 `LogVerse_Agentic_Pipeline.ipynb`
- **Purpose:** End-to-end Jupyter Notebook demonstrating step-by-step execution of the entire pipeline, from Medallion data engineering to PyTorch ML anomaly scoring, Multi-Agent workflow logging, Matplotlib RCA graph rendering, and SLM diagnostic output generation.

---

### 3.2 Pre-Existing Workspace Artifacts & Pretrained Models (Preserved Unmodified)

- **`transformer_backbone.pth`**: Pretrained PyTorch state dictionary containing learned weights for embedding, positional encoder, transformer encoder layers, and linear output head.
- **`event2id.joblib`**: Joblib dictionary mapping event string keys (`'E1'`, `'E5'`, `'E7'`, etc.) to integer vocabulary indices (`1` to `29`).
- **`id2event.joblib`**: Reverse joblib dictionary mapping integer vocabulary indices back to event string keys.
- **`test_samples_corrected.txt`**: Sample HDFS log sequence file used for baseline testing and model evaluation.
- **`streamlit_app.py`**: Legacy baseline Streamlit application preserved for reference.
- **`pre-processing/preprocess.py`**: Original HDFS log parsing script defining the 29 standard HDFS event templates.
- **`slm_Qwen3_0_6ipynb.ipynb`**: Original notebook demonstrating Qwen local SLM integration and baseline dataset discovery.

---

## 4. End-to-End Data & Execution Flow

```
[Raw Log File / Stream] 
       │
       ▼ (1) Ingestion
[Bronze Layer] ---> Immutable raw text storage & metadata assignment
       │
       ▼ (2) Parsing & Sessionization
[Silver Layer] ---> Regex template matching (E1-E29), timestamp extraction & Block ID grouping
       │
       ▼ (3) Feature Extraction
[Gold Layer]   ---> Aggregated session sequences & occurrence matrices
       │
       ├─────────────────────────────────┐
       ▼ (4) Model Inference             ▼ (5) Metadata Registration
[PyTorch Transformer Backbone]     [AI Catalog Metadata Registry]
(transformer_backbone.pth)               │
       │ (Sequence Probabilities &               │ (Data Lineage & Schemas)
       │  Feature Embeddings)                    │
       ▼                                         ▼
+-------------------------------------------------------------------+
|                     MULTI-AGENT SYSTEM ORCHESTRATOR               |
|                                                                   |
|   Planner Agent ➔ Catalog Agent ➔ ML Agent ➔ RCA Agent            |
|                     ➔ SLM Agent ➔ Remediation Agent               |
+-------------------------------------------------------------------+
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Causal RCA Network Graph]        [Local SLM Reasoning Engine]
(Vis.js / NetworkX)               (Qwen / Local Prompt Adapter)
       │                                 │
       ▼                                 ▼
(Highlighted Root Cause Node)     (Plain-English Executive Summary & SOP Runbook)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
       [LOGVERSE AI DESKTOP PRODUCT UI]
```

---

## 5. Multi-Agent Communication Sequence & Protocol

When a log dataset is submitted, the **MultiAgentOrchestrator** coordinates inter-agent messaging as follows:

1. **User Interface $\rightarrow$ Planner Agent:**
   - *Action:* `REQUEST_ANALYSIS`
   - *Payload:* Log source name and total line count.
2. **Planner Agent $\rightarrow$ Catalog Agent:**
   - *Action:* `EXECUTE_INGESTION_PLAN`
   - *Payload:* 7-step execution plan instructions.
3. **Catalog Agent $\rightarrow$ ML Anomaly Agent:**
   - *Action:* `DATASET_REGISTERED`
   - *Payload:* Record counts for Bronze, Silver, and Gold tiers, and AI Catalog schema confirmation.
4. **ML Anomaly Agent $\rightarrow$ RCA Graph Agent:**
   - *Action:* `ANOMALY_DETECTION_COMPLETE`
   - *Payload:* Evaluated block session scores, anomaly probabilities, and identified anomalous block IDs.
5. **RCA Graph Agent $\rightarrow$ SLM Diagnostic Agent:**
   - *Action:* `GRAPH_CONSTRUCTED`
   - *Payload:* Causal graph node/edge statistics and primary identified root cause string.
6. **SLM Diagnostic Agent $\rightarrow$ Remediation Agent:**
   - *Action:* `DIAGNOSIS_GENERATED`
   - *Payload:* Natural language mechanism breakdown and operational impact analysis.
7. **Remediation Agent $\rightarrow$ User Interface:**
   - *Action:* `WORKFLOW_READY`
   - *Payload:* Urgency level and step-by-step runnable SOP bash commands.

---

## 6. Industry-Level Additions & Novelty Summary

1. **Medallion Architecture Integration:** Applying data engineering principles ($\text{Bronze} \rightarrow \text{Silver} \rightarrow \text{Gold}$) ensures high data quality, immutability, and auditable lineage.
2. **Enterprise AI Catalog:** Registering datasets, schemas, and model embeddings into a centralized metadata catalog enables dynamic data discovery by autonomous agents.
3. **Multi-Agent Visual Canvas:** Rather than a black-box model, users can watch agents think, send structured messages, and collaborate in real-time.
4. **Graph-Based Root Cause Analysis (XAI):** Visualizing causal propagation across system components provides explainable evidence for security and ops teams.
5. **On-Premises SLM Independence:** Uses local small language models (SLM) and local PyTorch transformers, eliminating reliance on costly third-party cloud APIs and protecting log data privacy.
6. **Live Kubernetes & Docker Stream Adapters:** Designed for seamless transition from static log files to real-time container log streams (`kubectl logs` / Docker log daemons).

---

## 7. Instructions for Execution

### Running the Desktop Application
```bash
python launch_desktop.py
```
Open your browser to: **`http://localhost:8501`**

### Running the Jupyter Notebook
Open `LogVerse_Agentic_Pipeline.ipynb` in VS Code or Jupyter Lab and execute all cells sequentially.
