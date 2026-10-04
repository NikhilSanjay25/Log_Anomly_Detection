# Agentic HDFS Log Anomaly Detection

A multi-agent system for HDFS logs. A Coordinator Agent runs specialised agents for log identification, preprocessing, anomaly detection (Transformer → FAISS → RAG → Random Forest, plus an advisory second opinion from a fine-tuned Qwen3-0.6B LoRA), root cause analysis and SLM-generated insights (Qwen3-1.7B). The results appear in a Streamlit UI.

```
Log Ingestion (upload / paste / Kubernetes: future)
  → Log Identification Agent       identifies HDFS format, picks the workflow, rejects other families
  → Preprocessing & Cleaning Agent parsing, normalisation, de-duplication, noise removal, formatting
  → Medallion processing           Bronze (raw cleaned) → Silver (structured, enriched) → Gold (per-block sessions, correlations)
  → Anomaly Detection Agent        Transformer encoder → FAISS → RAG augmentation → Random Forest
                                   + advisory Qwen3-0.6B LoRA opinion on uncertain sessions (never overrides)
  → Root Cause Analysis Agent      anomaly evidence, similar logs, context and correlation → ranked root causes
  → Insight Agent (SLM)            explanation, impact, recommendations, next-best actions (validated JSON)
  → User Interface                 summary, RCA, AI insights, recommendations, evidence, reports, agent trace
Coordinator Agent: plans the workflow, assigns tasks, keeps shared context and run memory, carries messages
between agents, validates each agent's output (retry or fallback) and runs the workflow end to end.
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_rag_metadata.py     # once: builds rag_metadata.npz + event_stats.json (~30 s)
python scripts/make_sample_logs.py       # once: demo inputs in samples/
streamlit run streamlit_app.py
```

Command line:

```bash
python -m log_agents.cli samples/hdfs_raw_sample.log                  # full run, Markdown report to stdout
python -m log_agents.cli my.log --no-slm --lora off --out report.md    # fast, no SLMs
python -m pytest tests -q                                              # 16 tests, no SLM weights needed
```

## Supported inputs (HDFS only)

| Format | Example |
|---|---|
| Raw HDFS log lines | `081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating` |
| `Event_traces.csv` | `BlockId,Label,Features,...`. If labels are present, the run also reports accuracy, precision, recall and F1 |
| Event sequences | `E22 E5 E5 E5 E11 E9 ...` or `blk_123: E22 E5 ...`, one session per line |

## Required files

| File | Purpose |
|---|---|
| `transformer_backbone.pth`, `model_config.joblib`, `event2id.joblib`, `id2event.joblib` | Transformer encoder |
| `hdfs_faiss_index.index` | FAISS vector DB of 460,048 training embeddings |
| `rag_random_forest.joblib` | RAG-augmented Random Forest |
| `rag_metadata.npz`, `event_stats.json` | built by `scripts/build_rag_metadata.py`; maps FAISS ids to sequences and labels |
| `qwen3-hdfs-ckpt/qwen3-hdfs-lora/` | optional fine-tuned LoRA adapter from `slm_Qwen3_0_6ipynb.ipynb` |
| `preprocessed/preprocessed/` | HDFS_v1 dataset; needed only to rebuild the metadata or the samples |

Qwen3-1.7B (3.8 GB) and Qwen3-0.6B (1.5 GB) download from Hugging Face on first use.

## Configuration (environment variables)

`LAD_INSIGHT_MODEL` (default `Qwen/Qwen3-1.7B`), `LAD_LORA_DIR`, `LAD_ARTIFACTS_DIR`, `LAD_DATASET_DIR`, `LAD_DATA_DIR`, `LAD_LORA_MAX`. See `log_agents/config.py`.

Use a CUDA build of PyTorch to run the SLMs on the GPU. Measured on the 175-block sample: LoRA step about 12 s and Qwen3-1.7B insight about 25 s on an RTX 4060, versus about 150 s and 137 s on CPU.

## Measured results

All numbers come from the scripts in this repo and the held-out fold of the RF's 5-fold CV (115,012 sessions, 3,368 anomalous).

| What | Result |
|---|---|
| Transformer → FAISS → RAG → RF, full held-out fold | P 0.9982 · R 0.9997 · F1 0.9990 (7 errors) |
| Same, only the 2,414 sessions whose sequence never appears in the RF training fold | F1 0.9955 (6 errors) |
| "Uncertain" sessions flagged by the agent | 62 sessions, containing 6 of the 7 errors |
| On those 62: RF vs LoRA SLM | RF 6 errors · LoRA 27 errors (all false positives) |

That last row is why the LoRA SLM is **advisory only**. Its vote is shown and disagreements are flagged, but the RF label is final. The LoRA was trained on a different split (`slm_Qwen3_0_6ipynb.ipynb`), so some of these sessions may have been in its training data. Its error rate here is still far higher than the RF's.

## Limitations

* **RCA has no ground truth.** HDFS_v1 labels say *whether* a block is anomalous, not *why*. The rules are checked only for coverage. Each fires on ≤0.33% of normal sessions, and every anomalous session gets a named cause, but whether each diagnosis is correct is not measured. The per-hypothesis percentage is a share of heuristic rule scores, not a probability.
* **SLM validation catches invented block ids, event ids, IPs and numbers, but not wrong reasoning.** In 3 test runs on the samples, Qwen3-1.7B passed validation on the first attempt each time. That is a small sample.
* **Duplicate removal is unverified on real raw logs.** The Preprocessing agent drops exactly repeated lines; the labelled dataset was built without that step. The raw HDFS.log isn't in this repo, so I couldn't check whether real logs contain legitimately repeated lines.
* **HDFS only.** Kubernetes ingestion is a stub.
* The RF pickle was made with scikit-learn 1.7.0 and loads under 1.8.0 with a version warning. Outputs were never compared against 1.7. The held-out results above were produced under 1.8.0.

## Notes

* **RAG feature fix.** The RF was trained on `concat(feature, mean of raw neighbour features)`. The FAISS index stores *L2-normalised* vectors, and the old app averaged those, which caused train/serve skew. `rag_metadata.npz` restores the raw neighbour features. With the fix, a 20k sample of the held-out CV fold scores F1 0.999.
* Medallion layers are written to `data/medallion/<run_id>/` and run summaries to `data/runs/`. Run summaries are the coordinator's long-term memory.
* Gemini is no longer used. All language-model work runs locally.
* `pre-processing/preprocess.py` now counts an event once per block per line. Before, E21 lines, which contain the block id twice, were double-counted, so its output did not match the labelled dataset.
