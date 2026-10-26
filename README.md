# Agentic Log Anomaly Detection: HDFS · BGL · OpenStack

A multi-agent system that finds anomalous sessions in three kinds of system logs, explains them, and lets you
question the result in a chat. A Coordinator Agent runs specialised agents (identification, cleaning, medallion
processing, detection, root-cause analysis, insight) and validates each step. Detection uses a Transformer encoder,
FAISS retrieval and a Random Forest, trained separately for each dataset. The only language model is a local Qwen3,
and it is used **only by the chat**, which is fed the relevant raw log lines.

```
Log Ingestion (upload / paste / Kubernetes: future)
  → Log Identification Agent       which dataset (HDFS, BGL, OpenStack) and which input format? rejects anything else
  → Preprocessing & Cleaning Agent parsing, normalisation, de-duplication, noise removal
  → Medallion processing           Bronze (cleaned lines) → Silver (events) → Gold (sessions)
  → Anomaly Detection Agent        Transformer encoder → FAISS → RAG augmentation → Random Forest (that dataset's models)
  → Root Cause Analysis Agent      dataset knowledge base + evidence + similar sessions + correlations → ranked causes
  → Insight Agent                  explanation, impact, recommendations, next-best actions (knowledge base, no LLM)
  → User Interface / CLI           summary, RCA, insights, evidence, reports, agent trace, chat
Chat (Qwen3-1.7B)                  answers follow-up questions from the run's facts + the relevant raw log lines
Coordinator Agent: plans the workflow, carries messages between agents, validates every output (retry / fallback).
```

## The three datasets

| | HDFS | BGL (BlueGene/L) | OpenStack |
|---|---|---|---|
| Source | LogHub HDFS_v1 | LogHub BGL | LogHub / DeepLog OpenStack |
| A **session** is | a block (`blk_…`) | a window of 100 consecutive messages (`win_N`) | a VM instance (UUID in `[instance: …]`) |
| Input formats | raw lines · `Event_traces.csv` · event sequences | raw RAS lines | raw nova lines |
| Events | 29 hand-written templates | 510 templates mined with Drain | 23 templates mined with Drain |
| Labels | per block | per line (alert tag); a window is anomalous if it holds an alert | 4 instances with injected delays |
| Features the Random Forest reads | embedding + neighbours | severity counts + bag-of-words of the messages (**only these**) | embedding + neighbours + the four `Took N seconds` timings |
| Root causes | 10 hand-written HDFS causes | alert categories learned from the labels + family text | timing, incomplete-spawn, error / warning causes |
| Models | repo root (`transformer_backbone.pth`, …) | `artifacts/bgl/` | `artifacts/openstack/` |

The Identification Agent picks the dataset from the log lines (or you force one). Everything downstream is written once
against a **dataset profile** (`log_agents/datasets/`) that supplies what differs: parsing, sessions, event meaning,
model artifacts and root causes.

## Repository layout

```
streamlit_app.py            web UI (Streamlit)
log_agents/
  agents/                   the agents: identification, preprocessing, anomaly, rca, insight, coordinator
  datasets/                 one profile per dataset (HDFS, BGL, OpenStack): parsing, sessions, events, root causes
  drain.py                  template miner used for BGL and OpenStack
  medallion.py              Bronze / Silver / Gold layers
  ml.py                     Transformer encoder, FAISS retrieval, Random Forest scoring
  chat.py, slm.py           follow-up chat and the Qwen model wrapper (the only language model)
  knowledge.py              HDFS root-cause knowledge base
scripts/                    train_dataset.py (BGL, OpenStack), build_rag_metadata.py (HDFS), sample builders
artifacts/bgl, artifacts/openstack   trained models, metrics.json, templates
(repo root)                 HDFS models: transformer_backbone.pth, rag_random_forest.joblib, hdfs_faiss_index.index (Git LFS)
samples/                    demo inputs for all three datasets
tests/                      39 tests
OtherTests/, Baselines/     experiment notebooks and baseline models (not part of the pipeline)
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_sample_logs.py          # HDFS demo inputs (already in samples/; needs HDFS_v1 to rebuild)
python scripts/make_dataset_samples.py      # BGL + OpenStack demo inputs
streamlit run streamlit_app.py
```

Command line:

```bash
python -m log_agents.cli samples/bgl_raw_sample.log                       # dataset detected automatically
python -m log_agents.cli samples/openstack_raw_sample.log --out report.md
python -m log_agents.cli my.log --dataset bgl --chat                       # then ask follow-up questions
python -m pytest tests -q                                                  # 39 tests, no language-model weights needed
```

### Getting the data and training BGL / OpenStack

The datasets are not committed (they are in `.gitignore`). Download **BGL.zip** (57 MB) and **OpenStack.tar.gz** (5 MB) from the
[LogHub release](https://zenodo.org/records/8196385) and extract them to `Dataset_BGL/` (`BGL.log`) and
`Dataset_OpenStack/` (`openstack_*.log`, `anomaly_labels.txt`). HDFS_v1 (`HDFS_v1.zip`, 187 MB, same release) is only needed to
rebuild the HDFS samples or metadata; point `LAD_DATASET_DIR` at its `preprocessed/` folder. Then:

```bash
python scripts/train_dataset.py --dataset bgl         # ~2 min on a GPU
python scripts/train_dataset.py --dataset openstack   # ~15 s
```

The trained artifacts (13 MB and 1 MB) are committed in `artifacts/`, so training is only needed to reproduce them.
Training uses the same cleaning, Silver and Gold code as inference and evaluates through the real `AnomalyPipeline`.
HDFS artifacts come from `OtherTests/RAG__Trans_RF.ipynb`; `scripts/build_rag_metadata.py` rebuilds
`rag_metadata.npz` and `event_stats.json` for them.

## The chat

Qwen3-1.7B (`LAD_CHAT_MODEL`) answers questions about a finished run. It is given:

* the run's **facts sheet** (root causes, evidence, similar historical sessions, normal examples) and the analysis
  the Insight Agent wrote, in the system prompt;
* **block / window / instance details** for every session the question names;
* **raw log lines** chosen for the question: the lines of a named session (error and warning lines first, capped),
  the lines of an event id such as `E12`, the lines containing the question's keywords, or, for a general question,
  the lines of the most anomalous sessions.

Answers are not blocked. Anything the run cannot back up (unknown session ids, event ids, node names / IPs, or
numbers that appear neither in the facts nor in the log lines the model was shown) is flagged under the answer.
Inputs without raw lines (HDFS event sequences, the trace CSV) get no log section.

## Measured results

All numbers come from the scripts in this repo (`artifacts/*/metrics.json` for BGL and OpenStack).

**HDFS** (575,061 blocks, 2.9% anomalous). The deployed model was trained on the training split of fold 1 of
`StratifiedKFold(5, shuffle=True, random_state=42)`. Evaluated on the fold it never saw (fold 1 validation, n = 115,013):
precision 0.9935 · recall 0.9979 · **F1 0.9957** (29 errors). An earlier README quoted F1 0.9990; that was measured on
fold 5, which lies inside the training set, so it was optimistic.

**BGL** (4.7 M lines → 47,135 windows, 10.2% anomalous). Evaluated by *forward chaining*: train on the first part of the
log, test on the next 20%. The deployed model is the last fold (train on the first 80% = 37,708 windows, test on the
last 9,427, of which 8.6% are anomalous). The test period contains alert types never seen in training.

| Last 20% of the log | Precision | Recall | F1 |
|---|---|---|---|
| **Deployed model** (Random Forest on word + severity features) | 0.974 | 0.970 | **0.972** |
| Random Forest on event counts | 0.969 | 0.387 | 0.553 |
| Logistic regression on event counts | 0.974 | 0.326 | 0.488 |
| Same Random Forest fed the Transformer embedding and FAISS neighbours as well | 0.579 | 0.684 | 0.627 |

With less history the score is lower: training on the first 40% and testing on the next 20% gives F1 0.752, 60% → 0.764,
80% → **0.972** (mean 0.829). Alert types keep changing, so the model needs recent data, and it should be retrained as new
logs arrive. The first two test periods are also small or unusual (1% and 12% anomalous).

*Why the Random Forest reads words.* Event ids are not stable over time: 259 of the 263 windows the first version missed
contained templates that never occurred in training, and the id-based Transformer embedding and its neighbours misled the
forest (last row). The words of a new template ("error", "failed", the log level and component) still carry the signal,
and the vocabulary is built from training windows only. The Transformer and FAISS remain in the pipeline for retrieving
similar windows (evidence for the root-cause analysis and the chat) and for the uncertainty flag. The labels
define alerts by message content, so a model that reads message text is expected to do well once the alert types are
known; the score says nothing about alert types the log has not yet shown.

**OpenStack** (2,069 instances, **only 4 labelled anomalous**). The labelled anomalies are injected delays: their event
order is identical to a normal instance and only the timings differ (spawn 29–51 s against 19–27 s for every normal one).
Leave-one-anomaly-out cross-validation (4 folds, each holds out one anomaly and a quarter of the normal instances):

| | Anomalies found | False positives (of 2,065 normal) | F1 |
|---|---|---|---|
| With timing features (deployed) | **4 of 4** | 1 | **0.889** (precision 0.80, recall 1.0) |
| Event sequence only | no usable signal (every instance scores ≈ 0.54) | 1,734 | 0.005 |

The one false positive is the slowest normal instance (spawn 26.7 s). It landed in a fold whose training data contained
no normal instance slower than 22.6 s, so the model placed its boundary near 26 s. Any model fitted on that data would
make the same call; getting it right would need that instance in the training set, which would make the estimate leaky.
Four positives is far too few for a reliable estimate: this shows that timing carries the signal, not an accuracy figure.
The deployed model is trained on all instances.

**What changed after the first results (disclosure).** Both models were first built with a shared design and scored
BGL F1 0.62 and OpenStack F1 0.86. I then changed, after seeing those results: (BGL) added word features and let the forest
read only them; (OpenStack) made the forest see every feature and sample (`bootstrap=False`, `max_features=None`, because
there are only 4 positives and the 6 informative columns were rarely sampled) and removed session duration and longest
gap from the model input, because they depend on how long a VM was kept alive. These are model-design choices tested on
the same evaluation data, so the final numbers are somewhat optimistic compared with a design fixed in advance.

## Configuration (environment variables)

`LAD_CHAT_MODEL` (default `Qwen/Qwen3-1.7B`), `LAD_ARTIFACTS_DIR` (HDFS), `LAD_ARTIFACTS_ROOT` (BGL / OpenStack),
`LAD_DATASET_DIR` (HDFS_v1), `LAD_BGL_DIR`, `LAD_OPENSTACK_DIR`, `LAD_DATA_DIR`, `LAD_CHAT_LOG_LINES`, `LAD_CHAT_LOG_CHARS`.
See `log_agents/config.py`. Use a CUDA build of PyTorch to run the chat model on the GPU. The weights (3.8 GB) download
from Hugging Face on first use.

## Limitations

* **RCA has no ground truth.** HDFS labels say *whether* a block is anomalous, not *why*. BGL's alert tags give coarse
  categories but are assigned by external expert rules; OpenStack's four anomalies have a known cause (a delay) but no
  more. The per-hypothesis percentage is a share of heuristic rule scores, not a probability.
* **BGL windows are labelled by the alert lines inside them**, and the alert message text also determines the template,
  so a text-reading model does well on known alert types. A new kind of alert that shares no words with earlier ones would
  still be missed; the model also needs retraining as the log evolves.
* **Templates are mined from the whole log** (BGL: all 4.7 M lines). A deployment would meet unseen templates and have
  to mine them online; lines matching no template are dropped and counted ("matched no template").
* **OpenStack uses only lines tagged `[instance: …]`.** Other lines (API polling, image cache) stay in Bronze.
* **The chat can reason wrongly.** Grounding checks catch invented ids, events, nodes and numbers, not wrong reasoning.
* **Duplicate removal is unverified on real raw HDFS logs** (the raw HDFS.log is not in this repo).
* **Kubernetes ingestion is a stub.** A fourth dataset needs a profile, a template miner run and a trained model.
* The HDFS RF pickle was made with scikit-learn 1.7.0 and loads under newer versions with a version warning.
* `qwen3-hdfs-ckpt/` (a LoRA classifier from `slm_Qwen3_0_6ipynb.ipynb`) and the notebooks in `OtherTests/` are no longer
  part of the pipeline: the LoRA model was measured worse than the Random Forest and was removed from detection.

## Notes

* **RAG feature fix (HDFS).** The RF was trained on `concat(feature, mean of raw neighbour features)`. The FAISS index
  stores *L2-normalised* vectors, so `rag_metadata.npz` restores the raw neighbour features to match training.
* **Leave-self-out retrieval (OpenStack).** When building the RF's training features, each training session's own
  entry is removed from its neighbours, so the RF never learns from a neighbour that does not exist for a new session.
  The HDFS RF was trained by the notebook without this. (BGL's forest does not read the neighbours at all.)
* Medallion layers are written to `data/medallion/<run_id>/` and run summaries to `data/runs/`.
* `pre-processing/preprocess.py` counts an event once per block per line; before, E21 lines (which contain the block
  id twice) were double-counted, so its output did not match the labelled dataset.
