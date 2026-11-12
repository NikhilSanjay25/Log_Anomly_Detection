# Agentic Log Anomaly Detection: HDFS · BGL · OpenStack

A multi-agent system that finds anomalous sessions in three kinds of system logs, **investigates each one with a local
language model**, explains the result, and lets you question it in a chat. A Coordinator Agent runs specialised agents
(identification, cleaning, medallion processing, detection, **investigation**, insight) and validates each step.
Detection uses a Transformer encoder, FAISS retrieval and a Random Forest, trained separately for each dataset. The
**Investigator Agent** is a Qwen3 model (1.7B, 4B or 8B, run locally) that decides which evidence to look at next and
which root cause the evidence supports. The same model answers follow-up questions in the chat, fed the relevant raw logs.

```
Log Ingestion (upload / paste / Kubernetes: future)
  → Log Identification Agent       which dataset (HDFS, BGL, OpenStack) and which input format? rejects anything else
  → Preprocessing & Cleaning Agent parsing, normalisation, de-duplication, noise removal
  → Medallion processing           Bronze (cleaned lines) → Silver (events) → Gold (sessions)
  → Anomaly Detection Agent        Transformer encoder → FAISS → RAG augmentation → Random Forest (that dataset's models)
  → Investigator Agent  (Qwen3)    per anomaly: picks tools from a menu, reads the evidence, chooses the root cause
        tools = ordinary code: read logs · similar past cases · event statistics · timing / lifecycle checks ·
                node & time correlation · repeat pattern · rule-engine ranking · knowledge-base text match
        fallback = the rule-based root-cause analysis (also the baseline it is measured against)
  → Insight Agent                  explanation, impact, recommendations from the investigation's conclusions + knowledge base
  → User Interface / CLI           summary, investigation trace, RCA, insights, evidence, reports, chat
Chat (same Qwen3 model)            answers follow-up questions from the run's facts + the relevant raw log lines
Coordinator Agent: plans the workflow, carries messages between agents, validates every output (retry / fallback).
```

## How the investigator works

For each group of anomalies with an identical event pattern (one representative per group, at most 30 groups per run) the
model runs a short loop. **Each turn it sees the evidence gathered so far and a lettered menu of actions, and answers with
one letter.** It chooses which evidence to look at, in which order, whether to ask the rule engine for its ranking, and
when to stop; then it is shown a shortlist of candidate root causes and picks one.

* **Nothing is parsed from free text.** An answer is the allowed letter the model scores highest
  (`LocalLLM.choose`: one forward pass, softmax over the allowed letters only), so a small model cannot produce a
  malformed action or invent a cause.
* **The model decides the path, not the facts.** It does not label sessions anomalous (the trained detector does), the
  tools are ordinary code, the shortlist comes from the rule engine plus a text match against the knowledge base, and the
  report wording comes from the knowledge base and the evidence. Every step, observation and probability is recorded and
  shown in the **Investigation** tab and the report.
* **Position bias is handled.** Models of this size like the first menu option (on the development set the 1.7B model answered "A" in all 51
  menus without evidence and in 38 of 51 with it). The shortlist is shuffled per case, and the final choice averages the model's scores over every
  rotation of the menu (`listwise_debiased`). Two alternatives are implemented and were compared on a development set
  (`RunOptions.agent_decision`).
* **It degrades safely.** If the weights are missing, the GPU cannot load them, or the model keeps failing, the
  rule-based analysis runs instead and the report says so (`agent_model="rules"` selects it on purpose).
* **The BGL alert-tag column is never shown to the model** (a label is not a log field), and the BGL root-cause signatures
  come from training windows only.

## Which model? A measured comparison

The same investigator, tools and shortlist were run with each model and compared with the rule-based analysis it replaces.
Run on an RTX 4060 laptop GPU (8 GB). Qwen3-4B and 8B run in 4-bit NF4 (bitsandbytes) so they fit; the 1.7B model is
shown in bf16 and in 4-bit.

**Ground truth.** BGL labels every alert line with a category tag (APPSEV, KERNTERM, …). For 146 truly anomalous windows from
the held-out last ~18% of the log (that the detector flagged; the other windows are normal windows mixed in so the base rate
is ~10%) a chosen cause is correct if it is one of the window's tags. The label column is replaced by `-` in the text the
system analyses. This scores root-cause analysis only, not detection: every configuration sees the identical detector
output. The final menu contains the true cause in 96% of cases, which is the ceiling for any configuration.

| BGL, 146 cases | Correct | Accuracy (95% CI) | vs rules: wins / losses (sign-test p) | Seconds per case |
|---|---|---|---|---|
| Rule-based analysis (the old engine) | 64/146 | **0.44** (0.36–0.52) | – | – |
| Text retrieval over the knowledge base (no language model) | 136/146 | **0.93** (0.88–0.96) | 76 / 4 | – |
| Random pick from the shortlist | 32/146 | 0.22 (0.16–0.29) | 16 / 48 | – |
| Qwen3-1.7B, no evidence | 20/146 | 0.14 (0.09–0.20) | 0 / 44 | 0.7 |
| Qwen3-1.7B, investigation | 53/146 | 0.36 (0.29–0.44) | 9 / 20 (p = 0.06) | 2.3 |
| Qwen3-1.7B 4-bit, investigation | 30/146 | 0.21 (0.15–0.28) | 13 / 47 | 2.7 |
| Qwen3-4B, no evidence | 21/146 | 0.14 (0.10–0.21) | 11 / 54 | 2.0 |
| **Qwen3-4B, investigation** | 128/146 | **0.88** (0.81–0.92) | **70 / 6** (p < 0.001) | 9.1 |
| Qwen3-8B, no evidence | 22/146 | 0.15 (0.10–0.22) | 0 / 42 | 3.2 |
| **Qwen3-8B, investigation** | 129/146 | **0.88** (0.82–0.93) | **72 / 7** (p < 0.001) | 14.7 |

Full table (including every configuration's tool usage) in [`artifacts/agent_comparison.json`](artifacts/agent_comparison.json);
`python scripts/format_comparison.py` prints it as Markdown.

**What this shows**

* **The investigation is what works, and it needs a big enough model.** Without evidence every model is at chance
  (0.14–0.15, the shortlist is shuffled), with it Qwen3-4B and 8B reach 0.88 and beat the rule engine in 70 and 72 cases
  while losing 6 and 7. **Qwen3-1.7B, the model the chat used before, is not good enough**: it stays below the rule engine
  (0.36), and in 4-bit it behaves differently again (it reads the logs, decides immediately, and scores 0.21).
* **4B is the sweet spot.** 8B is no more accurate (0.88 vs 0.88) and 1.6× slower. The 4B model uses the same five tools on
  every case (read logs, text match, similar cases, event evidence, structural check) and never asks the rule engine; the 8B
  model varies its route more (node correlation in 68 cases, repeat pattern in 32). The 1.7B model asks the rule engine in
  86% of cases and mostly follows it, which is why it does no better than the rules.
* **On BGL a plain text-retrieval baseline is as good.** Matching the window's message templates to the knowledge-base descriptions
  with TF-IDF, no language model at all, scores 0.93 (0.88–0.96), nominally above the 4B and 8B investigators (the intervals
  overlap). The agents use that same match as one of their tools, in every case. So what the investigator adds over the
  retrieval baseline is not accuracy on this task but a recorded, step-by-step investigation (readable in the UI), the ability
  to consult correlations, timings and logs when they matter, and a decision that can be audited.
  In a repeat run of the 4B configuration (which reproduced 128/146 exactly, so the pipeline is deterministic; per-case choices in
  [`artifacts/agent_per_case_4b.json`](artifacts/agent_per_case_4b.json)), the investigator picked the retrieval tool's top
  suggestion in 128 of 146 cases (88%) and was right in 122 of them. In the 18 cases where it chose differently it gained 2
  that retrieval got wrong and lost 10 that retrieval got right. On this task the model mostly reads and relays the retrieval
  tool's answer, and its own deviations cost more than they gain.
* **On OpenStack the rule engine wins, and the language models make a consistent mistake.** All four injected anomalies share one
  event pattern, so they are one investigated case (a sample of one). The rule engine names the slow spawn (correct: the
  spawn took 41.8 s against a normal maximum of 26.7 s). The text-retrieval baseline and every investigation (1.7B, 4B and
  8B, with 0.78–0.80 confidence for 4B and 8B) choose "slow build" instead: the log line "Took 42.8 seconds to build
  instance" matches that cause's wording, although its description says the spawn was *not* the slow part, which the
  evidence contradicts. They read the words and miss that the build includes the spawn. (Qwen3-4B without any evidence
  "passes" by chance: 0.35 against 0.24 for the next candidate.) An earlier version of this comparison accepted slow build as
  correct and showed 4/4 everywhere; it was too lenient and is not used.
* **HDFS** has no root-cause ground truth, so it is not scored. The investigator runs on it (tested with a real model) and
  the Investigation tab shows what it did.

**How the design was chosen (disclosure).** A first 20-case pilot with Qwen3-1.7B showed two problems: the rule engine
offered a single candidate whenever a window contained message templates it had never seen, and the model answered "A" almost
always. I then added the knowledge-base text match (shortlist padding and an optional tool) and the position-bias fixes, and chose
between three decision methods on a separate development set of 59 windows from the *training* period (a different seed,
Qwen3-1.7B only): plain listwise 0.29, pointwise 0.46, debiased listwise 0.51. The final numbers above use a different seed and
the held-out period. They were not tuned further, but the pilot cases were seen during design, the decision method was
chosen on the 1.7B model only, and the same five-step budget and prompts were used for all models. BGL's root-cause
signatures were first built from the whole log; they now come from the training windows only, so the rule baseline does not
see the test period's labels.

## The three datasets

| | HDFS | BGL (BlueGene/L) | OpenStack |
|---|---|---|---|
| Source | LogHub HDFS_v1 | LogHub BGL | LogHub / DeepLog OpenStack |
| A **session** is | a block (`blk_…`) | a window of 100 consecutive messages (`win_N`) | a VM instance (UUID in `[instance: …]`) |
| Input formats | raw lines · `Event_traces.csv` · event sequences | raw RAS lines | raw nova lines |
| Events | 29 hand-written templates | 510 templates mined with Drain | 23 templates mined with Drain |
| Labels | per block | per line (alert tag); a window is anomalous if it holds an alert | 4 instances with injected delays |
| Features the Random Forest reads | embedding + neighbours | severity counts + bag-of-words of the messages (**only these**) | embedding + neighbours + the four `Took N seconds` timings |
| Root causes the investigator chooses from | 10 hand-written HDFS causes | 28 alert categories learned from the training labels + family text | timing, incomplete-spawn, error / warning causes |
| Models | repo root (`transformer_backbone.pth`, …) | `artifacts/bgl/` | `artifacts/openstack/` |

The Identification Agent picks the dataset from the log lines (or you force one). Everything downstream is written once
against a **dataset profile** (`log_agents/datasets/`) that supplies what differs: parsing, sessions, event meaning,
model artifacts and root causes.

## Repository layout

```
streamlit_app.py            web UI (Streamlit)
log_agents/
  agents/                   identification, preprocessing, anomaly, investigator (LLM), rca (rule engine + tools), insight, coordinator
  datasets/                 one profile per dataset (HDFS, BGL, OpenStack): parsing, sessions, events, root causes
  slm.py                    the local Qwen model: constrained choice (investigator) and chat, 4-bit loading
  chat.py                   follow-up chat, fed the relevant raw log lines
  drain.py                  template miner used for BGL and OpenStack
  medallion.py              Bronze / Silver / Gold layers
  ml.py                     Transformer encoder, FAISS retrieval, Random Forest scoring
  knowledge.py              HDFS root-cause knowledge base
scripts/                    train_dataset.py (BGL, OpenStack), compare_agents.py + format_comparison.py (the model comparison),
                            download_models.py, build_rag_metadata.py (HDFS), sample builders
artifacts/bgl, artifacts/openstack   trained models, metrics.json, templates;  artifacts/agent_comparison.json
(repo root)                 HDFS models: transformer_backbone.pth, rag_random_forest.joblib, hdfs_faiss_index.index (Git LFS)
samples/                    demo inputs for all three datasets
tests/                      53 tests
OtherTests/, Baselines/     experiment notebooks and baseline models (not part of the pipeline)
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_models.py Qwen/Qwen3-4B      # 8 GB; the recommended investigator (see the comparison)
streamlit run streamlit_app.py
```

`python scripts/download_models.py` with no argument downloads all three (1.7B 4 GB, 4B 8 GB, 8B 16 GB) into
`~/.cache/lad_models/`, outside the project folder. Nothing is downloaded implicitly: a model whose weights are missing
makes the investigator fall back to the rule-based analysis and say so. The 4-bit models need a CUDA GPU
(`bitsandbytes` is in `requirements.txt`).

In the UI pick the **Investigator** under *Agent settings* (the Investigation tab shows each case's steps). Command line:

```bash
python -m log_agents.cli samples/bgl_raw_sample.log --agent-model Qwen/Qwen3-4B     # dataset detected automatically
python -m log_agents.cli samples/openstack_raw_sample.log --agent-model rules --out report.md   # rule-based, no model
python -m log_agents.cli my.log --dataset bgl --chat                                # then ask follow-up questions
python -m pytest tests -q                                                           # 53 tests, fast, no GPU needed
python scripts/compare_agents.py --cases 20 --models Qwen/Qwen3-4B                  # a quick comparison pilot
```

### Getting the data and training BGL / OpenStack

The datasets are not committed (they are in `.gitignore`). Download **BGL.zip** (57 MB) and **OpenStack.tar.gz** (5 MB) from the
[LogHub release](https://zenodo.org/records/8196385) and extract them to `Dataset_BGL/` (`BGL.log`) and
`Dataset_OpenStack/` (`openstack_*.log`, `anomaly_labels.txt`). HDFS_v1 (`HDFS_v1.zip`, 187 MB, same release) is only needed to
rebuild the HDFS samples or metadata; point `LAD_DATASET_DIR` at its `preprocessed/` folder. Then:

```bash
python scripts/train_dataset.py --dataset bgl         # ~2.5 min on a GPU
python scripts/train_dataset.py --dataset openstack   # ~15 s
```

The trained artifacts (13 MB and 1 MB) are committed in `artifacts/`, so training is only needed to reproduce them.
Training uses the same cleaning, Silver and Gold code as inference and evaluates through the real `AnomalyPipeline`.
HDFS artifacts come from `OtherTests/RAG__Trans_RF.ipynb`; `scripts/build_rag_metadata.py` rebuilds
`rag_metadata.npz` and `event_stats.json` for them.

## The chat

The same Qwen model that investigated (or Qwen3-1.7B after a rule-based run) answers questions about a finished run. It is given:

* the run's **facts sheet** (root causes, evidence, similar historical sessions, normal examples, and what the investigator
  did) and the analysis the Insight Agent wrote, in the system prompt;
* **block / window / instance details** for every session the question names;
* **raw log lines** chosen for the question: the lines of a named session (error and warning lines first, capped),
  the lines of an event id such as `E12`, the lines containing the question's keywords, or, for a general question,
  the lines of the most anomalous sessions.

Answers are not blocked. Anything the run cannot back up (unknown session ids, event ids, node names / IPs, or
numbers that appear neither in the facts nor in the log lines the model was shown) is flagged under the answer.
Inputs without raw lines (HDFS event sequences, the trace CSV) get no log section.

## Detection results

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
similar windows (evidence for the investigation and the chat) and for the uncertainty flag. The labels
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

**What changed after the first detection results (disclosure).** Both models were first built with a shared design and
scored BGL F1 0.62 and OpenStack F1 0.86. I then changed, after seeing those results: (BGL) added word features and let the
forest read only them; (OpenStack) made the forest see every feature and sample (`bootstrap=False`, `max_features=None`,
because there are only 4 positives and the 6 informative columns were rarely sampled) and removed session duration and
longest gap from the model input, because they depend on how long a VM was kept alive. These are model-design choices
tested on the same evaluation data, so the final numbers are somewhat optimistic compared with a design fixed in advance.

## Configuration (environment variables)

`LAD_AGENT_MODEL` (default `Qwen/Qwen3-4B`; `rules` = no language model), `LAD_CHAT_MODEL` (chat model after a rule-based
run, default `Qwen/Qwen3-1.7B`), `LAD_MODELS_DIR` (default `~/.cache/lad_models`), `LAD_ALLOW_DOWNLOAD=1` (let Hugging Face
download missing weights), `LAD_AGENT_MAX_STEPS` (5), `LAD_AGENT_MAX_CASES` (30), `LAD_AGENT_DECISION`
(`listwise_debiased` | `pointwise` | `listwise`), `LAD_ARTIFACTS_DIR` (HDFS), `LAD_ARTIFACTS_ROOT` (BGL / OpenStack),
`LAD_DATASET_DIR` (HDFS_v1), `LAD_BGL_DIR`, `LAD_OPENSTACK_DIR`, `LAD_DATA_DIR`, `LAD_CHAT_LOG_LINES`, `LAD_CHAT_LOG_CHARS`.
See `log_agents/config.py`.

## Limitations

* **The agent is only as good as its model, and on this task not better than a text match.** Qwen3-1.7B is worse than the
  rule engine; 4B and 8B are far better than the rules but not better than the TF-IDF baseline. The 4B model's route is
  nearly the same on every case, so "it decides its own steps" is true of the mechanism, and much less true of the behaviour.
* **The investigator can be wrong with confidence.** On OpenStack all language-model investigations picked a cause whose own
  description contradicts the evidence, and the rule engine was right. Evidence tools that state the relation explicitly
  ("the build includes the spawn") would likely fix this case; I did not change the tools after seeing it, to avoid tuning
  on a sample of one.
* **The shortlist is not the model's.** Candidates come from the rule engine and a knowledge-base text match, so the model
  can only choose among them (they contain the truth in 96% of BGL cases). It cannot discover a cause the knowledge base does
  not describe.
* **Root-cause ground truth exists only for BGL** (alert tags, assigned by external expert rules) **and a single
  OpenStack case.** HDFS has none, so nothing can say whether its causes are right. The per-hypothesis percentage is a
  share of heuristic rule scores, not a probability.
* **The investigator needs a GPU and the downloaded weights**; without them it falls back to the rules. 4-bit
  quantisation hurts the 1.7B model; the 4B and 8B models were only tested in 4-bit. Per-case cost on the RTX 4060: about
  2 s (1.7B), 9 s (4B), 15 s (8B).
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
* **GPU memory.** Qwen3-8B in 4-bit takes 6.1 GB of an 8 GB card (the embedding and output layers are not quantised). The
  investigator prefills prompts in chunks (`CHOOSE_CHUNK`); without that a 1.7k-token prompt pushed the card into Windows'
  shared memory and one case took 165 s instead of 15 s. Chunking changes the matmul shapes, so scores shift slightly
  (identical at the default chunk of 384 on short prompts; up to ~0.07 in probability on a near-tied prompt at a chunk of 64).
* Medallion layers are written to `data/medallion/<run_id>/` and run summaries to `data/runs/`.
* `pre-processing/preprocess.py` counts an event once per block per line; before, E21 lines (which contain the block
  id twice) were double-counted, so its output did not match the labelled dataset.
