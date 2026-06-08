# ux-evidence-graph

Turn a large pile of unstructured UX text into a **reviewable evidence graph** that helps you answer higher-level reasoning questions — running entirely **locally** with [LM Studio](https://lmstudio.ai/).

UX researchers often have a huge pile of text: interview transcripts, survey comments, support tickets, app reviews, usability notes, and old research reports. A plain summary is not enough, because the important insight usually lives in the **relationships between pieces of evidence**. This tool maps concepts, claims, pain points, user groups, behaviors, outcomes, and the links between them, so you can ask questions that are hard to answer from raw text alone.

> This is a lightweight **research prototype**, not a validated automated analysis tool. It is a **direction-sign system**: it suggests possible connections, and **you** review, approve, reject, edit, reweight, and correct them.

---

## What this project does

- Reads `.txt`, `.md`, `.csv` (and optionally `.pdf`) files.
- Splits documents into overlapping chunks.
- Uses a **local** LLM (via LM Studio) to extract a structured graph of **nodes** and **edges**.
- Stores everything in a local **SQLite** file.
- Lets you **review** every suggested connection through plain **CSV files** (Excel, Numbers, Google Sheets).
- Answers higher-level questions using **hybrid retrieval**: semantic text search **plus** graph search.
- Grounds every answer in retrieved evidence and **cites** documents and chunks.
- Exports the graph to JSON, GraphML, CSV tables, and a readable markdown summary.

## What this project does NOT do

- It is **not** a web app. There is no UI, no Streamlit/Gradio/Dash/Flask/React. Everything runs through Python scripts, CLI commands, config files, CSV review files, and importable modules.
- It does **not** require any cloud API, paid API, Neo4j, or external database server.
- It does **not** treat the model's output as truth. The graph points you toward possible reasoning paths; **humans validate the links**.

## Why structure matters

A summary flattens your data. A graph keeps the *relationships*: which pain point reduces trust, which behavior leads to drop-off, which complaint is evidenced across multiple documents. Structure is what lets you ask "what are the hidden chains connecting user behavior to business outcomes?" instead of just "what did people say?"

## Why human review matters

Local models are helpful but imperfect. They will over-connect, mislabel, and occasionally hallucinate relationships. Treating model output as ground truth is dangerous for research. That is why **every model-created edge starts as `pending`** and nothing is trusted until a human approves it. Rejected edges are never used in retrieval or answers.

---

## The pipeline

```
Raw UX text
   → chunks
   → structured extraction (nodes + edges)
   → human review (CSV)
   → vector retrieval + graph search (HybridRAG / GraphSearch)
   → evidence-grounded answer
```

1. **Raw text becomes chunks.** Long documents are split into ~900-word overlapping chunks.
2. **Chunks become structured nodes and edges.** The local model extracts a typed graph and must cite evidence text for every edge.
3. **Researchers review the connections.** Export edges to CSV, edit `review_status` / `weight` / `confidence`, import back.
4. **The system uses two retrieval paths.** Path A: semantic vector search over chunks. Path B: graph search over nodes and edges.
5. **The answer is generated from retrieved evidence, not from memory.** The model is told to use only the provided context, distinguish strong vs weak evidence, flag pending links, and cite sources.
6. **The graph suggests reasoning paths; humans validate them.**

---

## Installation

Requires **Python 3.11+**.

```bash
git clone <your-repo-url> ux-evidence-graph
cd ux-evidence-graph

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> The first time you use the sentence-transformers fallback, the `all-MiniLM-L6-v2` model (~90 MB) is downloaded automatically.

## Start the LM Studio local server

1. Install and open **LM Studio**.
2. Download and **load a chat model** (any instruct model works; a model good at JSON output is ideal).
3. Optionally load an **embedding model** (otherwise the local sentence-transformers fallback is used).
4. Open the **Local Server / Developer** tab and click **Start Server**.
5. Confirm the server URL is `http://localhost:1234/v1` (the default).

Check connectivity at any time:

```bash
python main.py health
```

## Configure model names

Copy the example env file and set the model ids you loaded in LM Studio:

```bash
cp .env.example .env
```

```dotenv
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_CHAT_MODEL=your-loaded-chat-model
LMSTUDIO_EMBEDDING_MODEL=your-loaded-embedding-model
EMBEDDING_PROVIDER=lmstudio        # or sentence_transformers
```

Project-wide defaults live in `config.yaml`. Anything set in `.env` (or real environment variables) overrides the LM Studio / embedding settings at runtime, so secrets and per-machine model names stay out of git.

---

## Quick start (with the included sample data)

```bash
pip install -r requirements.txt
python main.py init
python main.py ingest --input data/sample
python main.py extract
python main.py export-review --output outputs/review/edges_for_review.csv
python main.py ask --question "Which pain points are connected to trust?"
```

The repo ships with fake but connected UX data in `data/sample/` (onboarding confusion, pricing uncertainty, lack of trust, extra verification behavior, support contact, drop-off risk) so you can try the whole flow immediately.

---

## Commands

| Command | What it does |
| --- | --- |
| `python main.py init` | Create the SQLite database and tables. |
| `python main.py health` | Check that LM Studio is reachable. |
| `python main.py ingest --input data/raw` | Read + chunk documents from a folder. |
| `python main.py extract` | Extract nodes/edges from chunks via LM Studio. |
| `python main.py embed` | Create embeddings for all chunks. |
| `python main.py export-review --output outputs/review/edges_for_review.csv` | Export edges to a review CSV. |
| `python main.py import-review --input outputs/review/edges_for_review.csv` | Apply your edits to edges. |
| `python main.py export-nodes-review` / `import-nodes-review` | Review nodes via CSV. |
| `python main.py suggest-merges` | Suggest near-duplicate nodes to merge. |
| `python main.py import-merges` | Apply approved merges. |
| `python main.py ask --question "..."` | Ask a higher-level question over the graph. |
| `python main.py export-graph --format graphml` | Export `graph.graphml`. |
| `python main.py export-graph --format json` | Export `graph.json`. |
| `python main.py export-tables` | Export `nodes.csv`, `edges.csv`, `adjacency.csv`. |
| `python main.py export-summary` | Write `outputs/graphs/graph_summary.md`. |
| `python main.py export-image` | Optional static PNG of the graph (matplotlib). |

### How to review edges using CSV

1. Run `python main.py export-review`.
2. Open `outputs/review/edges_for_review.csv` in any spreadsheet editor.
3. Change any of these columns:
   - `review_status` → `pending`, `approved`, `rejected`, or `edited`
   - `weight`, `confidence` (0–1)
   - `relation_type` (must be an allowed edge type)
   - `description`, `review_note`
4. Save and run `python main.py import-review`.

Approved edges are prioritized in retrieval; rejected edges are never used.

---

## Python API

The whole pipeline is importable:

```python
from uxeg import EvidenceGraphProject

project = EvidenceGraphProject.from_config("config.yaml")
project.init()
project.ingest("data/raw")
project.extract()
project.export_review("outputs/review/edges_for_review.csv")

# ...review the CSV, then:
project.import_review("outputs/review/edges_for_review.csv")

answer = project.ask("Which pain points are connected to trust?")
print(answer)
```

---

## Schema

**Node types:** `USER_GROUP, PARTICIPANT, TASK, FEATURE, PAIN_POINT, CAUSE, BEHAVIOR, EMOTION, OUTCOME, METRIC, RECOMMENDATION, CLAIM, EVIDENCE, DECISION, CONCEPT`

**Edge types:** `EXPERIENCES, STRUGGLES_WITH, CAUSES, LEADS_TO, INCREASES, REDUCES, SUPPORTS, CONTRADICTS, APPEARS_IN, ADDRESSES, RELATED_TO, PART_OF, EVIDENCED_BY`

**Review statuses:** `pending` (default for model output), `approved`, `rejected`, `edited`.

Each edge carries its `evidence_text`, `source_doc`, `source_chunk`, `confidence`, and `weight`, so every connection is traceable back to the original text.

---

## Example workflow

```bash
# 1. Set up
python main.py init

# 2. Drop your files into data/raw (or use data/sample) and ingest
python main.py ingest --input data/raw

# 3. Extract structure (LM Studio must be running)
python main.py extract

# 4. De-duplicate concepts
python main.py suggest-merges
#    edit approve_merge -> yes for accepted merges
python main.py import-merges

# 5. Review connections
python main.py export-review
#    edit review_status / weight / confidence
python main.py import-review

# 6. Ask questions
python main.py ask --question "Which complaints may be symptoms of the same underlying issue?"

# 7. Export for sharing / external graph tools
python main.py export-graph --format graphml
python main.py export-summary
```

See `data/sample/questions.txt` for more example questions.

---

## Tests

```bash
pytest
```

Tests cover chunking, schema validation, SQLite node/edge insertion, review CSV round-trip, graph construction, and retrieval — none of them require LM Studio.

---

## Error handling

The tool gives clear, actionable messages for common situations:

- LM Studio server not running → connection checklist.
- Chat/embedding model name missing → tells you which env var to set.
- Invalid JSON from the model → one automatic repair retry, then the raw response is saved to `outputs/errors/` and processing continues.
- Empty file / unsupported file → skipped with a warning.
- No chunks found → reminds you to ingest first.
- Embedding model unavailable in LM Studio → automatic fallback to sentence-transformers.
- Database not initialized → reminds you to run `python main.py init`.

---

## Limitations

- Extraction quality depends on your local model. Smaller models over-connect and mislabel.
- It is **not** a truth machine. Treat all unreviewed edges as hypotheses.
- Normalization is conservative; it suggests merges but does not aggressively combine concepts.
- Vector and keyword retrieval are intentionally simple (cosine similarity, token overlap).
- The graph reflects what the model found in the text, not objective reality.

## Future improvements

- Smarter entity resolution / clustering for normalization.
- Per-relation confidence calibration.
- Incremental re-extraction and change tracking.
- Optional reranking of retrieved chunks.
- Richer contradiction and bias detection.

---

## Author

Created and maintained by **Mohsen Rafiei, Ph.D.**

If you use this project in your research or share it with colleagues, a mention is appreciated.

## License

MIT © 2026 Mohsen Rafiei, Ph.D. — see [LICENSE](LICENSE).
