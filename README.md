# MediCore - Automated ICD-10-CM Medical Coding

MediCore reads a free-text clinical note and assigns the corresponding
**ICD-10-CM** diagnosis codes using a **local Ollama model** and a
**retrieve-then-assign** (retrieval-augmented generation) pipeline.

Because the 2026 ICD-10-CM set has ~75,000 assignable codes - far too many to
put in any prompt - MediCore first retrieves a small candidate set for each
note, then asks the LLM to select the supported codes from that set. This keeps
the model responsible only for judgement over ~120 candidates while an index
handles recall over the full code space.

> Full design rationale, trade-offs, assumptions, results, and a roadmap are in
> **[`docs/MediCore_Design_Writeup.pdf`](docs/MediCore_Design_Writeup.pdf)**.

---

## How it works (short version)

```
note --> 1. concept extraction (LLM, JSON)
      --> 2. hybrid retrieval  (BM25 + embeddings, fused with RRF)  --> ~120 candidates
      --> 3. code assignment   (LLM picks from candidates, JSON)
      --> 4. validation        (normalize, drop hallucinated codes)  --> ICD-10-CM codes
```

The code knowledge base (both indexes) is built once and cached under `.cache/`.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Developed on 3.12. |
| **Ollama** | Install from <https://ollama.com>. Must be running (`ollama serve`). |
| **Disk** | ~2.5 GB for the models; ~220 MB for the cached embeddings. |
| **Data** | `data/icd10cm_order_2026.txt` (CMS order file) and `data/icd10_cm_cases.json` (labelled cases). |

Pull the models used by default:

```bash
ollama pull qwen2.5:3b          # chat model (extraction + assignment)
ollama pull nomic-embed-text    # embedding model (semantic retrieval)
```

---

## 2. Setup

```bash
# from the project root
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# macOS/Linux:           source .venv/bin/activate

pip install -r requirements.txt
```

Make sure Ollama is up:

```bash
ollama serve            # if not already running as a service
ollama list             # should list qwen2.5:3b and nomic-embed-text
```

---

## 3. Configuration

All settings live in **[`config.yaml`](config.yaml)**. The key section:

```yaml
ollama:
  base_url: "http://localhost:11434"   # Ollama server URL
  api_key: ""                          # optional; see note below
  llm_model: "qwen2.5:3b"
  embed_model: "nomic-embed-text"
  temperature: 0.0                     # deterministic coding
  num_ctx: 8192

retrieval:
  use_embeddings: true      # false => BM25-only (no embedding model needed)
  final_candidates: 120     # candidates shown to the assignment LLM
  billable_only: true       # only assignable codes are candidates

pipeline:
  extract_concepts: true    # LLM concept-extraction stage (recommended)
```

**About the API key.** A standard local Ollama install needs **no key** - leave
`api_key` blank. The field exists so the *same* code can target a secured or
remote endpoint (Ollama behind a reverse proxy, Ollama Turbo, or an
OpenAI-compatible gateway); when set it is sent as `Authorization: Bearer <key>`.
You can also override without editing the file:

```bash
export OLLAMA_API_KEY="sk-..."          # takes precedence over config.yaml
export OLLAMA_BASE_URL="https://my-ollama.example.com"
```

---

## 4. Running

### Evaluate on the labelled dataset (recall, time, tokens)

```bash
python scripts/run_eval.py
```

First run builds and caches the code embedding index (a one-time step over
~75k codes; several minutes on CPU). Subsequent runs load the cache instantly.

Useful flags:

```bash
python scripts/run_eval.py --limit 10                      # quick smoke run on 10 cases
python scripts/run_eval.py --no-embeddings                 # BM25-only retrieval
python scripts/run_eval.py --model llama3.1:8b --prefix eval_llama   # benchmark another model
```

Output:
- Console summary (recall / precision / F1, retrieval recall, avg time, avg tokens).
- `reports/eval_summary.json` - machine-readable metrics + per-case detail.
- `reports/eval_per_case.csv` - one row per case.

### Code a single note

```bash
python scripts/code_note.py --file path/to/note.txt
# or pipe it:
echo "A 45-year-old male presented with ..." | python scripts/code_note.py
```

Prints the extracted concepts, the assigned codes with descriptions, and a
short justification per code.

### Regenerate the design PDF (with the latest measured metrics)

```bash
python docs/make_diagram.py          # architecture figure
python docs/make_charts.py           # chart suite (reads reports/)
python docs/generate_writeup.py      # -> docs/MediCore_Design_Writeup.pdf
```

The write-up automatically embeds the numbers from `reports/eval_summary.json`
(and `reports/eval_llama_summary.json` for the model-comparison section), so run
the evaluation first if you want the results section populated.

---

## 5. Metrics reported

| Metric | Meaning |
|---|---|
| **Recall** (primary) | Fraction of gold codes correctly assigned. Micro (code-weighted) and macro (note-weighted). |
| Precision / F1 | Reported for context. |
| **Retrieval recall** | Fraction of gold codes that appeared in the retrieved candidate pool - the ceiling the assignment stage can reach. |
| **Avg time / note** | Wall-clock seconds per note, including retrieval. |
| **Avg tokens / note** | Prompt + completion tokens, read from Ollama's `prompt_eval_count` / `eval_count`. |

### Measured results on the provided dataset (54 cases)

Same pipeline (hybrid retrieval + concept extraction), two Ollama chat models.
The retrieval ceiling is identical because retrieval is unchanged - the
difference is entirely the assignment model's specificity judgement.

| Metric | `qwen2.5:3b` (baseline) | `llama3.1:8b` |
|---|---|---|
| **Recall - micro / macro** | **0.190 / 0.256** | **0.354 / 0.433** |
| Precision - micro | 0.155 | 0.308 |
| F1 - micro | 0.170 | 0.329 |
| Retrieval recall (ceiling) | 0.754 | 0.754 |
| **Avg time / note** | **23.1 s** | **24.3 s** |
| **Avg tokens / note** | **3,533** (3,275 + 259) | **3,134** (2,891 + 242) |

Key insight: retrieval surfaces ~75% of gold codes, but the small model selects
only ~19% correctly; the larger model nearly doubles recall at the same latency
and fewer tokens. The remaining gap is a model-capability limit, not a pipeline
limit. Full analysis and figures are in the PDF write-up (Section 6). All numbers
are measured on CPU and reproduced by `scripts/run_eval.py`.

---

## 6. Project layout

```
medicore/
|-- config.yaml                 # all settings (models, retrieval, paths)
|-- requirements.txt
|-- README.md
|-- data/
|   |-- icd10cm_order_2026.txt  # CMS ICD-10-CM order file (all codes)
|   \-- icd10_cm_cases.json     # 54 labelled cases (note + gold codes)
|-- medicore/                   # library
|   |-- config.py               # config loading
|   |-- data.py                 # parse order file + cases
|   |-- kb.py                   # code knowledge base + embedding cache
|   |-- retrieval.py            # BM25 + embeddings + RRF fusion
|   |-- llm.py                  # Ollama client (chat/embeddings) + telemetry
|   |-- extract.py              # stage 1: concept extraction
|   |-- assign.py               # stage 3: assignment + validation
|   |-- prompts.py              # prompt templates
|   |-- pipeline.py             # end-to-end orchestration
|   \-- evaluate.py             # metrics + report writers
|-- scripts/
|   |-- run_eval.py             # evaluate over the dataset
|   \-- code_note.py            # code a single note
|-- docs/
|   |-- make_diagram.py         # architecture figure
|   |-- make_charts.py          # chart suite
|   |-- generate_writeup.py     # build the design PDF
|   \-- MediCore_Design_Writeup.pdf
\-- reports/                    # generated metrics (git-ignored)
```

---

## 7. Troubleshooting

- **`Ollama not reachable`** - start the server (`ollama serve`) and confirm
  `curl http://localhost:11434/api/tags` responds.
- **Model not found** - `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text`.
- **First run is slow** - that's the one-time embedding index build; it is
  cached to `.cache/` and reused thereafter. Delete `.cache/` to rebuild, or run
  with `--no-embeddings` to skip it entirely.
- **Out of memory / slow CPU** - set `retrieval.use_embeddings: false` for
  BM25-only retrieval, or lower `retrieval.final_candidates`.
