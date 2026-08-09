# Nearmiss

Finds rising risk patterns in aviation near-miss narratives — the free-text safety reports
crews file after something almost went wrong — and proves the pattern was detectable before
the incident it preceded.

An agent running on Featherless decides what to investigate and calls eight tools to do it.
**Every number in the output is computed by Python from the report database. Models read
narratives, name patterns and write prose. No model ever produces a statistic.**

---

## Run it in five minutes

```bash
pip install -r requirements.txt -r requirements-ingest.txt
python make_seed.py                          # synthetic corpus, works offline
python ingest.py --csv data/seed_reports.csv --clusters 14
export FEATHERLESS_API_KEY=sk-...
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 and hit a preset. `GET /api/health` tells you whether your three
models are actually live on Featherless and how large the catalog is.

## Deploy to Render

Push to GitHub, then New → Web Service → connect the repo. `render.yaml` is already here, or
set it manually:

| Field | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Env var | `FEATHERLESS_API_KEY` |

Three things that will otherwise cost you the demo:

1. **Commit `data/nearmiss.db`.** Ingestion uses scikit-learn and will exhaust the free tier's
   512 MB. Build the database on your laptop, commit it, ship it. `requirements.txt` has no
   scikit-learn in it on purpose.
2. **Bind `0.0.0.0` and `$PORT`** exactly as above or Render's health check fails.
3. **Load the URL five minutes before you present.** The free tier sleeps and cold-starts in
   about 50 seconds.

## Where the models are used

One Featherless key, three model families, because the roles want different things.

| Role | Default | Why |
|---|---|---|
| `AGENT_MODEL` | `Qwen/Qwen3-32B` | Native tool calling. This is the one that must be reliable. |
| `EXTRACT_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Cheap, fast, called per report. |
| `VERIFY_MODEL` | `mistralai/Mistral-Small-24B-Instruct-2501` | Different lineage from the agent, so it can actually disagree. |

All three are environment variables. If `/api/health` reports one unavailable, swap the string
— that is the entire cost of changing models here, and it is the reason this runs on Featherless
rather than a single-vendor API.

Set `FEATHERLESS_CONCURRENCY` to your plan's simultaneous-request cap minus one. `llm.py`
holds a semaphore at that value; without it a fan-out returns a wall of 429s.

## The eight tools

| Tool | Provenance |
|---|---|
| `search_reports` | SQLite FTS5 |
| `cluster_incidents` | SQL |
| `compute_trend` | **Python only** — monthly counts, rate ratio, least-squares slope, exact binomial tail p-value |
| `compare_baseline` | **Python only** — same cluster at every other field |
| `backtest_asof` | **Python only** — recomputes a trend using data strictly before a cutoff month |
| `get_airport_context` | SQL |
| `extract_causal_chain` | model reads one narrative → structured JSON |
| `verify_finding` | independent model family checks the conclusion against sources |

`compute_trend` treats the recent window as a binomial draw conditional on the window total,
so the p-value is exact and needs no scipy. Read `tools.py::_binom_tail`.

## Real data

The seed corpus is **synthetic** and says so. It exists so the app works before you have
anything. Real reports come from NASA's Aviation Safety Reporting System:

<https://asrs.arc.nasa.gov/search/database.html>

Export CSV, then `python ingest.py --csv asrs_export.csv --asrs`. ASRS ships two header rows
and renames columns periodically; `COLUMN_GUESSES` in `ingest.py` handles the common variants
and prints the headers it saw if it can't find a narrative column.

## What the seed data contains

Two patterns are planted, and the system finds both without being told they exist:

- **KSFO, parallel-approach runway confusion.** Ramps from April 2025. Measured rate ratio
  4.4×, p = 0.00004, while peer airports sit near 1.1× — so it is local, not a national
  reporting shift.
- **KBOS, hot brakes on short turnarounds.** Ramps into May 2024, then drops after a fix.
  `backtest_asof(cutoff_ym="2024-05")` uses data only through April 2024 and returns
  ratio 4.8×, p = 0.0019, `would_have_flagged: true`.

## Demo, 90 seconds

1. Ask *"Is anything getting worse at SFO?"* Let the strips fill the bay — the judges watch it
   choose what to investigate.
2. Point at a blue strip: this arithmetic is Python, not a model.
3. Point at `compare_baseline`: the reason it isn't fooled by a nationwide reporting change.
4. Ask *"Was the BOS brake-temperature pattern detectable before May 2024?"* and land on
   **would have flagged, one month early, on data that existed at the time.**

## Honest limits

Synthetic seed data proves the pipeline, not the science — swap in real ASRS before claiming
a result. Clustering is TF-IDF + MiniBatchKMeans, which merges themes that share vocabulary;
raise `--clusters` if a pattern looks diluted. A rising report count can mean rising risk or
rising reporting, which is exactly why `compare_baseline` exists and why the agent is
instructed to say so. This is a prototype, not an operational safety tool.

