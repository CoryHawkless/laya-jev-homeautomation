# Home-automation demo

The floor-plan UI and 200-case benchmark — pure clients of a `laya-serve`
`/v1/systemone` endpoint. No custom server code; the UI is a static HTML
file that calls the inference server over HTTP.

## Pieces

| File | What it is |
|---|---|
| `index.html` | Floor-plan UI — type a command, watch the rooms light up |
| `home_questions.py` | The `QUESTIONS` schema (shared by the UI mirror and the bench) |
| `cases.jsonl` | 200 labelled utterances |
| `run_bench.py` | Benchmark runner → `results/{raw.jsonl, summary.json, report.md}` |
| `Caddyfile` | Optional reverse proxy: serves the UI + proxies `/v1/systemone` (solves CORS) |

## Run the UI

The UI is a static HTML file. It needs to call `/v1/systemone` on a
`laya-serve` instance. Two ways to make that work:

### Option A — Caddy reverse proxy (recommended for remote laya-serve)

Serves the UI and proxies `/v1/systemone` + `/health` to a remote
`laya-serve` at the same origin — no CORS issues in the browser.

```bash
LAYA_UPSTREAM=epyc02:8000 caddy run --config Caddyfile
# open http://localhost:8080
```

### Option B — Same host (local dev)

Run `laya-serve` on `localhost:8000`, then open `index.html` directly in a
browser. Set `SERVER_URL` at the top of the `<script>` block to
`http://localhost:8000`. (Browsers will block cross-origin without CORS
headers from laya-serve; use Option A for any non-local setup.)

## Run the benchmark

```bash
pip install -r requirements.txt
python run_bench.py                                     # local laya-serve
python run_bench.py --router http://epyc02:8000         # remote
python run_bench.py --limit 20                          # smoke test
python run_bench.py --out ./runs/gpu-run-1              # custom output dir
```

Output lands in `results/`:
- `raw.jsonl` — per-case predictions + timing
- `summary.json` — aggregate metrics, confusion matrix, failure list
- `report.md` — human-readable report

## Sample output

```
$ python run_bench.py --router http://epyc02:8000 --limit 5
Loaded 5 cases from cases.jsonl
Waiting for http://epyc02:8000/health ...
  ready: device=cuda loaded=['english']
  5/5  (3.2 req/s)

Intent accuracy:         100.0%
All fields correct:      80.0%
Server p50 latency:      55.2 ms
Failures:                1
Report:                  results/report.md
```
