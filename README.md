# 🏠 Home Router Bench

> I was reading about Jev — a typed-decision model from TypeSafe — and got excited about the concept: one forward pass, multiple structured fields out the other side. But Jev isn't open-source. I found [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations, which implements the same "answer multiple questions in a single pass" architecture under an Apache-2.0 license, and decided to wrap it in a little test rig for a problem domain I use every day: home automation.
>
> This repo is **not** a custom inference server. Laya ships its own production-grade HTTP server (`laya-serve`) that exposes the same Jev-compatible `POST /v1/systemone` wire protocol. We just package it in Docker and build example clients on top — a floor-plan UI, a benchmark harness, a triage demo, and a direct in-process Python example.

---

## What's in here

```
home-router-bench/
├── docker-compose.yml              # runs the official laya-serve (CPU + GPU profiles)
├── server/
│   └── Dockerfile                  # 5 lines: pip install laya[serve]; CMD laya-serve
└── examples/
    ├── home-automation/            # the floor-plan UI + 200-case benchmark
    │   ├── index.html              # dark-themed animated floor plan
    │   ├── home_questions.py       # QUESTIONS schema (shared)
    │   ├── cases.jsonl             # 200 labelled utterances
    │   ├── run_bench.py            # benchmark runner
    │   └── Caddyfile               # optional: serves UI + proxies to laya-serve
    ├── triage/                     # support-ticket triage (your triage.ts use case)
    │   ├── triage.py
    │   └── tickets.jsonl
    └── python-direct/              # in-process Laya (no HTTP)
        └── direct.py
```

**No `app.py`. No custom server.** The inference endpoint is the official `laya-serve` — we add nothing to it. Every example is a pure HTTP (or in-process) client.

---

## Quick start

### 1. Deploy the inference server

```bash
docker compose up -d --build laya-serve           # CPU (default)
docker compose --profile gpu up -d --build laya-serve-gpu   # GPU (NVIDIA toolkit required)
```

First run downloads the ~800 MB Laya checkpoint into a persistent volume. Wait for health:

```bash
curl -s localhost:8000/health | jq
# → {"status": "ok", "loaded": ["english"], "device": "cuda", ...}
```

### 2. Make a decision

```bash
curl -s http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"utterance": "turn on the kitchen lights"},
    "questions": {
      "intent":   {"type": "choice", "instructions": "What does the user want to do?",
                   "criteria": {"light_on": "turn a light on", "unknown": "not a command"}},
      "room":     {"type": "choice", "instructions": "Which room?",
                   "criteria": {"kitchen": "kitchen", "whole_home": "anywhere"}}
    }
  }' | jq
```

Returns Laya's raw answer — `choice`, `probabilities`, `answer_confidence`, and a `routing` block recording which checkpoint answered. Same shape for any question schema.

### 3. Run the examples

```bash
# Floor-plan UI (with Caddy proxy to a remote laya-serve)
cd examples/home-automation
LAYA_UPSTREAM=epyc02:8000 caddy run --config Caddyfile
# open http://localhost:8080

# Benchmark
python run_bench.py --router http://localhost:8000

# Triage
cd ../triage
python triage.py --server http://localhost:8000

# Direct in-process (no HTTP)
cd ../python-direct
python direct.py --device cuda
```

---

## API — the official `laya-serve`

We don't implement this; we just run it. Docs: [nandhakishorm.github.io/laya/http-api](https://nandhakishorm.github.io/laya/http-api/)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness (open) + checkpoint/device state (with auth) |
| `POST` | `/v1/systemone` | One decision: `state` + `questions` → typed `answers` |
| `POST` | `/v1/systemone/batch` | Same questions over an array of states |

Configuration is all env vars — `LAYA_DEVICE`, `LAYA_PRELOAD`, `LAYA_API_KEY`, `LAYA_MAX_CONCURRENT`, `LAYA_IDLE_UNLOAD_SECONDS`, etc. See the [official docs](https://nandhakishorm.github.io/laya/docker/).

---

## Benchmark results

Run on the hardware I had lying around, with **Laya 0.3.26**.

### Hardware

| Machine | CPU | GPU | Device |
|---|---|---|---|
| Local dev box | Intel Xeon E5-2698 v3 (8c/16t @ 2.30GHz) | NVIDIA RTX 2060 | `cuda` |
| epyc02 | AMD EPYC 7402P (24c/48t @ 2.80GHz) | 2× NVIDIA RTX PRO 6000 Blackwell | `cuda` |
| epyc02 (CPU) | AMD EPYC 7402P | — | `cpu` |

### Latency (200-case corpus)

| Configuration | Mean | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| **RTX 2060 (GPU)** | 59 ms | **55 ms** | 69 ms | 137 ms |
| **EPYC 7402P (CPU)** | 657 ms | **620 ms** | 836 ms | 1153 ms |
| **Xeon E5-2698 v3 (CPU)** | 973 ms | **974 ms** | 1000 ms | 1008 ms |

GPU offload gives ~11× speedup. The model fits in ~1.6 GB VRAM.

### Accuracy (200-case corpus, identical across devices)

| Field | Accuracy |
|---|---:|
| Intent | **88.5%** |
| Room (excluding unknowns) | 99.0% |
| Direction (excluding unknowns) | 88.5% |
| All three fields correct | **77.5%** |

Per-intent breakdown, confusion matrix, and the 45 failure cases are
generated fresh by `run_bench.py` → `results/report.md`.

### Weak spots on the base checkpoint

- **`music_vol` (72%)** — "quieter music" → `music_pause`, "crank the music" → `music_pause`
- **`light_dim` (76%)** — "dim the bedroom lights" → `light_off`
- **`unknown` (70%)** — "unlock the front door" → `light_off`

The failure list is a fine-tuning starter set. The base Laya checkpoint
reaches strong accuracy after a single fine-tune pass on ~30k typed-decision
examples.

---

## Configuration

The `docker-compose.yml` is thin packaging around the official image. Useful
env vars (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `LAYA_DEVICE` | `cpu` | `cpu` or `cuda` |
| `LAYA_PRELOAD` | `1` | Warm checkpoints at startup |
| `LAYA_API_KEY` | (none) | Require `Authorization: Bearer <key>` |
| `LAYA_MAX_CONCURRENT` | `16` | In-flight request cap (excess → 503) |
| `LAYA_IDLE_UNLOAD_SECONDS` | `0` | Free VRAM after N idle seconds |
| `LAYA_LOG_LEVEL` | `info` | uvicorn log level |

Full list in the [official docs](https://nandhakishorm.github.io/laya/docker/#server-configuration).

---

## License

MIT for this repo. Laya itself is Apache-2.0 (© Convai Innovations).
