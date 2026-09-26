# 🏠 Home Router Bench

> I was reading about Jev — a typed-decision model from [TypeSafe](https://typesafe.ai) — and got pretty excited about the concept: one forward pass, multiple structured fields out the other side. But Jev isn't open-source. I poked around, found [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations, which implements the same "answer multiple questions in a single pass" architecture, and realised I could wrap it in a little test rig for a problem domain I actually use every day: home automation.

> So here it is. A FastAPI service that turns "dim the bedroom lights" into `{intent: "light_dim", room: "bedroom", direction: "down"}` in one shot. A dark-themed floor-plan UI that animates lights, music, and climate controls as you type. And a 200-case benchmark harness so you can see how the model actually performs — confusion matrix, per-intent accuracy, latency distribution, the works.

> It's not production. It's not polished. It's just a fun way to exercise a cool little model and see what it can do.

---

## How it works

The router classifies natural-language home commands into **four fields in a single model pass**:

| Field | What it captures | Example values |
|---|---|---|
| **Intent** | What the user wants to do | `light_on`, `light_off`, `light_dim`, `music_play`, `music_pause`, `music_vol`, `climate_set`, `scene`, `unknown` |
| **Room** | Where | `living_room`, `kitchen`, `bedroom`, `office`, `outside`, `whole_home` |
| **Direction** | Up or down (for dimming, volume, temperature) | `up`, `down`, `none` |
| **Confirm** | Whether the command is ambiguous | boolean |

The UI has a floor plan grid — type "play some music in the living room" and the living room tile lights up with a pulsing music icon, the glow shifts to purple, and the sidebar logs it. It makes the model's behaviour instantly visible.

---

## Quick start

### 1. Start the router

```bash
docker compose up -d --build router
```

First run downloads the ~800 MB Laya checkpoint into a persistent volume. Wait for the health check:

```bash
docker compose ps          # STATUS should show "healthy"
curl -s localhost:8010/health | jq
```

<details>
<summary>GPU mode (optional)</summary>

```bash
docker compose --profile gpu up -d --build router-gpu
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
</details>

### 2. Open the UI

Navigate to [http://localhost:8010](http://localhost:8010). Type a command or click a suggestion chip:

> `turn on the kitchen lights` · `dim the bedroom lights` · `make it warmer` · `play some music` · `lights off`

### 3. Call the API

```bash
curl -s http://localhost:8010/classify \
  -H 'Content-Type: application/json' \
  -d '{"utterance":"turn on the kitchen lights"}' | jq
```

Returns:

```json
{
  "utterance": "turn on the kitchen lights",
  "intent": "light_on",
  "room": "kitchen",
  "direction": "none",
  "confirm": false,
  "confidence": {
    "intent": 0.998,
    "room": 0.995,
    "direction": 0.980,
    "confirm": 0.965
  },
  "latency_ms": 234.56
}
```

---

## Benchmark results

I ran the full 200-case benchmark on the hardware I had lying around — here's what it looks like.

### Hardware

| Component | Spec |
|---|---|
| **CPU** | Intel Xeon E5-2698 v3 (8-core / 16-thread @ 2.30 GHz) |
| **GPU** | NVIDIA GeForce RTX 2060 (not utilised — this was a CPU-only run) |
| **Model** | `convaiinnovations/laya` |
| **Device** | `cpu` |
| **Container** | Docker, CPU profile |

### Headline accuracy

| Field | Accuracy |
|---|---:|
| **Intent** | **88.5%** |
| Room (excluding unknowns) | 99.0% |
| Direction (excluding unknowns) | 88.5% |
| All three fields correct | **77.5%** |

Room accuracy is excellent — the model nearly always picks the right room or correctly falls back to `whole_home`. Direction gets confused when a command uses directional language that isn't really directional ("start up the kitchen lights" — "up" leaks through). Intent is the main lever to pull.

### Latency (CPU, no GPU offload)

| | Mean | p50 | p95 | p99 | Max |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Server (inference) | 973 ms | 974 ms | 1000 ms | 1008 ms | 1036 ms |
| Client (network + inference) | 976 ms | 977 ms | 1003 ms | 1011 ms | 1039 ms |

~1 second per classify on an older Xeon, no accelerators. On the CPU profile the model fits in ~6 GB of RAM. With GPU offload (the `gpu` compose profile) you'd expect p50 latency around 50–150 ms on this RTX 2060 — I haven't run that benchmark yet, but the improvement would be dramatic.

### Per-intent breakdown

| Intent | N | Accuracy | Mean confidence (correct) | Mean confidence (wrong) |
|:---:|:---:|:---:|:---:|:---:|
| `light_on` | 25 | **100%** | 0.97 | — |
| `light_off` | 25 | 96% | 0.99 | 1.0 |
| `light_dim` | 25 | 76% | 0.80 | 0.77 |
| `music_play` | 20 | 95% | 0.91 | 1.0 |
| `music_pause` | 20 | **100%** | 0.99 | — |
| `music_vol` | 25 | 72% | 0.94 | 0.75 |
| `climate_set` | 25 | 96% | 0.89 | 0.44 |
| `scene` | 15 | 93% | 0.86 | 0.36 |
| `unknown` | 20 | **70%** | 0.63 | 0.62 |

The weak spots are predictable:

- **`music_vol` (72%)** — "quieter music" gets treated as `music_pause`, "crank the music" gets treated as `music_pause`, "bring the music up" gets treated as `music_play`. The boundary between volume, pause, and play is genuinely fuzzy in natural language.
- **`light_dim` (76%)** — "dim the bedroom lights" → `light_off`, "lower the living room lights" → `light_off`. The model interprets "lower" as off rather than dim.
- **`unknown` (70%)** — "unlock the front door" → `light_off`, "arm the alarm" → `light_on`, "cancel" → `music_pause`. These are security/lock commands that the model tries to map onto the closest home-automation intent.

### Confusion matrix

| expected \ predicted | `climate` | `dim` | `off` | `on` | `pause` | `play` | `vol` | `scene` | `unknown` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `climate_set` | **24** | 1 | — | — | — | — | — | — | — |
| `light_dim` | — | **19** | 4 | 2 | — | — | — | — | — |
| `light_off` | — | — | **24** | 1 | — | — | — | — | — |
| `light_on` | — | — | — | **25** | — | — | — | — | — |
| `music_pause` | — | — | — | — | **20** | — | — | — | — |
| `music_play` | — | — | — | — | 1 | **19** | — | — | — |
| `music_vol` | — | — | — | — | 5 | 2 | **18** | — | — |
| `scene` | — | — | — | — | — | — | — | **14** | 1 |
| `unknown` | — | — | 3 | 1 | 1 | — | — | 1 | **14** |

### What this means for fine-tuning

The 45 failure cases are an excellent fine-tuning starter set. The base Laya checkpoint reaches strong accuracy after a single fine-tune pass on ~30k typed-decision examples (per the Convai Innovations fine-tuning notebook). If I were taking this further, I'd:

1. Add 5–10 paraphrases of each edge case from the failures list
2. Toss in 20–30 adversarial non-home commands that should be `unknown`
3. Fine-tune with the Laya recipe

But for a weekend project that's just exercising a model I thought was neat? The base checkpoint does fine. It lights up the right room on the floor plan and it gets the intent right most of the time.

---

## Project layout

```
home-router-bench/
├── docker-compose.yml          # CPU (default) + GPU profile
├── router/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                  # FastAPI: /classify, /dispatch, /health
│   └── index.html              # Demo UI with animated floor plan
└── bench/
    ├── requirements.txt
    ├── cases.jsonl             # 200 labelled utterances
    └── run_bench.py            # Benchmark runner + report generator
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Model status, device, ready flag |
| `POST` | `/classify` | Classify an utterance → structured fields + confidence |
| `POST` | `/dispatch` | Classify + call a configurable home-automation backend |

### Configuration

| Env var | Default | Description |
|---|---|---|
| `LAYA_MODEL` | `convaiinnovations/laya` | HuggingFace model |
| `LAYA_DEVICE` | `cpu` | `cpu` or `cuda` |
| `LAYA_PRELOAD` | `1` | Warm the model on startup |
| `CONFIDENCE_THRESHOLD` | `0.75` | Min confidence for auto-dispatch |
| `HOME_API_BASE` | `http://home.local/api` | Backend for `/dispatch` |

---

## Running the benchmark yourself

```bash
cd bench
pip install -r requirements.txt
python run_bench.py
```

Options:

```bash
python run_bench.py --router http://other-host:8010   # remote router
python run_bench.py --limit 20                         # smoke test
python run_bench.py --out ./runs/my-run                # custom output dir
```

Output lands in `bench/results/`:
- **`raw.jsonl`** — every case + prediction
- **`summary.json`** — aggregate metrics + confusion matrix
- **`report.md`** — human-readable report

---

## License

MIT
