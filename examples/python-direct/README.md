# Python-direct example

Run Laya **in-process** — no HTTP server, no Docker, no API. Just load the model and call `predict()`.

Use this when:
- You want the lowest possible latency (no network hop)
- You don't need the Jev-compatible `/v1/systemone` endpoint
- You're embedding Laya into another Python application

## Run

```bash
pip install "laya>=0.3.26"
python direct.py                  # CPU
python direct.py --device cuda    # GPU
```

## What it does

Loads `convaiinnovations/laya`, runs six sample home-automation utterances through `agent.predict(state, questions)`, and prints the typed answers with per-field confidence and latency.

```python
import laya
agent = laya.Agent("convaiinnovations/laya", device="cuda")
result = agent.predict({"utterance": "turn on the kitchen lights"}, HOME_QUESTIONS)
print(result["answers"]["intent"]["choice"])  # → "light_on"
```

The `questions` dict is the same schema you'd send to `/v1/systemone` — the in-process API and the HTTP API share the same question/answer contract.
