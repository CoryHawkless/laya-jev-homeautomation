# Support-ticket triage example

A port of the classic Jev/OpenRouter `triage.ts` pattern, calling a Laya
`/v1/systemone` endpoint instead. Same question schema, same answer shape —
only the base URL changes.

Demonstrates Laya's `choice` + `noul` question types working together: one
forward pass returns the primary intent **and** the escalation probability.

## Run

```bash
pip install -r requirements.txt
python triage.py                                     # against local laya-serve
python triage.py --server http://epyc02:8000         # against a remote instance
```

## Question schema

```python
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the primary intent of the ticket?",
        "criteria": INTENTS,   # 5 labels: order_status, return_refund, billing_dispute, product_question, account_access
    },
    "escalate": {
        "type": "noul",
        "instructions": "Does the ticket describe a threat of legal action, fraud, safety hazard, ...?",
    },
}
```

## Jev → Laya notes

| Field | Jev | Laya |
|---|---|---|
| `intent.choice` | ✅ | ✅ identical |
| `intent.confidence` | ✅ | ✅ identical |
| `intent.probabilities` | ✅ | ✅ identical |
| `escalate.noul` | ✅ | ✅ the "yes" probability |
| `escalate.action` / `answer_confidence` | — | extra fields (harmless) |
| `usage.cost` | ✅ USD cost | ❌ not present (self-hosted) |

Your `triage.ts` works against `laya-serve` with a one-line `baseUrl` change.
