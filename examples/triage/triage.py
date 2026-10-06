#!/usr/bin/env python3
"""
Support-ticket triage using the /v1/systemone endpoint.

Port of the Jev/OpenRouter triage.ts pattern — call a Laya inference server
with choice + noul questions, get structured answers back.

Usage:
    python triage.py                                     # local laya-serve
    python triage.py --server http://epyc02:8000         # remote
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

INTENTS: dict[str, str] = {
    "order_status": "The customer asks where an order is, when it will ship or arrive, wants to change or cancel an order before delivery, or reports a package missing or partially delivered.",
    "return_refund": "The customer wants to return, exchange, or replace an item they received, or asks about the status or rules of a return they already started.",
    "billing_dispute": "The customer says a charge, invoice, tax, discount, or refund amount is wrong, duplicated, unexpected, or unauthorized.",
    "product_question": "The customer asks about a product's features, compatibility, sizing, materials, stock, warranty, or safety before or after buying, without asking to return it.",
    "account_access": "The customer cannot log in, needs to change login or account details, or asks to merge, delete, secure, or share an account.",
}

ESCALATE = "Does the ticket describe any of the following: a threat of legal action, a regulator complaint, or a chargeback; suspected fraud or an account takeover; a safety hazard such as fire, smoke, or injury; or a customer who says this is a repeated failure and threatens to publicize it?"

QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What is the primary intent of the ticket?", "criteria": INTENTS},
    "escalate": {"type": "noul", "instructions": ESCALATE},
}


def load_tickets(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def print_result(ticket: str, ans: dict, latency_ms: float) -> None:
    intent = ans["intent"]
    esc = ans["escalate"]
    print(f"  Ticket: {ticket[:60]}...")
    print(f"    Intent:      {intent['choice']:20s}  (conf: {intent.get('answer_confidence', 0):.3f})")
    probs = intent.get("probabilities", {})
    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    print(f"    Top 3:       {', '.join(f'{k}={v:.2f}' for k, v in top3)}")
    print(f"    Escalate:    {'YES' if esc.get('noul', 0) > 0.5 else 'no':>3s}  (prob: {esc.get('noul', 0):.3f})")
    print(f"    Latency:     {latency_ms:.1f} ms")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000", help="laya-serve base URL")
    ap.add_argument("--tickets", type=Path, default=Path(__file__).with_name("tickets.jsonl"), help="path to tickets JSONL")
    args = ap.parse_args()

    tickets = load_tickets(args.tickets)
    print(f"Loaded {len(tickets)} tickets, server: {args.server}")
    print()

    with httpx.Client() as client:
        for ticket in tickets:
            t0 = time.perf_counter()
            r = client.post(f"{args.server}/v1/systemone",
                json={"state": {"ticket": ticket["ticket"]}, "questions": QUESTIONS}, timeout=60)
            r.raise_for_status()
            result = r.json()
            latency = (time.perf_counter() - t0) * 1000.0
            print_result(ticket["ticket"], result["answers"], latency)

    return 0


if __name__ == "__main__":
    exit(main())
