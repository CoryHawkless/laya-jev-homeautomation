"""
Benchmark the home-automation router.

Runs every case in cases.jsonl against the router's /classify endpoint,
records raw predictions and latencies, then produces:

    results/raw.jsonl     one line per case, prediction + timing
    results/summary.json  aggregate metrics
    results/report.md     Markdown report you can paste into a doc

Usage:
    python run_bench.py                                         # defaults below
    python run_bench.py --router http://epyc02:8000 --out out/
    python run_bench.py --limit 20                              # smoke-test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from home_questions import HOME_QUESTIONS


# ---- IO -------------------------------------------------------------------

def load_cases(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def wait_ready(client: httpx.Client, url: str, timeout_s: float = 300.0) -> dict:
    """Poll /health until status=ok or timeout. First run downloads weights."""
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = client.get(f"{url}/health", timeout=10)
            r.raise_for_status()
            h = r.json()
            if h.get("status") == "ok":
                return h
        except Exception as e:  # noqa: BLE001 -- best-effort retry
            last_err = e
        time.sleep(2.0)
    raise RuntimeError(
        f"laya-serve at {url} not ready within {timeout_s:.0f}s (last err: {last_err})"
    )


def run_one(client: httpx.Client, url: str, utterance: str) -> dict:
    t0 = time.perf_counter()
    r = client.post(f"{url}/v1/systemone", json={"state": {"utterance": utterance}, "questions": HOME_QUESTIONS}, timeout=60)
    r.raise_for_status()
    result = r.json()
    result["latency_ms"] = float(r.headers.get("X-Inference-Time-Ms", 0))
    result["client_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return result


# ---- Evaluation -----------------------------------------------------------

def _percentile(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    k = max(0, min(len(sorted_xs) - 1, int(round(p * (len(sorted_xs) - 1)))))
    return sorted_xs[k]


def evaluate(cases: list[dict], results: list[dict]) -> dict:
    intent_correct = room_correct = direction_correct = all_correct = 0
    per_intent: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "correct": 0, "conf_correct": [], "conf_wrong": []}
    )
    confusion: dict[str, Counter] = defaultdict(Counter)
    latencies_server: list[float] = []
    latencies_client: list[float] = []
    failures: list[dict] = []

    for case, result in zip(cases, results):
        exp = case["expected"]
        ans = result["answers"]
        got_intent = ans["intent"]["choice"]
        got_room = ans["room"]["choice"]
        got_dir = ans["direction"]["choice"]
        conf = ans["intent"]["answer_confidence"]

        latencies_server.append(result["latency_ms"])
        latencies_client.append(result["client_latency_ms"])

        i_ok = got_intent == exp["intent"]
        # For an unknown command we only care about intent; the router won't
        # dispatch anything, so room/direction shouldn't be scored against
        # arbitrary ground truth.
        if exp["intent"] == "unknown":
            r_ok = d_ok = True
        else:
            r_ok = got_room == exp["room"]
            d_ok = got_dir == exp["direction"]

        intent_correct += int(i_ok)
        room_correct += int(r_ok)
        direction_correct += int(d_ok)
        if i_ok and r_ok and d_ok:
            all_correct += 1

        pi = per_intent[exp["intent"]]
        pi["n"] += 1
        pi["correct"] += int(i_ok)
        (pi["conf_correct"] if i_ok else pi["conf_wrong"]).append(conf)

        confusion[exp["intent"]][got_intent] += 1

        if not (i_ok and r_ok and d_ok):
            failures.append({
                "id": case["id"],
                "utterance": case["utterance"],
                "expected": exp,
                "got": {"intent": got_intent, "room": got_room, "direction": got_dir},
                "confidence": round(conf, 3),
            })

    n = len(cases)
    s_sorted = sorted(latencies_server)
    c_sorted = sorted(latencies_client)

    def _lat(sorted_xs: list[float]) -> dict[str, float]:
        return {
            "mean": round(statistics.mean(sorted_xs), 2),
            "p50":  round(_percentile(sorted_xs, 0.50), 2),
            "p95":  round(_percentile(sorted_xs, 0.95), 2),
            "p99":  round(_percentile(sorted_xs, 0.99), 2),
            "max":  round(sorted_xs[-1], 2),
        }

    return {
        "n": n,
        "intent_accuracy":     round(intent_correct / n, 4),
        "room_accuracy":       round(room_correct / n, 4),
        "direction_accuracy":  round(direction_correct / n, 4),
        "all_fields_accuracy": round(all_correct / n, 4),
        "latency_server_ms": _lat(s_sorted),
        "latency_client_ms": _lat(c_sorted),
        "per_intent": {
            k: {
                "n": v["n"],
                "accuracy": round(v["correct"] / v["n"], 4),
                "mean_conf_correct": round(statistics.mean(v["conf_correct"]), 3)
                    if v["conf_correct"] else None,
                "mean_conf_wrong": round(statistics.mean(v["conf_wrong"]), 3)
                    if v["conf_wrong"] else None,
            }
            for k, v in per_intent.items()
        },
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "failures": failures,
    }


# ---- Report ---------------------------------------------------------------

def render_markdown(summary: dict, meta: dict) -> str:
    lines: list[str] = []
    lines += [
        "# Home Automation Router — Benchmark Report",
        "",
        f"- **Timestamp:** {meta['timestamp']}",
        f"- **Router URL:** `{meta['router_url']}`",
        f"- **Model:** `{meta['model']}` on `{meta['device']}`",
        f"- **Cases:** {summary['n']}",
        "",
        "## Headline",
        "",
        "| Field | Accuracy |",
        "|---|---:|",
        f"| Intent | **{summary['intent_accuracy']:.1%}** |",
        f"| Room (excluding unknowns) | {summary['room_accuracy']:.1%} |",
        f"| Direction (excluding unknowns) | {summary['direction_accuracy']:.1%} |",
        f"| All three fields correct | {summary['all_fields_accuracy']:.1%} |",
        "",
        "## Latency",
        "",
        "| | Mean | p50 | p95 | p99 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Server-measured (`/classify` inference)", "latency_server_ms"),
                       ("Client-measured (network + inference)",   "latency_client_ms")):
        lat = summary[key]
        lines.append(
            f"| {label} | {lat['mean']} ms | {lat['p50']} ms | "
            f"{lat['p95']} ms | {lat['p99']} ms | {lat['max']} ms |"
        )
    lines += [
        "",
        "## Per-intent accuracy",
        "",
        "| Intent | N | Accuracy | Mean confidence (correct) | Mean confidence (wrong) |",
        "|---|---:|---:|---:|---:|",
    ]
    for intent, s in sorted(summary["per_intent"].items()):
        cc = s["mean_conf_correct"] if s["mean_conf_correct"] is not None else "–"
        cw = s["mean_conf_wrong"] if s["mean_conf_wrong"] is not None else "–"
        lines.append(f"| `{intent}` | {s['n']} | {s['accuracy']:.1%} | {cc} | {cw} |")

    all_intents = sorted(summary["per_intent"].keys())
    lines += [
        "",
        "## Confusion matrix (rows = expected, columns = predicted)",
        "",
        "| expected \\ predicted | " + " | ".join(f"`{i}`" for i in all_intents) + " |",
        "|---" + "|---:" * len(all_intents) + "|",
    ]
    for exp_intent in all_intents:
        row = summary["confusion"].get(exp_intent, {})
        cells = [str(row.get(got, 0)) for got in all_intents]
        lines.append(f"| `{exp_intent}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## Failures ({len(summary['failures'])})",
        "",
    ]
    if not summary["failures"]:
        lines.append("_None — every case matched expectations._")
    else:
        lines += [
            "Format: expected `intent / room / direction` → predicted `intent / room / direction`.",
            "Confidence is the model's probability for its predicted intent.",
            "",
            "| ID | Utterance | Expected | Got | Conf |",
            "|---|---|---|---|---:|",
        ]
        for f in summary["failures"]:
            e = f["expected"]
            g = f["got"]
            exp_s = f"`{e['intent']}` / `{e['room']}` / `{e['direction']}`"
            got_s = f"`{g['intent']}` / `{g['room']}` / `{g['direction']}`"
            utt = f["utterance"].replace("|", "\\|")
            lines.append(f"| {f['id']} | {utt} | {exp_s} | {got_s} | {f['confidence']} |")

    return "\n".join(lines) + "\n"


# ---- Main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="http://localhost:8000",
                    help="base URL of the laya-serve instance")
    ap.add_argument("--cases", type=Path,
                    default=Path(__file__).with_name("cases.jsonl"),
                    help="path to the JSONL case file")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("results"),
                    help="output directory for raw.jsonl, summary.json, report.md")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N cases (0 = all)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} cases from {args.cases}")

    with httpx.Client() as client:
        print(f"Waiting for {args.router}/health ...")
        health = wait_ready(client, args.router)
        print(f"  ready: device={health.get('device','?')} loaded={health.get('loaded',[])}")

        results: list[dict] = []
        t_start = time.perf_counter()
        for i, case in enumerate(cases, 1):
            results.append(run_one(client, args.router, case["utterance"]))
            if i % 25 == 0 or i == len(cases):
                elapsed = time.perf_counter() - t_start
                rate = i / elapsed if elapsed else 0
                print(f"  {i}/{len(cases)}  ({rate:.1f} req/s)")

    # Persist raw predictions alongside their ground truth
    raw_path = args.out / "raw.jsonl"
    with raw_path.open("w") as f:
        for case, result in zip(cases, results):
            f.write(json.dumps({"case": case, "result": result}) + "\n")

    summary = evaluate(cases, results)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "router_url": args.router,
        "model": health.get("model"),
        "device": health.get("device"),
    }
    report_md = render_markdown(summary, meta)
    (args.out / "report.md").write_text(report_md)

    print()
    print(f"Intent accuracy:         {summary['intent_accuracy']:.1%}")
    print(f"All fields correct:      {summary['all_fields_accuracy']:.1%}")
    print(f"Server p50 latency:      {summary['latency_server_ms']['p50']} ms")
    print(f"Failures:                {len(summary['failures'])}")
    print(f"Report:                  {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
