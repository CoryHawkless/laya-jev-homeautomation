#!/usr/bin/env python3
"""
Run Laya directly in Python — no HTTP server needed.

This is the in-process API: load the model, call predict(), get answers.
Use this when you don't want a separate inference server.

Requires: pip install "laya>=0.3.26"

Usage:
    python direct.py                  # CPU (default)
    python direct.py --device cuda    # GPU (if available)
"""
from __future__ import annotations

import argparse
import time

import laya

HOME_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the user want to do?",
        "criteria": {
            "light_on":    "turn a light on",
            "light_off":   "turn a light off",
            "light_dim":   "change light brightness up or down",
            "music_play":  "start or resume music playback",
            "music_pause": "stop or pause music",
            "music_vol":   "change music volume up or down",
            "climate_set": "change temperature or thermostat",
            "scene":       "activate a preset scene like movie mode or goodnight",
            "unknown":     "the request is not a home-automation command or is unclear",
        },
    },
    "room": {
        "type": "choice",
        "instructions": "Which area of the home does the command target?",
        "criteria": {
            "living_room": "living room, lounge, den, main area",
            "kitchen":     "kitchen",
            "bedroom":     "master bedroom",
            "office":      "home office or study",
            "outside":     "outdoor, garden, patio, yard",
            "whole_home":  "the whole home, everywhere, or no specific room is mentioned",
        },
    },
    "direction": {
        "type": "choice",
        "instructions": "If the command implies a directional change, which way?",
        "criteria": {
            "up":   "increase, louder, brighter, warmer, hotter",
            "down": "decrease, quieter, dimmer, cooler, colder",
            "none": "no directional change requested",
        },
    },
    "confirm": {
        "type": "choice",
        "instructions": "Is the command ambiguous enough that we should confirm before acting?",
        "criteria": {
            "A": "yes, ask the user to confirm before acting",
            "B": "no, the command is clear enough to act on",
        },
    },
}

UTTERANCES = [
    "turn on the kitchen lights",
    "dim the bedroom lights",
    "play some music in the living room",
    "make it warmer",
    "goodnight",
    "what's the weather like",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya",
                    help="HuggingFace model id")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="torch device")
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device} ...")
    t0 = time.perf_counter()
    agent = laya.Agent(args.model, device=args.device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    for utterance in UTTERANCES:
        t0 = time.perf_counter()
        result = agent.predict({"utterance": utterance}, HOME_QUESTIONS)
        ms = (time.perf_counter() - t0) * 1000.0
        ans = result["answers"]
        print(f'  "{utterance}"')
        print(f"    intent:    {ans['intent']['choice']:12s}  (conf: {ans['intent']['answer_confidence']:.3f})")
        print(f"    room:      {ans['room']['choice']:12s}  (conf: {ans['room']['answer_confidence']:.3f})")
        print(f"    direction: {ans['direction']['choice']:12s}  (conf: {ans['direction']['answer_confidence']:.3f})")
        print(f"    confirm:   {'yes' if ans['confirm']['choice'] == 'A' else 'no':12s}  (conf: {ans['confirm']['answer_confidence']:.3f})")
        print(f"    latency:   {ms:.1f} ms")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
