"""
Home-automation intent router.

Wraps Laya (an open decision model) behind a small HTTP API:

    POST /classify             -> typed decision only (used by the benchmarker)
    POST /dispatch             -> classify + call the home-automation backend
    GET  /health               -> readiness probe

OpenAI-compatible endpoints (for Trifrost / any OpenAI proxy):

    GET  /v1/models            -> list available model IDs
    POST /v1/chat/completions   -> OpenAI-chat-format classify

The OpenAI wrapper extracts the last user message as the utterance, runs
classification, and returns the result as JSON content in the response.

Model, device and confidence threshold are driven by env vars:
    LAYA_MODEL              (default: convaiinnovations/laya)
    LAYA_DEVICE             (cpu | cuda; default cpu)
    LAYA_PRELOAD            (1 to warm at startup; default 1)
    CONFIDENCE_THRESHOLD    (float, default 0.75)
    HOME_API_BASE           (base URL for /dispatch; default http://home.local/api)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import laya
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


MODEL_REPO = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
DEVICE = os.environ.get("LAYA_DEVICE", "cpu")
PRELOAD = os.environ.get("LAYA_PRELOAD", "1") == "1"
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
HOME_API_BASE = os.environ.get("HOME_API_BASE", "http://home.local/api")

# OpenAI model ID exposed on /v1/models and accepted in chat completions
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "laya-home-router")


# ---- Decision schema ------------------------------------------------------
#
# One classify call scores all four fields in a single forward pass.
# Note the "confirm" field uses a two-option choice with neutral A/B labels
# rather than a noul — Laya's noul primitive can latch onto the literal
# true/false label text on the English checkpoint (upstream issue #156).

QUESTIONS: dict[str, dict[str, Any]] = {
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


# ---- OpenAI request/response schemas --------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=OPENAI_MODEL_ID)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool | None = False
    temperature: float | None = None
    max_tokens: int | None = None


# ---- Model lifecycle ------------------------------------------------------

_state: dict[str, Any] = {"agent": None, "ready": False, "loaded_at": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(f"[startup] loading {MODEL_REPO} on {DEVICE} (preload={PRELOAD})")
    t0 = time.perf_counter()
    agent = laya.load(MODEL_REPO, device=DEVICE)
    if PRELOAD:
        # Warm the forward pass so the first real request isn't a cold cache.
        agent.predict({"utterance": "turn on the lights"}, QUESTIONS)
    _state["agent"] = agent
    _state["loaded_at"] = time.time()
    _state["ready"] = True
    print(f"[startup] ready in {(time.perf_counter() - t0):.1f}s")
    yield


app = FastAPI(title="home-router", lifespan=lifespan)

# ---- Web UI ----------------------------------------------------------------

_HERE = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
def web_ui():
    return (_HERE / "index.html").read_text()


# ---- Native API -----------------------------------------------------------

class ClassifyRequest(BaseModel):
    utterance: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {
        "ready": _state["ready"],
        "model": MODEL_REPO,
        "device": DEVICE,
        "loaded_at": _state["loaded_at"],
    }


@app.post("/classify")
def classify(req: ClassifyRequest):
    t0 = time.perf_counter()
    result = _state["agent"].predict({"utterance": req.utterance}, QUESTIONS)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ans = result["answers"]
    return {
        "utterance": req.utterance,
        "intent":    ans["intent"]["choice"],
        "room":      ans["room"]["choice"],
        "direction": ans["direction"]["choice"],
        "confirm":   ans["confirm"]["choice"] == "A",
        "confidence": {
            "intent":    ans["intent"]["confidence"],
            "room":      ans["room"]["confidence"],
            "direction": ans["direction"]["confidence"],
            "confirm":   ans["confirm"]["confidence"],
        },
        "latency_ms": round(latency_ms, 2),
    }


# Map (intent, direction) -> (HTTP method, path template, body)
_ROUTES = {
    "light_on":    ("POST",  "/rooms/{room}/lights",  lambda d: {"state": "on"}),
    "light_off":   ("POST",  "/rooms/{room}/lights",  lambda d: {"state": "off"}),
    "light_dim":   ("PATCH", "/rooms/{room}/lights",  lambda d: {"brightness_delta": 20 if d == "up" else -20}),
    "music_play":  ("POST",  "/rooms/{room}/media",   lambda d: {"action": "play"}),
    "music_pause": ("POST",  "/rooms/{room}/media",   lambda d: {"action": "pause"}),
    "music_vol":   ("PATCH", "/rooms/{room}/media",   lambda d: {"volume_delta": 10 if d == "up" else -10}),
    "climate_set": ("PATCH", "/rooms/{room}/climate", lambda d: {"setpoint_delta": 1 if d == "up" else -1}),
    "scene":       ("POST",  "/rooms/{room}/scene",   lambda d: {"activate": True}),
}


@app.post("/dispatch")
def dispatch(req: ClassifyRequest):
    c = classify(req)
    intent = c["intent"]
    if intent == "unknown" or c["confidence"]["intent"] < CONFIDENCE_THRESHOLD or c["confirm"]:
        return {"status": "needs_clarification", **c}

    method, path_tpl, body_fn = _ROUTES[intent]
    path = path_tpl.format(room=c["room"])
    body = body_fn(c["direction"])
    try:
        r = httpx.request(method, HOME_API_BASE + path, json=body, timeout=5.0)
        return {"status": "ok", "http": r.status_code, "call": {"method": method, "path": path, "body": body}, **c}
    except httpx.HTTPError as e:
        return {"status": "dispatch_error", "error": str(e), "call": {"method": method, "path": path, "body": body}, **c}


# ---- OpenAI-compatible endpoints (for Trifrost integration) ---------------


@app.get("/v1/models")
def openai_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_MODEL_ID,
                "object": "model",
                "created": int(_state.get("loaded_at", 0)),
                "owned_by": "home-router",
            }
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions.

    Extracts the last user message, runs Laya classification, and returns
    the structured result as the response content.
    """
    # Extract utterance from the last user message
    utterance: str | None = None
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content.strip():
            utterance = msg.content.strip()
            break

    if not utterance:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "No user message found with non-empty content",
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
        )

    # Run classification
    t0 = time.perf_counter()
    result = _state["agent"].predict({"utterance": utterance}, QUESTIONS)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ans = result["answers"]
    classification = {
        "intent":    ans["intent"]["choice"],
        "room":      ans["room"]["choice"],
        "direction": ans["direction"]["choice"],
        "confirm":   ans["confirm"]["choice"] == "A",
        "confidence": {
            "intent":    ans["intent"]["confidence"],
            "room":      ans["room"]["confidence"],
            "direction": ans["direction"]["confidence"],
            "confirm":   ans["confirm"]["confidence"],
        },
        "latency_ms": round(latency_ms, 2),
    }

    content_json = json.dumps(classification, indent=2)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or OPENAI_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_json,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(utterance.split()),
            "completion_tokens": len(content_json.split()),
            "total_tokens": len(utterance.split()) + len(content_json.split()),
        },
    }
