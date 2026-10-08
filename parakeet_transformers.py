#!/usr/bin/env python3
"""Original Parakeet TDT 0.6B v3 on ROCm (HuggingFace Transformers / PyTorch).

Engine behind ``nemo_speech_rocm.py --engine transformers``: it runs the *original*
``nvidia/parakeet-tdt-0.6b-v3`` checkpoint through Transformers on the AMD GPU and
emits the same JSON contract as the parakeet.cpp (ROCm) engine, so the two are
drop-in interchangeable.

Usage:
    parakeet_transformers.py --model nvidia/parakeet-tdt-0.6b-v3 \
        --input audio16k.wav --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np


def _words_from_timestamps(ts) -> list[dict]:
    """Merge BPE sub-word pieces into word-level timestamps.

    Transformers' ``decode(..., durations=...)`` returns one entry per token
    (e.g. ``"O"``, ``"ne"``, ``" by"``); NeMo and nemo-speech report whole words.
    A piece beginning with a space/``\u2581`` starts a new word.
    """
    pieces: list[dict] = []
    for item in ts or []:
        if not isinstance(item, dict):
            continue
        token = item.get("word", item.get("token", item.get("text", "")))
        if token in (None, ""):
            continue
        pieces.append(
            {
                "word": str(token),
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
                "confidence": float(item.get("confidence", item.get("conf", 1.0))),
            }
        )

    words: list[dict] = []
    cur: dict | None = None
    for p in pieces:
        starts_new = p["word"].startswith((" ", "\u2581")) or cur is None
        if starts_new:
            if cur:
                words.append(cur)
            cur = {
                "word": p["word"].lstrip(" \u2581"),
                "start": p["start"],
                "end": p["end"],
                "confidence": p["confidence"],
            }
        else:
            cur["word"] += p["word"]
            cur["end"] = p["end"]
            cur["confidence"] = min(cur["confidence"], p["confidence"])
    if cur:
        words.append(cur)
    return [w for w in words if w["word"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--input", required=True)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtypes[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with wave.open(args.input, "rb") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / sr

    processor = AutoProcessor.from_pretrained(args.model)

    def build(dtype_name):
        d = dtypes[dtype_name]
        m = AutoModelForTDT.from_pretrained(args.model, dtype=d, device_map=device)
        m.eval()
        return m, d

    model, dtype = build(args.dtype)

    def decode():
        inputs = processor([audio], sampling_rate=sr, return_tensors="pt").to(device)
        if "input_features" in inputs and torch.is_floating_point(inputs["input_features"]):
            inputs["input_features"] = inputs["input_features"].to(dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, return_dict_in_generate=True)
        durations = getattr(out, "durations", None)
        if durations is not None:
            text, ts = processor.decode(out.sequences, durations=durations, skip_special_tokens=True)
            text = text[0] if isinstance(text, (list, tuple)) else text
            ts = ts[0] if isinstance(ts, (list, tuple)) and ts and isinstance(ts[0], (list, tuple)) else ts
            return str(text).strip(), _words_from_timestamps(ts)
        text = processor.decode(out.sequences, skip_special_tokens=True)
        text = text[0] if isinstance(text, (list, tuple)) else text
        return str(text).strip(), []

    try:
        text, words = decode()
    except Exception as exc:  # noqa: BLE001
        print(f"{args.dtype} failed on ROCm -> float32: {exc!r}", file=sys.stderr)
        del model
        torch.cuda.empty_cache()
        model, dtype = build("float32")
        text, words = decode()

    payload = {
        "file": str(Path(args.input)),
        "text": text,
        "confidence": 1,
        "duration": round(duration, 3),
        "languages": [args.lang] if args.lang else [],
        "words": words,
        "engine": f"transformers/{args.dtype}/{'cuda(ROCm)' if device == 'cuda' else 'cpu'}",
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
