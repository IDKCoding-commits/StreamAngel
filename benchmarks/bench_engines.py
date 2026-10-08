#!/usr/bin/env python3
"""Head-to-head ROCm benchmark: parakeet.cpp (HIP, Q8_0) vs original Parakeet
(Transformers/PyTorch on ROCm).

Both engines consume the same GGUF/checkpoint and emit the same transcript
contract, so this measures what actually differs: wall-clock, throughput, memory
and transcript agreement.

    python3 benchmarks/bench_engines.py audio.wav --engines parakeet-cpp transformers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nemo_speech_rocm as ns  # noqa: E402


def _norm(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def _wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, 1):
            cost = 0 if rw == hw else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(ref)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", default=str(ns.DEFAULT_MODEL))
    ap.add_argument("--engines", nargs="+", default=["parakeet-cpp", "transformers"])
    ap.add_argument("--torch-python", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    report: dict = {"model": args.model, "runs": [], "transcripts": {}}
    for engine in args.engines:
        for audio in args.audio:
            try:
                t0 = time.time()
                hyps = ns.transcribe([audio], model=args.model, engine=engine,
                                     torch_python=args.torch_python)
                wall = time.time() - t0
            except Exception as exc:  # noqa: BLE001
                report["runs"].append({"engine": engine, "audio": str(audio), "error": str(exc)})
                continue
            h = hyps[0]
            dur = h.duration or 0.0
            report["transcripts"].setdefault(str(audio), {})[engine] = h.text
            report["runs"].append({
                "engine": engine,
                "audio": str(audio),
                "wall_s": round(wall, 2),
                "audio_s": round(dur, 2),
                "rtfx": round(dur / wall, 2) if wall else None,
                "words": len(h.words),
                "chars": len(h.text),
                "text": h.text,
            })

    # agreement between engines, per audio
    agreement = {}
    for audio, by_engine in report["transcripts"].items():
        if len(by_engine) >= 2:
            names = list(by_engine)
            base = _norm(by_engine[names[0]])
            for other in names[1:]:
                agreement[audio] = {
                    "between": [names[0], other],
                    "wer": round(_wer(base, _norm(by_engine[other])), 4),
                }
    report["agreement"] = agreement

    print(f"{'engine':<14}{'audio':<28}{'wall_s':>8}{'audio_s':>9}{'RTF':>7}{'words':>7}")
    for r in report["runs"]:
        if "error" in r:
            print(f"{r['engine']:<14}{Path(r['audio']).name:<28}  ERROR: {r['error'][:60]}")
            continue
        print(f"{r['engine']:<14}{Path(r['audio']).name:<28}{r['wall_s']:>8}{r['audio_s']:>9}"
              f"{r['rtfx']:>7}{r['words']:>7}")
    for audio, agr in agreement.items():
        print(f"agreement {Path(audio).name}: WER({agr['between'][0]} vs {agr['between'][1]}) "
              f"= {agr['wer']:.4f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
