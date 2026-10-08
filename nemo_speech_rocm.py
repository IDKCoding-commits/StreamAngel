#!/usr/bin/env python3
"""streamangel-asr -- a drop-in for NVIDIA's documented Nemotron Speech runtime,
running on whatever acceleration the machine actually has.

The ``nvidia/parakeet-tdt-0.6b-v3`` model card documents NVIDIA's ``nemo-speech``
(NeMo-Speech.cpp) as the native C++ runtime:

    nemo-speech transcribe audio.wav --model models/parakeet-tdt-0.6b-v3.q8_0.gguf

This module provides the *same command surface* and the *same return shapes*, but
it does not care which backend produces them. It discovers whatever is installed
and uses it:

``parakeet-cpp``   mudler/parakeet.cpp -- ggml, any GPU backend (ROCm/HIP, CUDA,
                   Vulkan, Metal) or CPU. Runs the GGUF published in
                   ``mudler/parakeet-cpp-gguf``.
``nemo-speech``    NVIDIA NeMo-Speech.cpp, if installed (CPU / CUDA / Metal /
                   Vulkan; upstream ships no ROCm backend). Runs NVIDIA's own
                   GGUF from the model repo.
``transformers``   The original checkpoint via HF Transformers/PyTorch (needs a
                   venv with torch; runs on CUDA or ROCm).

Engine order is tried automatically; ``--engine`` forces one. Nothing here is
tied to a particular machine, GPU, or OS: discovery uses environment variables
plus ``PATH``, and the model is fetched on first use.

NOTE ON MODELS -- the two C++ runtimes do NOT share a GGUF. The files carry
different tensor layouts (parakeet.cpp uses ``parakeet.*`` metadata keys,
NeMo-Speech.cpp uses ``asr.*``), so each engine downloads its own conversion of
the same checkpoint. They produce the same transcript.

Interface parity (verified against ``nemo-speech`` 0.2.0 on the model card's own
sample ``2086-149220-0033.wav`` -- ``-f text``, ``-f srt`` and ``-f vtt`` are
byte-identical; JSON differs only in a few word *end* times):

  * ``transcribe INPUT [--model M] [-f text|json|srt|vtt] [--word-times] ...``
  * JSON:  {"file","text","confidence","duration","languages",
            "words":[{"word","start","end","confidence"}]}
  * SRT/VTT subtitle rendering
  * global ``--json / --quiet / --verbose / --version``
  * ``doctor``, ``model list``, ``pull``

Python parity with the model card's ``asr_model.transcribe([...])`` usage is
available through :func:`transcribe` and :class:`Hypothesis`
(``.text`` and ``.timestamp["word"|"segment"]``).

Sibling helper: ``parakeet_transformers.py`` (the transformers engine runner).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__version__ = "0.2.0"

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("STREAMANGEL_MODELS") or (PROJECT_ROOT / "models"))
TRANSFORMERS_HELPER = PROJECT_ROOT / "parakeet_transformers.py"

# --------------------------------------------------------------------------- #
# Model registry -- keyed per engine, because the GGUF formats differ.
# The NVIDIA (nemo-speech) and mudler (parakeet.cpp) files are conversions of the
# same checkpoint; sizes/hashes come from the respective Hugging Face repos.
# --------------------------------------------------------------------------- #
MODELS: dict[str, dict[str, Any]] = {
    "parakeet-tdt": {
        "default": True,
        "transformers": "nvidia/parakeet-tdt-0.6b-v3",
        "nemo-speech": {
            "repo": "nvidia/parakeet-tdt-0.6b-v3",
            "filename": "parakeet-tdt-0.6b-v3.q8_0.gguf",
            "size": 713975456,
            "sha256": "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e",
        },
        "parakeet-cpp": {
            "repo": "mudler/parakeet-cpp-gguf",
            "filename": "tdt-0.6b-v3-q8_0.gguf",
            "size": 940663680,
            "sha256": "4d69a4a6683f4f2d952bad794c1357ca6eb628027695b4699c5a9ad4cd07d757",
        },
    },
    "nemotron-3.5": {
        "transformers": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "nemo-speech": {
            "repo": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "filename": "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
            "size": 741548352,
            "sha256": "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae",
        },
        "parakeet-cpp": {
            "repo": "mudler/parakeet-cpp-gguf",
            "filename": "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
            "size": 983696512,
            "sha256": "ba2f13eccd4a5245be728f77e6149bd6a4fdcdd133ff2e08ac6005bcef7a99f1",
        },
    },
    "nemotron-en": {
        "transformers": "nvidia/nemotron-speech-streaming-en-0.6b",
        "nemo-speech": {
            "repo": "nvidia/nemotron-speech-streaming-en-0.6b",
            "filename": "nemotron-speech-streaming-en-0.6b.q8_0.gguf",
            "size": 699872960,
            "sha256": "d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d",
        },
    },
    "parakeet-ctc": {
        "transformers": "nvidia/parakeet-ctc-1.1b",
        "nemo-speech": {
            "repo": "nvidia/parakeet-ctc-1.1b",
            "filename": "parakeet-ctc-1.1b.q8_0.gguf",
            "size": 1178100960,
            "sha256": "6584fc0fdacf1c220401ea4c3a1d5b44454b655c141cb8672178072c203d92b8",
        },
        "parakeet-cpp": {
            "repo": "mudler/parakeet-cpp-gguf",
            "filename": "ctc-1.1b-q8_0.gguf",
            "size": 1526301120,
            "sha256": "62f9d341173a548af3d5a26aa44aacc34bd2126706f032a25b889f249b51da09",
        },
    },
}
MODEL_ALIASES = {a: n for n, m in MODELS.items() for a in (n, m.get("transformers", "")) if a}
ENGINES = ("parakeet-cpp", "nemo-speech", "transformers")
DEFAULT_MODEL_NAME = next(n for n, m in MODELS.items() if m.get("default"))

# A file already on disk under a different (legacy/local) name.
LEGACY_LOCAL: dict[tuple[str, str], tuple[str, ...]] = {
    ("parakeet-tdt", "parakeet-cpp"): ("parakeet-tdt-0.6b-v3-q8_0.gguf",),
}

ENV_PARAKEET_CLI = "PARAKEET_CLI"
ENV_NEMO_SPEECH = "NEMO_SPEECH_CLI"
ENV_TORCH_PYTHON = "PARAKEET_TORCH_PYTHON"


# --------------------------------------------------------------------------- #
# Executable discovery (portable: env var, then PATH, then user-local bins)
# --------------------------------------------------------------------------- #
def _user_bin_dirs() -> list[Path]:
    dirs = [Path.home() / ".local" / "bin"]
    if sys.platform == "darwin":
        dirs += [Path("/opt/homebrew/bin"), Path("/usr/local/bin")]
    if sys.platform.startswith("win"):
        appdata = os.environ.get("LOCALAPPDATA")
        if appdata:
            dirs.append(Path(appdata) / "Programs")
    return dirs


def find_executable(name: str, env_var: str | None = None, extra: Sequence[str] = ()) -> str | None:
    """Locate ``name`` via explicit path -> env var -> PATH -> user-local bins."""
    candidates: list[str] = list(extra)
    if env_var and os.environ.get(env_var):
        candidates.append(os.environ[env_var])
    if which := shutil.which(name):
        candidates.append(which)
    for d in _user_bin_dirs():
        candidates.append(str(d / name))
    # NeMo-Speech.cpp installer layout
    candidates.append(str(Path.home() / ".local" / "share" / "nemo-speech" / "bin" / name))
    for cand in candidates:
        path = Path(cand).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def find_parakeet_cli(explicit: str | os.PathLike | None = None) -> str | None:
    return find_executable("parakeet-cli", ENV_PARAKEET_CLI, [str(explicit)] if explicit else [])


def find_nemo_speech(explicit: str | os.PathLike | None = None) -> str | None:
    return find_executable("nemo-speech", ENV_NEMO_SPEECH, [str(explicit)] if explicit else [])


def find_torch_python(explicit: str | None = None) -> str | None:
    """A Python interpreter, preferring one that can import torch+transformers."""
    candidates = [explicit, os.environ.get(ENV_TORCH_PYTHON)]
    if venv := os.environ.get("VIRTUAL_ENV"):
        candidates.append(str(Path(venv) / "bin" / "python"))
    candidates += [sys.executable, shutil.which("python3"), shutil.which("python")]
    existing = [c for c in candidates if c and Path(c).is_file()]
    for cand in existing:
        if _python_has_torch(cand):
            return cand
    return existing[0] if existing else None


def _python_has_torch(python: str) -> bool:
    try:
        proc = subprocess.run([python, "-c", "import torch, transformers"],
                              capture_output=True, text=True, timeout=60)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def available_engines() -> dict[str, str]:
    """Map engine name -> executable/interpreter (only usable engines)."""
    found: dict[str, str] = {}
    if cli := find_parakeet_cli():
        found["parakeet-cpp"] = cli
    if ns := find_nemo_speech():
        found["nemo-speech"] = ns
    py = find_torch_python()
    if py and TRANSFORMERS_HELPER.is_file() and _python_has_torch(py):
        found["transformers"] = py
    return found


# --------------------------------------------------------------------------- #
# Transcript model (mirrors NeMo's Hypothesis / nemo-speech JSON)
# --------------------------------------------------------------------------- #
def _num(value: float, nd: int = 3):
    """Round and drop a trailing ``.0`` -- nemo-speech prints ``6`` not ``6.0``."""
    value = round(float(value), nd)
    return int(value) if float(value).is_integer() else value


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": _num(self.start, 3),
            "end": _num(self.end, 3),
            "confidence": _num(self.confidence, 4),
        }


@dataclass
class Hypothesis:
    """Mirrors the return of ``asr_model.transcribe([...])[0]``.

    ``.text`` is the punctuated/capitalised transcript; ``.timestamp`` holds
    ``{"word": [...], "segment": [...]}`` in NeMo's layout.
    """

    text: str
    file: str = ""
    duration: float = 0.0
    confidence: float = 1.0
    languages: list[str] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    tokens: list[dict[str, Any]] = field(default_factory=list)
    engine: str = ""

    @property
    def timestamp(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "word": [
                {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                for w in self.words
            ],
            "segment": list(self.segments),
        }

    def as_json_dict(self) -> dict[str, Any]:
        """The exact key set/order ``nemo-speech transcribe -f json`` emits."""
        return {
            "file": self.file,
            "text": self.text,
            "confidence": _num(self.confidence, 4),
            "duration": _num(self.duration, 3),
            "languages": list(self.languages),
            "words": [w.as_dict() for w in self.words],
        }


# --------------------------------------------------------------------------- #
# Model resolution / download (stdlib only, so clone-and-run needs no pip)
# --------------------------------------------------------------------------- #
def _resolve_name(spec: str | os.PathLike | None) -> str | None:
    if spec is None:
        return DEFAULT_MODEL_NAME
    return MODEL_ALIASES.get(str(spec))


def _local_candidates(model_name: str, engine: str) -> list[str]:
    """Filenames this engine would accept for ``model_name`` if already present."""
    entry = MODELS.get(model_name, {}).get(engine)
    names = [entry["filename"]] if isinstance(entry, dict) else []
    names += list(LEGACY_LOCAL.get((model_name, engine), ()))
    return names


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, *, quiet: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "streamangel-asr"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = last = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if not quiet and total and done - last >= (100 << 20):
                last = done
                print(f"  {_human(done)} / {_human(total)}", file=sys.stderr, flush=True)
    tmp.replace(dest)


def ensure_model(
    spec: str | os.PathLike | None,
    engine: str,
    *,
    allow_download: bool = True,
    quiet: bool = False,
    verify: bool = True,
) -> Path:
    """Return the GGUF path for ``engine``, downloading it from HF if needed."""
    name = _resolve_name(spec)
    if name is None:
        path = Path(str(spec)).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"model not found: {path}")

    for filename in _local_candidates(name, engine):
        if (candidate := MODELS_DIR / filename).is_file():
            return candidate

    entry = MODELS[name].get(engine)
    if not isinstance(entry, dict):
        raise FileNotFoundError(
            f"no {engine} build of '{name}' is published here; pass --model <path>. "
            f"See `model list`."
        )
    dest = MODELS_DIR / entry["filename"]
    if not allow_download:
        raise FileNotFoundError(f"missing {dest.name}; run `pull {name}` or drop --offline")
    url = f"https://huggingface.co/{entry['repo']}/resolve/main/{entry['filename']}"
    if not quiet:
        print(f"downloading {entry['repo']}/{entry['filename']} (~{_human(entry['size'])}) -> {dest}",
              file=sys.stderr, flush=True)
    _download(url, dest, quiet=quiet)
    if verify and entry.get("sha256"):
        digest = _hash_file(dest)
        if digest != entry["sha256"]:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {dest.name} (got {digest[:12]}...)")
    return dest


# --------------------------------------------------------------------------- #
# Audio normalisation
# --------------------------------------------------------------------------- #
def _to_wav16k(src: Path, dst: Path) -> None:
    """Normalise input to 16 kHz mono PCM WAV (what the engines expect)."""
    if src.resolve() == dst.resolve():
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if src.suffix.lower() == ".wav":
            shutil.copyfile(src, dst)
            return
        raise RuntimeError("ffmpeg not found; only WAV input is supported without it")
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True, text=True,
    )


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0


def _decode_json(stdout: str) -> dict[str, Any]:
    """Engines write progress to stderr and one JSON object to stdout."""
    text = stdout.strip()
    start = text.find("{")
    if start == -1:
        raise RuntimeError(f"engine produced no JSON: {text[:400]!r}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


# --------------------------------------------------------------------------- #
# Engine runners
# --------------------------------------------------------------------------- #
def _run_parakeet_cpp(cli: str, model: Path, wav: Path, language: str | None = None) -> dict[str, Any]:
    cmd = [cli, "transcribe", "--model", str(model), "--input", str(wav), "--json"]
    if language:
        cmd += ["--lang", language]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"parakeet-cli failed ({proc.returncode}):\n{(proc.stderr or proc.stdout)[-800:]}")
    if re.search(r"no ROCm-capable device|failed to initialize ROCm", proc.stderr or ""):
        print(
            "warning: no ROCm device is visible to this user -- parakeet-cli fell back to CPU. "
            "On Linux add the user to the 'render' group (or use a GPU-enabled account).",
            file=sys.stderr,
        )
    return _decode_json(proc.stdout)


def _run_nemo_speech(binary: str, model: Path, wav: Path,
                     language: str | None = None, device: str | None = None) -> dict[str, Any]:
    cmd = [binary, "transcribe", str(wav), "--model", str(model), "-f", "json", "--quiet"]
    if language:
        cmd += ["--language", language]
    if device and device != "auto":
        cmd += ["--device", device]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nemo-speech failed ({proc.returncode}):\n{(proc.stderr or proc.stdout)[-800:]}")
    return _decode_json(proc.stdout)


def _run_transformers(repo: str, wav: Path, language: str | None = None,
                      torch_python: str | None = None) -> dict[str, Any]:
    if not TRANSFORMERS_HELPER.is_file():
        raise RuntimeError(f"missing helper: {TRANSFORMERS_HELPER}")
    py = find_torch_python(torch_python)
    if py is None:
        raise RuntimeError(f"no Python with torch+transformers found; set {ENV_TORCH_PYTHON}")
    cmd = [py, str(TRANSFORMERS_HELPER), "--model", repo, "--input", str(wav), "--json"]
    if language:
        cmd += ["--lang", language]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"transformers engine failed ({proc.returncode}):\n{(proc.stderr or proc.stdout)[-800:]}")
    return _decode_json(proc.stdout)


# --------------------------------------------------------------------------- #
# Normalisation -> Hypothesis
# --------------------------------------------------------------------------- #
def _normalise(raw: dict[str, Any], file: str, duration: float, engine: str) -> Hypothesis:
    words: list[Word] = []
    for item in raw.get("words", []) or []:
        words.append(
            Word(
                word=item.get("word", item.get("w", "")),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                confidence=float(item.get("confidence", item.get("conf", 1.0))),
            )
        )
    conf = float(raw.get("confidence", 1.0))
    if "confidence" not in raw and words:
        conf = sum(w.confidence for w in words) / len(words)
    hyp = Hypothesis(
        text=str(raw.get("text", "")).strip(),
        file=file,
        duration=duration or float(raw.get("duration", 0.0)),
        confidence=conf,
        languages=list(raw.get("languages", []) or []),
        words=words,
        tokens=list(raw.get("tokens", []) or []),
        engine=engine,
    )
    hyp.segments = _segments_from_words(hyp.words, hyp.duration)
    return hyp


# --------------------------------------------------------------------------- #
# Subtitle grouping (approximates NeMo-Speech.cpp's cue layout)
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"[.!?]['\"\u201d\u2019)]?$")


def _segments_from_words(words: list[Word], duration: float = 0.0) -> list[dict[str, Any]]:
    """Group words into subtitle-like segments.

    Heuristic tuned against ``nemo-speech`` output on the model card's sample:
    cut at a sentence end, at a pause >= 0.7 s, at a cue longer than ~4 s, or when
    a cue would exceed two ~29-character lines.  A cue's end is stretched to the
    start of the next word (as NeMo-Speech.cpp does), so trailing silence belongs
    to the preceding cue.
    """
    segments: list[dict[str, Any]] = []
    cue: list[Word] = []
    max_chars, max_dur, max_gap = 58, 4.0, 0.7

    def flush(next_start: float | None) -> None:
        if not cue:
            return
        text = " ".join(w.word for w in cue).strip()
        end = next_start if next_start is not None else (duration or cue[-1].end)
        segments.append({"start": round(cue[0].start, 3), "end": round(end, 3), "segment": text})

    for i, w in enumerate(words):
        if cue:
            gap = w.start - cue[-1].end
            projected = len(" ".join(x.word for x in cue + [w]))
            if projected > max_chars or (cue[-1].end - cue[0].start) >= max_dur or gap >= max_gap:
                flush(w.start)
                cue = []
        cue.append(w)
        if _SENTENCE_END.search(w.word) and len(cue) >= 3:
            nxt = words[i + 1].start if i + 1 < len(words) else None
            flush(nxt)
            cue = []
    flush(None)
    return segments


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _wrap(text: str, width: int = 29, max_lines: int = 2) -> list[str]:
    """Wrap a cue into at most ``max_lines`` lines.

    NeMo-Speech.cpp renders two balanced lines when a cue needs them, so a
    midpoint word-boundary split is used before falling back to greedy wrap.
    """
    if len(text) <= width:
        return [text]
    if max_lines == 2 and len(text) <= 2 * width:
        mid, best = len(text) / 2, None
        for i, ch in enumerate(text):
            if ch == " " and (best is None or abs(i - mid) < abs(best - mid)):
                best = i
        if best is not None:
            return [text[:best], text[best + 1:]]

    import textwrap

    lines = textwrap.wrap(text, width=width) or [text]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return lines


def _fmt_ts_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ts_vtt(seconds: float) -> str:
    return _fmt_ts_srt(seconds).replace(",", ".")


def render_text(hyp: Hypothesis) -> str:
    return hyp.text + "\n"


def render_json(hyp: Hypothesis) -> str:
    """Byte-for-byte the layout ``nemo-speech transcribe -f json`` emits."""
    d = hyp.as_json_dict()
    lines = ["{"]
    for key in ("file", "text", "confidence", "duration", "languages"):
        lines.append(f"  {json.dumps(key)}: {json.dumps(d[key], ensure_ascii=False)},")
    lines.append('  "words": [')
    words = d["words"]
    for i, word in enumerate(words):
        lines.append("    " + json.dumps(word, ensure_ascii=False) + ("," if i < len(words) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_srt(hyp: Hypothesis) -> str:
    segs = hyp.segments or _segments_from_words(hyp.words, hyp.duration)
    out: list[str] = []
    for n, seg in enumerate(segs, 1):
        out.append(str(n))
        out.append(f"{_fmt_ts_srt(seg['start'])} --> {_fmt_ts_srt(seg['end'])}")
        out.extend(_wrap(seg["segment"]))
        out.append("")
    return "\n".join(out) + "\n"


def render_vtt(hyp: Hypothesis) -> str:
    segs = hyp.segments or _segments_from_words(hyp.words, hyp.duration)
    out = ["WEBVTT", ""]
    for seg in segs:
        out.append(f"{_fmt_ts_vtt(seg['start'])} --> {_fmt_ts_vtt(seg['end'])}")
        out.extend(_wrap(seg["segment"]))
        out.append("")
    return "\n".join(out) + "\n"


RENDERERS = {"text": render_text, "json": render_json, "srt": render_srt, "vtt": render_vtt}


# --------------------------------------------------------------------------- #
# High-level API (documented Python-parity entry point)
# --------------------------------------------------------------------------- #
def transcribe(
    audio: str | os.PathLike | Sequence[str | os.PathLike],
    model: str | os.PathLike | None = None,
    *,
    timestamps: bool = True,
    engine: str = "auto",
    language: str | None = None,
    device: str | None = None,
    cli: str | None = None,
    nemo_speech: str | None = None,
    torch_python: str | None = None,
    allow_download: bool = True,
    quiet: bool = False,
) -> list[Hypothesis]:
    """Transcribe one or more files.

    Mirrors the model card's ``asr_model.transcribe([...], timestamps=True)``:
    returns a list of :class:`Hypothesis`, each with ``.text`` and
    ``.timestamp['word'|'segment']``.
    """
    files = [Path(audio)] if isinstance(audio, (str, os.PathLike)) else [Path(a) for a in audio]
    results: list[Hypothesis] = []
    with tempfile.TemporaryDirectory(prefix="streamangel-asr-") as tmp:
        for src in files:
            wav = Path(tmp) / (src.stem + ".16k.wav")
            _to_wav16k(src, wav)
            results.append(_transcribe_one(
                engine, src, wav, _wav_duration(wav), model,
                language, device, cli, nemo_speech, torch_python, allow_download, quiet))
    return results


def _transcribe_one(engine, src, wav, duration, model_spec, language, device,
                    cli, nemo_speech, torch_python, allow_download, quiet) -> Hypothesis:
    order = [engine] if engine != "auto" else list(ENGINES)
    errors: list[str] = []

    for name in order:
        try:
            if name == "transformers":
                model_name = _resolve_name(model_spec) or DEFAULT_MODEL_NAME
                repo = MODELS[model_name].get("transformers", "nvidia/parakeet-tdt-0.6b-v3")
                raw = _run_transformers(repo, wav, language, torch_python)
            elif name == "nemo-speech":
                binary = find_nemo_speech(nemo_speech)
                if not binary:
                    raise FileNotFoundError("nemo-speech not installed")
                model_path = ensure_model(model_spec, name, allow_download=allow_download, quiet=quiet)
                raw = _run_nemo_speech(binary, model_path, wav, language, device)
            else:
                binary = find_parakeet_cli(cli)
                if not binary:
                    raise FileNotFoundError("parakeet-cli not installed")
                model_path = ensure_model(model_spec, name, allow_download=allow_download, quiet=quiet)
                raw = _run_parakeet_cpp(binary, model_path, wav, language)
            return _normalise(raw, str(src), duration, name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            if engine != "auto":
                raise

    raise RuntimeError(
        "no working ASR engine found.\n  " + "\n  ".join(errors)
        + "\nInstall one of:\n"
        "  * parakeet.cpp   (ROCm/HIP, CUDA, Vulkan, Metal, CPU)  -> parakeet-cli\n"
        "  * NeMo-Speech.cpp (CPU, CUDA, Metal, Vulkan)           -> nemo-speech\n"
        "  * PyTorch + transformers (CUDA/ROCm) for the original checkpoint\n"
        "Then re-run; `doctor` reports what was detected."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _emit(msg: str, *, quiet: bool, err: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def _cmd_transcribe(args: argparse.Namespace) -> int:
    inputs: list[Path] = []
    for entry in args.input:
        p = Path(entry)
        if p.is_dir():
            it = p.rglob("*.wav") if args.recursive else p.glob("*.wav")
            inputs.extend(sorted(it))
        else:
            inputs.append(p)
    if not inputs:
        raise SystemExit("no input audio found")

    hyps = transcribe(
        inputs, model=args.model, engine=args.engine, language=args.language,
        device=args.device, cli=args.cli, nemo_speech=args.nemo_speech,
        torch_python=args.torch_python, allow_download=not args.offline, quiet=args.quiet,
    )
    if args.confidence == "nemo":
        for hyp in hyps:
            hyp.confidence = 1.0
            for word in hyp.words:
                word.confidence = 1.0
    if args.verbose:
        for hyp in hyps:
            _emit(f"{Path(hyp.file).name}: engine={hyp.engine} words={len(hyp.words)}", quiet=False, err=True)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for hyp in hyps:
        rendered = RENDERERS[args.format](hyp)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
            _emit(f"wrote {args.output}", quiet=args.quiet, err=True)
        elif out_dir:
            ext = {"text": "txt", "json": "json", "srt": "srt", "vtt": "vtt"}[args.format]
            dest = out_dir / f"{Path(hyp.file).stem}.{ext}"
            dest.write_text(rendered, encoding="utf-8")
            _emit(f"wrote {dest}", quiet=args.quiet, err=True)
        else:
            sys.stdout.write(rendered)
    return 0


def _gpu_report() -> list[str]:
    notes: list[str] = []
    if sys.platform.startswith("linux"):
        kfd = Path("/dev/kfd")
        if kfd.exists():
            ok = os.access(kfd, os.R_OK | os.W_OK)
            notes.append("ROCm device present (/dev/kfd)"
                         + ("" if ok else " but not writable by this user -> CPU fallback; "
                                          "add the user to the 'render' group"))
        if shutil.which("rocminfo"):
            try:
                out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20).stdout
                if archs := re.findall(r"Name:\s+(gfx\w+)", out):
                    notes.append("ROCm archs: " + ", ".join(sorted(set(archs))))
            except Exception:  # noqa: BLE001
                pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
            if out.strip():
                notes.append("CUDA: " + out.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            pass
    return notes


def _model_rows() -> list[dict[str, Any]]:
    rows = []
    for name, spec in MODELS.items():
        rows.append({
            "name": name,
            "default": bool(spec.get("default")),
            "transformers": spec.get("transformers"),
            "files": {
                engine: {
                    "repo": spec[engine]["repo"],
                    "filename": spec[engine]["filename"],
                    "present": any((MODELS_DIR / f).is_file() for f in _local_candidates(name, engine)),
                }
                for engine in ENGINES if isinstance(spec.get(engine), dict)
            },
        })
    return rows


def _cmd_doctor(args: argparse.Namespace) -> int:
    engines = available_engines()
    preferred = next((e for e in ENGINES if e in engines), None)
    data = {
        "version": f"streamangel-asr {__version__}",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "engines": engines,
        "default_engine": preferred,
        "models_dir": str(MODELS_DIR),
        "models": _model_rows(),
        "ffmpeg": shutil.which("ffmpeg"),
        "gpu": _gpu_report(),
    }
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(data["version"])
    print(f"platform      : {data['platform']} (python {data['python']})")
    if engines:
        for name, path in engines.items():
            print(f"engine        : {name:<13} {path}")
    else:
        print("engine        : NONE FOUND -- install parakeet.cpp or NeMo-Speech.cpp (see README)")
    print(f"models        : {data['models_dir']}")
    print(f"ffmpeg        : {data['ffmpeg'] or 'not found (WAV-only input)'}")
    for note in data["gpu"] or ["no GPU tooling detected (ROCm/nvidia-smi); CPU works for testing"]:
        print(f"gpu           : {note}")
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    rows = _model_rows()
    if args.subcommand == "inspect" or args.json:
        print(json.dumps({"models_dir": str(MODELS_DIR), "models": rows}, indent=2))
        return 0
    print("Available models (* = default)\n")
    print("ASR -- transcribe")
    for row in rows:
        mark = "*" if row["default"] else " "
        print(f"  {mark} {row['name']}")
        for engine, info in row["files"].items():
            state = "present" if info["present"] else "not downloaded"
            print(f"      {engine:<13} {info['repo']}/{info['filename']} ({state})")
    print("\nFetch one with:  pull <name> [--engine <engine>]")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    engines = available_engines()
    engine = args.engine if args.engine != "auto" else next((e for e in ENGINES if e in engines), "nemo-speech")
    path = ensure_model(args.model, engine, allow_download=True, quiet=args.quiet)
    print(json.dumps({"file": str(path), "engine": engine}, indent=2) if args.json
          else f"ready ({engine}): {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Global flags accepted both before and after the subcommand (as nemo-speech
    # does). SUPPRESS keeps a post-subcommand flag from clobbering a pre-set one.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="Emit machine-readable results and errors")
    common.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="Suppress non-result progress messages")
    common.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="Emit additional diagnostics on stderr")

    p = argparse.ArgumentParser(
        prog="nemo-speech",
        parents=[common],
        description="Drop-in for NVIDIA's documented Nemotron Speech runtime, on any GPU backend.",
    )
    p.add_argument("--version", action="version", version=f"streamangel-asr {__version__}")
    sub = p.add_subparsers(dest="command")

    t = sub.add_parser("transcribe", parents=[common], help="Transcribe an audio file or directory")
    t.add_argument("input", nargs="+", help="WAV file(s) or directory")
    t.add_argument("-m", "--model", default=None,
                   help=f"GGUF path, model name ({', '.join(MODELS)}), or HF repo id")
    t.add_argument("-l", "--language", default=None, help="Language code or prompt")
    t.add_argument("--engine", default="auto", choices=["auto", *ENGINES])
    t.add_argument("--cli", default=None, help="Path to parakeet-cli")
    t.add_argument("--nemo-speech", default=None, help="Path to the nemo-speech binary")
    t.add_argument("--torch-python", default=None, help="Python interpreter with torch+transformers")
    t.add_argument("--device", "--backend", dest="device", default="auto",
                   help="auto|cpu|cuda|rocm|vulkan (passed to nemo-speech; parakeet.cpp auto-selects)")
    t.add_argument("-f", "--format", default="text", choices=list(RENDERERS))
    t.add_argument("-o", "--output", default=None, help="Output path for one input")
    t.add_argument("--output-dir", default=None, help="Preserve layout under DIR")
    t.add_argument("-r", "--recursive", action="store_true")
    t.add_argument("--word-times", action="store_true", help="Compatibility flag; JSON implies it")
    t.add_argument("--confidence", default="engine", choices=["engine", "nemo"],
                   help="engine: real per-word posterior (default); nemo: constant 1")
    t.add_argument("--offline", action="store_true", help="Never download a model")
    t.add_argument("--stream", action="store_true", help="Compatibility flag (chunking is upstream)")
    t.add_argument("--no-punctuation", action="store_true", help="Compatibility flag")
    t.add_argument("--profanity-filter", action="store_true", help="Compatibility flag")
    t.set_defaults(func=_cmd_transcribe)

    d = sub.add_parser("doctor", parents=[common], help="Show detected engines and devices")
    d.set_defaults(func=_cmd_doctor)

    m = sub.add_parser("model", parents=[common], help="List or inspect models")
    m.add_argument("subcommand", nargs="?", choices=["list", "inspect"], default="list")
    m.set_defaults(func=_cmd_model)

    pl = sub.add_parser("pull", parents=[common], help="Download a model ahead of time")
    pl.add_argument("model", nargs="?", default=DEFAULT_MODEL_NAME)
    pl.add_argument("--engine", default="auto", choices=["auto", *ENGINES])
    pl.set_defaults(func=_cmd_pull)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    for name in ("json", "quiet", "verbose"):
        if not hasattr(args, name):
            setattr(args, name, False)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
