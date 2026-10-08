# StreamAngel ASR — the documented Nemotron interface, on any GPU backend

The `nvidia/parakeet-tdt-0.6b-v3` model card documents NVIDIA's `nemo-speech`
(NeMo-Speech.cpp) as the native C++ runtime:

```bash
nemo-speech transcribe audio.wav --model models/parakeet-tdt-0.6b-v3.q8_0.gguf
```

`nemo_speech_rocm.py` provides the *same command surface* and the *same return
shapes*, but does not care which backend produces them. It discovers whatever is
installed and uses it, so it works on **ROCm/HIP, CUDA, Vulkan, Metal or CPU**.

## Engines

| Engine | Binary | Backends | Model source |
|---|---|---|---|
| `parakeet-cpp` *(preferred)* | `parakeet-cli` from [mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp) | ROCm/HIP, CUDA, Vulkan, Metal, CPU | `mudler/parakeet-cpp-gguf` |
| `nemo-speech` | `nemo-speech` from [NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) | CPU, CUDA, Metal, Vulkan *(no ROCm upstream)* | `nvidia/parakeet-tdt-0.6b-v3` |
| `transformers` | a venv with `torch` + `transformers` | CUDA, ROCm | HF checkpoint (safetensors) |

Detection order for `--engine auto` is `parakeet-cpp → nemo-speech →
transformers`. Force one with `--engine <name>`. Discovery is portable: explicit
path (`--cli` / `PARAKEET_CLI`, `--nemo-speech` / `NEMO_SPEECH_CLI`, `--torch-python`
/ `PARAKEET_TORCH_PYTHON`), then `PATH`, then the usual user-local bin dirs.
Nothing is hard-coded to a particular machine, GPU or OS.

### The two C++ runtimes use **different GGUF files**

They are conversions of the same checkpoint but carry different tensor layouts
(parakeet.cpp's metadata uses `parakeet.*` keys; NeMo-Speech.cpp's uses `asr.*`).
Feeding the wrong one gives *"cannot load model"*. So each engine downloads its
own file — the registry in `nemo_speech_rocm.py` knows both, with sizes and
SHA-256 checks from the respective Hugging Face repos. They produce the same
transcript.

```bash
python3 nemo_speech_rocm.py model list                 # per-engine sources + local state
python3 nemo_speech_rocm.py pull parakeet-tdt --engine parakeet-cpp
```

## Usage

```bash
python3 nemo_speech_rocm.py doctor                     # engines, GPU, models
python3 nemo_speech_rocm.py transcribe a.wav           # text
python3 nemo_speech_rocm.py transcribe a.wav -f json   # file/text/confidence/duration/languages/words
python3 nemo_speech_rocm.py transcribe a.wav -f srt    # or vtt
python3 nemo_speech_rocm.py transcribe clips/ --output-dir out -r
python3 nemo_speech_rocm.py transcribe a.wav --confidence nemo   # constant-1 confidence
```

```python
import nemo_speech_rocm as ns
out = ns.transcribe(["a.wav"], timestamps=True)   # documented Python parity
out[0].text, out[0].timestamp["word"], out[0].timestamp["segment"]
```

Non-WAV input is transcoded with `ffmpeg` when available; WAV is handled natively.

## Verified parity vs `nemo-speech` 0.2.0

Measured on the model card's own sample (`2086-149220-0033.wav`), parakeet.cpp
engine vs the real `nemo-speech`:

| Check | Result |
|---|---|
| `-f text` | **byte-identical** |
| `-f srt` / `-f vtt` | **byte-identical** |
| JSON top-level keys | **identical** (`file,text,confidence,duration,languages,words`) |
| JSON word keys | **identical** (`word,start,end,confidence`) |
| word count / text / duration | **identical** (23 / 23, 7.435 s) |
| JSON word *end* times | a few differ by ≤ 0.08 s (one frame — engine frame estimate) |
| `.timestamp['word'|'segment']` | available in Python |

`confidence` is the engine's real per-word posterior by default; `--confidence
nemo` emits a constant `1` to match that build exactly.

The SRT/VTT **cue splitter** is a reimplementation of NeMo-Speech.cpp's, so on
longer or live audio cue *boundaries* can differ (cue counts track closely).
Text and JSON parity are exact; cue rendering is cosmetic.

## Throughput (RX 6950 XT, gfx1030)

| Engine | 60 s clip | 300 s clip | peak RAM |
|---|---|---|---|
| `parakeet-cpp` (HIP, Q8_0) | ~2.4 s | ~8.5 s | ~1.2 GB |
| `transformers` (ROCm, fp16) | ~36 s | ~24 s | ~3.6 GB |

Transcripts agree: byte-identical on a 60 s clip, 0.11 % WER on 300 s. The C++
engine starts far faster and uses ~⅓ the memory, so it is preferred. Reproduce
with `benchmarks/bench_engines.py`.

## GPU access (the one thing that bites)

If the engine cannot see a device it **falls back to CPU silently** — the
transcript is still correct, just slow. The CLI prints a warning when it detects
this, and `doctor` reports it.

* **Linux + ROCm**: `/dev/kfd` is `root:render` `0660`. A user outside the
  `render` group (and without the logind ACL) has no GPU. Fix:
  `sudo usermod -aG render <user>` then re-login. `rocminfo` printing
  *"Unable to open /dev/kfd"* is the tell.
* **NVIDIA**: `nvidia-smi -L` should list your card.
* **Vulkan**: both C++ runtimes can use it on AMD/Intel/NVIDIA.

## Notes

* The ASR layer is **Python-stdlib only** — no pip install required. `ffmpeg`
  is optional (needed only for non-WAV input).
* Model files are git-ignored; they are large and are fetched on demand.
* `pip install .` also exposes a `streamangel-asr` console command.
