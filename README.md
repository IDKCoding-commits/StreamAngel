# StreamAngel

StreamAngel is PodAngel's experimental sister project for real-time podcast discovery and streaming.

- **PodAngel:** [IDKCoding-commits/PodAngel](https://github.com/IDKCoding-commits/PodAngel)

This repository is intentionally private while the project is experimental.

---

## Speech recognition (plug and play, any GPU backend)

StreamAngel transcribes with NVIDIA's **Parakeet TDT 0.6B v3** and presents the
interface NVIDIA documents for it — the same commands and the same output as
[`nemo-speech`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — but it runs
on whatever acceleration your machine actually has: **ROCm/HIP, CUDA, Vulkan,
Metal, or plain CPU**. No API keys, nothing leaves your machine.

```bash
# Nothing to install for the ASR layer itself: it is Python-stdlib only.
# 1) get an engine (any one of these):
#    parakeet.cpp     https://github.com/mudler/parakeet.cpp   -> parakeet-cli   (ROCm/HIP, CUDA, Vulkan, Metal, CPU)
#    NeMo-Speech.cpp  https://github.com/NVIDIA/NeMo-Speech.cpp -> nemo-speech    (CPU, CUDA, Metal, Vulkan)
# 2) run it:
python3 nemo_speech_rocm.py doctor           # what was detected + your GPU
python3 nemo_speech_rocm.py transcribe audio.wav            # text
python3 nemo_speech_rocm.py transcribe audio.wav -f json    # text + word/segment timestamps
python3 nemo_speech_rocm.py transcribe audio.wav -f srt     # subtitles (also vtt)
python3 nemo_speech_rocm.py transcribe clips/ --output-dir out -r
```

The model (≈940 MB / ≈714 MB depending on engine) downloads once, automatically,
on first use.

```python
import nemo_speech_rocm as ns
out = ns.transcribe(["audio.wav"], timestamps=True)   # engine="auto"
print(out[0].text)
print(out[0].timestamp["word"])                       # [{'word','start','end','confidence'}]
```

See **[docs/ASR.md](docs/ASR.md)** for the full reference: engines, per-engine
GGUF details, the exact JSON contract, GPU-access troubleshooting
(`/dev/kfd`, the `render` group), and accuracy/throughput benchmarks.

### Files

| Path | Role |
|---|---|
| `nemo_speech_rocm.py` | ASR CLI + Python API (engine-agnostic) |
| `parakeet_transformers.py` | optional engine: original checkpoint via PyTorch |
| `bin/nemo-speech` | launcher, so the documented `nemo-speech …` command works verbatim |
| `benchmarks/bench_engines.py` | engine head-to-head (wall time, throughput, agreement) |
| `functions.py`, `main.py` | podcast search / chunked transcription helpers |
