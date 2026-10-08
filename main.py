import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "parakeet-tdt-0.6b-v3-q8_0.gguf"
DEFAULT_AUDIO = PROJECT_ROOT / "2086-149220-0033.wav"
LOCAL_CLI = PROJECT_ROOT / "parakeet.cpp" / "build" / "examples" / "cli" / "parakeet-cli"


def transcribe(audio_path: Path, model_path: Path, cli_path: str) -> dict[str, object]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Parakeet GGUF model not found: {model_path}")

    cli = shutil.which(cli_path)
    if cli is None and LOCAL_CLI.is_file():
        cli = str(LOCAL_CLI)
    if cli is None:
        raise FileNotFoundError(
            f"Could not find parakeet-cli ({cli_path}). Build parakeet.cpp with "
            "`-DPARAKEET_GGML_HIP=ON`, or pass its path with --cli."
        )

    result = subprocess.run(
        [
            cli,
            "transcribe",
            "--model",
            str(model_path),
            "--input",
            str(audio_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"parakeet-cli failed with exit code {result.returncode}:\n"
            f"{result.stderr or result.stdout}"
        )

    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with parakeet.cpp.")
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=DEFAULT_AUDIO,
        help=f"audio file to transcribe (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Parakeet GGUF model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--cli",
        default=os.environ.get("PARAKEET_CLI", "parakeet-cli"),
        help="parakeet-cli executable path (default: PARAKEET_CLI or PATH)",
    )
    args = parser.parse_args()

    transcription = transcribe(args.audio, args.model, args.cli)
    print(json.dumps(transcription, indent=2))


if __name__ == "__main__":
    main()
