"""Generate per-scene voiceover WAVs via `npx hyperframes tts` and record durations."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

NPX = shutil.which("npx") or "npx.cmd"

OUT = Path(__file__).parent / "composition" / "assets" / "voiceover"
OUT.mkdir(parents=True, exist_ok=True)

SCRIPT = Path(__file__).parent / "voiceover-script.txt"


def parse_scenes(text: str) -> dict[str, str]:
    scenes: dict[str, str] = {}
    parts = re.split(r"=== (SCENE \d+) — .+ ===", text)
    for i in range(1, len(parts), 2):
        key = f"scene{int(parts[i].split()[1]):02d}"
        scenes[key] = " ".join(parts[i + 1].split())
    return scenes


def wav_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def main() -> int:
    scenes = parse_scenes(SCRIPT.read_text(encoding="utf-8"))
    only = sys.argv[1:] or list(scenes)
    durations = {}
    dur_file = Path(__file__).parent / "voiceover-durations.json"
    if dur_file.exists():
        durations = json.loads(dur_file.read_text())
    for name in only:
        wav = OUT / f"voice_{name[-2:]}.wav"
        if wav.exists():
            durations[name] = wav_duration(wav)
            print(f"[skip] {name} exists, {durations[name]:.2f}s", flush=True)
            continue
        print(f"[tts ] {name} ({len(scenes[name].split())} words) ...", flush=True)
        subprocess.run(
            [NPX, "hyperframes", "tts", scenes[name],
             "--voice", "af_heart", "--output", str(wav)],
            check=True, capture_output=True, text=True)
        durations[name] = wav_duration(wav)
        print(f"       -> {durations[name]:.2f}s", flush=True)
    dur_file.write_text(json.dumps(durations, indent=2))
    total = sum(durations.values())
    print(f"TOTAL narration: {total:.1f}s ({total/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
