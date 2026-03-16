"""
diarization.py — Speaker Diarization with Loudness Analysis
------------------------------------------------------------
Identifies speakers in an audio file, separates their audio,
and produces a timeline + loudness report.

Powered by: pyannote.audio | pydub | numpy | torch
"""

# ==============================================================
#  USER CONFIGURATION
#  ──────────────────
#  Edit the values in this section before running the script.
#  Everything else can be left as-is for most use cases.
# ==============================================================

# --- HuggingFace Token ----------------------------------------
#
#  REQUIRED. The model will not load without a valid token.
#
#  How to get one (free):
#    1. Sign up at https://huggingface.co/join
#    2. Go to https://huggingface.co/settings/tokens
#    3. Click "New token" → select "Read" access → copy it
#    4. Accept the model licence at:
#       https://huggingface.co/pyannote/speaker-diarization-3.1
#
#  RECOMMENDED: Set it as an environment variable so it never
#  appears in your source code:
#
#    Windows PowerShell : $env:HF_TOKEN="hf_xxxxxxxxxxxx"
#    Windows CMD        : set HF_TOKEN=hf_xxxxxxxxxxxx
#    macOS / Linux      : export HF_TOKEN=hf_xxxxxxxxxxxx
#
#  If you prefer to paste it directly (less secure), replace
#  the empty string below with your token:
#
HF_TOKEN_FALLBACK = ""   # e.g. "hf_xxxxxxxxxxxxxxxxxxxx"
#                          Leave empty if using environment variable.

# --- Input audio file -----------------------------------------
#
#  Full path to the audio file you want to process.
#
#  Leave as None to auto-detect the first audio file on the Desktop.
#
#  Examples:
#    Windows : r"C:\Users\<USERNAME>\Desktop\recording.mp3"
#    macOS   : "/Users/<USERNAME>/Desktop/recording.mp3"
#    Linux   : "/home/<USERNAME>/Desktop/recording.mp3"
#
#  Supported formats: .mp3  .wav  .m4a  .flac  .ogg  .aac  .wma
#
INPUT_AUDIO_FILE = None   # e.g. r"C:\Users\<USERNAME>\Desktop\audio.mp3"

# --- Output folder --------------------------------------------
#
#  Folder where all output files will be saved:
#    - Individual speaker WAV files  (speaker_00_only.wav, ...)
#    - Main speaker WAV file         (speaker_00_MAIN.wav)
#    - Diarization report            (<filename>_diarization_report.txt)
#
#  Leave as None to save to the Desktop automatically.
#
#  Examples:
#    Windows : r"C:\Users\<USERNAME>\Documents\diarization_output"
#    macOS   : "/Users/<USERNAME>/Documents/diarization_output"
#    Linux   : "/home/<USERNAME>/Documents/diarization_output"
#
#  The folder is created automatically if it does not exist.
#
OUTPUT_FOLDER = None   # e.g. r"C:\Users\<USERNAME>\Desktop\output"

# --- Speaker count hints (optional) ---------------------------
#
#  If you already know how many speakers are in the recording,
#  providing these values improves accuracy significantly.
#  Leave as None to let the model estimate automatically.
#
#  Example — exactly 2 speakers:
#    MIN_SPEAKERS = 2
#    MAX_SPEAKERS = 2
#
#  Example — 2 to 5 speakers, not sure exactly:
#    MIN_SPEAKERS = 2
#    MAX_SPEAKERS = 5
#
MIN_SPEAKERS = None   # int or None
MAX_SPEAKERS = None   # int or None

# ==============================================================
#  END OF USER CONFIGURATION — no need to edit below this line
# ==============================================================


import os
import sys
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment
from pyannote.audio import Pipeline


# ── Constants ──────────────────────────────────────────────────

CHUNK_MS        = 5_000                     # audio padding chunk (ms)
DEFAULT_DESKTOP = Path.home() / "Desktop"   # cross-platform Desktop path
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma")


# ══════════════════════════════════════════════════════════════
#  PREFLIGHT
# ══════════════════════════════════════════════════════════════

def check_ffmpeg() -> None:
    """
    Verify that ffmpeg is on the system PATH.
    pydub cannot decode MP3/M4A/AAC without it.
    """
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "\nffmpeg not found on PATH.\n"
            "Install it before running:\n"
            "  Windows : https://ffmpeg.org/download.html  (add the bin/ folder to PATH)\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg\n"
        )


def resolve_hf_token() -> str:
    """
    Resolve the HuggingFace token in this order:
      1. HF_TOKEN environment variable  (recommended)
      2. HF_TOKEN_FALLBACK constant above (less secure)
    Raises EnvironmentError if neither is set.
    """
    token = os.environ.get("HF_TOKEN", "").strip() or HF_TOKEN_FALLBACK.strip()
    if not token:
        raise EnvironmentError(
            "\nHuggingFace token not found.\n"
            "Set it as an environment variable:\n"
            "  Windows PowerShell : $env:HF_TOKEN=\"hf_xxxxxxxxxxxx\"\n"
            "  Windows CMD        : set HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "  macOS / Linux      : export HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "Or paste it into HF_TOKEN_FALLBACK at the top of this script.\n"
            "Get a free token at: https://huggingface.co/settings/tokens\n"
        )
    return token


# ══════════════════════════════════════════════════════════════
#  MODEL CACHE (optional utility)
# ══════════════════════════════════════════════════════════════

def clear_model_cache() -> bool:
    """
    Delete the local pyannote model cache to force a fresh download.
    Only call this if you suspect a corrupted download.
    Not invoked automatically.
    """
    cache_dir = Path.home() / ".cache" / "torch" / "pyannote"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("Model cache cleared — fresh models will download on next run.")
            return True
        except OSError as exc:
            print(f"Could not clear cache: {exc}")
            return False
    print("Cache directory not found — nothing to clear.")
    return False

# To wipe the model cache, uncomment the line below and run once:
# clear_model_cache()


# ══════════════════════════════════════════════════════════════
#  PIPELINE LOADER
# ══════════════════════════════════════════════════════════════

def load_pipeline(hf_token: str) -> Pipeline:
    """
    Load pyannote speaker-diarization pipeline.
    Tries v3.1 first; falls back to the base version if unavailable.
    Moves automatically to GPU if CUDA is available.
    """
    models_to_try = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/speaker-diarization",
    ]
    pipeline = None
    for model_id in models_to_try:
        try:
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
            print(f"  Loaded model : {model_id}")
            break
        except Exception as exc:
            print(f"  Could not load {model_id}: {exc}")

    if pipeline is None:
        raise RuntimeError(
            "\nFailed to load any pyannote pipeline. Common causes:\n"
            "  1. Token is invalid or expired\n"
            "  2. Model licence not accepted at:\n"
            "     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3. No internet connection (required for first-time download)\n"
            "  4. Outdated package — try: pip install --upgrade pyannote.audio\n"
        )

    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print("  Device : GPU (CUDA)")
        else:
            print("  Device : CPU (no CUDA found)")
    except ImportError:
        print("  Device : CPU (torch import failed)")

    return pipeline


# ══════════════════════════════════════════════════════════════
#  AUDIO PREPARATION
# ══════════════════════════════════════════════════════════════

def prepare_audio(audio_path: Path, tmp_dir: str) -> Tuple[AudioSegment, Path]:
    """
    Load audio, force mono, pad to a multiple of CHUNK_MS,
    and write a temporary WAV to the system temp directory.

    Using tempfile avoids permission issues on Desktop, network
    drives, or read-only locations.

    Returns:
        (AudioSegment of padded mono audio, Path to the temp WAV)
    """
    print(f"  Loading  : {audio_path.name}")
    try:
        audio = AudioSegment.from_file(str(audio_path)).set_channels(1)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load audio: {audio_path}\n"
            "Ensure ffmpeg is installed and the file is not corrupted.\n"
            f"Detail: {exc}"
        )

    remainder = len(audio) % CHUNK_MS
    if remainder:
        padding_ms = CHUNK_MS - remainder
        audio += AudioSegment.silent(duration=padding_ms)
        print(f"  Padded   : +{padding_ms} ms  →  {len(audio)} ms total")

    tmp_wav = Path(tmp_dir) / (audio_path.stem + "_tmp.wav")
    audio.export(str(tmp_wav), format="wav")
    print(f"  Temp WAV : {tmp_wav}")
    return audio, tmp_wav


# ══════════════════════════════════════════════════════════════
#  LOUDNESS METRICS
# ══════════════════════════════════════════════════════════════

def calculate_loudness_metrics(segment: AudioSegment) -> Dict[str, float]:
    """
    Compute RMS and peak amplitude for one audio segment.

    Handles:
      - 8-bit, 16-bit, 32-bit PCM
      - Stereo → mono downmix before calculation
      - Silent / zero-length segments (safe -100 dBFS defaults)

    Returns:
        rms, peak     — linear amplitude in [0, 1]
        rms_db, peak_db — same values in dBFS
    """
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)

    bit_depth_divisors = {1: 128.0, 2: 32768.0, 4: 2147483648.0}
    samples /= bit_depth_divisors.get(segment.sample_width, 32768.0)

    if segment.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    if samples.size == 0:
        return {"rms": 0.0, "peak": 0.0, "rms_db": -100.0, "peak_db": -100.0}

    rms  = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))
    eps  = 1e-10

    return {
        "rms":     rms,
        "peak":    peak,
        "rms_db":  20.0 * np.log10(rms  + eps),
        "peak_db": 20.0 * np.log10(peak + eps),
    }


# ══════════════════════════════════════════════════════════════
#  DIARIZATION
# ══════════════════════════════════════════════════════════════

def run_diarization(
    pipeline: Pipeline,
    tmp_wav: Path,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> object:
    """
    Run speaker diarization on the prepared mono WAV.

    min_speakers / max_speakers are optional hints.
    Providing them when you know the count improves accuracy.
    If omitted, pyannote estimates the speaker count automatically.
    """
    kwargs: Dict = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    hint = f"min={min_speakers}, max={max_speakers}" if kwargs else "auto-detect"
    print(f"  Speaker count : {hint}")
    return pipeline(str(tmp_wav), **kwargs)


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ══════════════════════════════════════════════════════════════

def collect_speaker_data(
    diarization_result,
    audio: AudioSegment,
) -> Tuple[Dict[str, float], Dict[str, List], Dict[str, List]]:
    """
    Walk the diarization output and collect per-speaker data.

    Returns:
        speaker_times    — {speaker: total speaking seconds}
        speaker_segments — {speaker: [(start_s, end_s), ...]}
        speaker_loudness — {speaker: [{rms, peak, rms_db, peak_db, duration}, ...]}
    """
    speaker_times:    Dict[str, float] = {}
    speaker_segments: Dict[str, List]  = {}
    speaker_loudness: Dict[str, List]  = {}

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        duration = turn.end - turn.start
        speaker_times[speaker] = speaker_times.get(speaker, 0.0) + duration
        speaker_segments.setdefault(speaker, []).append((turn.start, turn.end))

        start_ms = int(turn.start * 1000)
        end_ms   = min(int(turn.end * 1000), len(audio))

        if start_ms < len(audio) and end_ms > start_ms:
            metrics = calculate_loudness_metrics(audio[start_ms:end_ms])
            metrics["duration"] = duration
            speaker_loudness.setdefault(speaker, []).append(metrics)

    return speaker_times, speaker_segments, speaker_loudness


def compute_average_loudness(
    speaker_loudness: Dict[str, List]
) -> Dict[str, Dict[str, float]]:
    """
    Duration-weighted average loudness per speaker.

    Composite = 0.6 * avg_rms + 0.4 * avg_peak  (linear scale).
    Averaging linear RMS (not dB) is physically correct for energy.
    """
    result: Dict[str, Dict[str, float]] = {}
    for speaker, segs in speaker_loudness.items():
        durations = np.array([s["duration"] for s in segs])
        total = durations.sum()
        if total <= 0.0:
            continue
        w = durations / total

        avg_rms     = float(np.average([s["rms"]     for s in segs], weights=w))
        avg_peak    = float(np.average([s["peak"]    for s in segs], weights=w))
        avg_rms_db  = float(np.average([s["rms_db"]  for s in segs], weights=w))
        avg_peak_db = float(np.average([s["peak_db"] for s in segs], weights=w))

        result[speaker] = {
            "avg_rms":   avg_rms,
            "avg_peak":  avg_peak,
            "avg_rms_db":  avg_rms_db,
            "avg_peak_db": avg_peak_db,
            "composite": 0.6 * avg_rms + 0.4 * avg_peak,
        }
    return result


# ══════════════════════════════════════════════════════════════
#  AUDIO EXPORT
# ══════════════════════════════════════════════════════════════

def export_speaker_audio(
    speaker: str,
    speaker_segments: Dict[str, List],
    audio: AudioSegment,
    output_dir: Path,
    suffix: str = "",
) -> Optional[Path]:
    """
    Concatenate all diarized segments for one speaker and save as WAV.
    Returns the saved path, or None if no audio was found.
    """
    combined = AudioSegment.empty()
    for start_s, end_s in speaker_segments[speaker]:
        start_ms = int(start_s * 1000)
        end_ms   = min(int(end_s * 1000), len(audio))
        if start_ms < len(audio) and end_ms > start_ms:
            combined += audio[start_ms:end_ms]

    if len(combined) == 0:
        print(f"  Warning: no audio found for {speaker} — skipping.")
        return None

    out_path = output_dir / f"{speaker.lower()}{suffix}.wav"
    combined.export(str(out_path), format="wav")
    return out_path


# ══════════════════════════════════════════════════════════════
#  REPORT WRITER
# ══════════════════════════════════════════════════════════════

def write_report(
    report_path: Path,
    audio_filename: str,
    speaker_times: Dict[str, float],
    avg_loudness: Dict[str, Dict[str, float]],
    time_dominant: str,
    loudness_dominant: str,
    main_speaker: str,
    timeline: List[Dict],
) -> None:
    """Write a plain-text diarization and loudness report to disk."""
    total_time = sum(speaker_times.values())

    with open(str(report_path), "w", encoding="utf-8") as fh:
        fh.write("SPEAKER DIARIZATION REPORT\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Source file  : {audio_filename}\n")
        fh.write(f"Speakers     : {len(speaker_times)}\n")
        fh.write(f"Total speech : {total_time:.1f}s\n\n")

        fh.write("-" * 60 + "\n")
        fh.write("DOMINANT SPEAKER ANALYSIS\n")
        fh.write("-" * 60 + "\n")
        fh.write(f"  Time-based dominant    : {time_dominant}\n")
        fh.write(f"  Loudness-based dominant: {loudness_dominant}\n")
        fh.write(f"  Selected main speaker  : {main_speaker}"
                 "  (time-based — most reliable)\n\n")

        fh.write("-" * 60 + "\n")
        fh.write("SPEAKER STATISTICS\n")
        fh.write("-" * 60 + "\n")
        for spk, t in sorted(speaker_times.items()):
            pct = (t / total_time * 100) if total_time > 0 else 0.0
            fh.write(f"  {spk:<24} {t:7.1f}s   ({pct:.1f}%)\n")
        fh.write("\n")

        fh.write("-" * 60 + "\n")
        fh.write("LOUDNESS METRICS  (duration-weighted averages)\n")
        fh.write("-" * 60 + "\n")
        for spk in sorted(avg_loudness):
            m = avg_loudness[spk]
            fh.write(f"\n  {spk}:\n")
            fh.write(f"    Speaking time : {speaker_times.get(spk, 0):.1f}s\n")
            fh.write(f"    Avg RMS       : {m['avg_rms']:.4f}  ({m['avg_rms_db']:.1f} dBFS)\n")
            fh.write(f"    Avg Peak      : {m['avg_peak']:.4f}  ({m['avg_peak_db']:.1f} dBFS)\n")
            fh.write(f"    Composite     : {m['composite']:.4f}\n")
        fh.write("\n")

        fh.write("-" * 60 + "\n")
        fh.write("SEGMENT TIMELINE\n")
        fh.write("-" * 60 + "\n")
        fh.write(
            f"  {'Start(s)':>9}  {'End(s)':>9}  "
            f"{'Speaker':<22}  {'Dur(s)':>7}  {'RMS(dBFS)':>10}\n"
        )
        fh.write("  " + "-" * 62 + "\n")
        for entry in timeline:
            fh.write(
                f"  {entry['start']:9.2f}  {entry['end']:9.2f}  "
                f"{entry['speaker']:<22}  {entry['duration']:7.2f}  "
                f"{entry['rms_db']:10.1f}\n"
            )


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:

    # ── CLI arguments (override the constants at the top) ──
    parser = argparse.ArgumentParser(
        description="Separate speakers from a multi-speaker audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Command-line examples (these override the constants at the top of the script):

  python diarization.py
      Uses INPUT_AUDIO_FILE and OUTPUT_FOLDER from the top of the script.
      If both are None, auto-detects audio on Desktop and outputs there.

  python diarization.py --input audio.mp3
  python diarization.py --input audio.mp3 --output ./results
  python diarization.py --input audio.mp3 --min-speakers 2 --max-speakers 4
        """
    )
    parser.add_argument("--input",  "-i", default=None,
                        help="Path to input audio file. Overrides INPUT_AUDIO_FILE.")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to output folder. Overrides OUTPUT_FOLDER.")
    parser.add_argument("--min-speakers", type=int, default=None, metavar="N",
                        help="Minimum expected speakers. Overrides MIN_SPEAKERS.")
    parser.add_argument("--max-speakers", type=int, default=None, metavar="N",
                        help="Maximum expected speakers. Overrides MAX_SPEAKERS.")
    args = parser.parse_args()

    # CLI args take priority over the constants defined at the top
    input_file   = args.input        or INPUT_AUDIO_FILE
    output_dir_s = args.output       or OUTPUT_FOLDER
    min_spk      = args.min_speakers if args.min_speakers is not None else MIN_SPEAKERS
    max_spk      = args.max_speakers if args.max_speakers is not None else MAX_SPEAKERS

    print("\n==============================================")
    print("   Speaker Diarization")
    print("==============================================\n")

    # ── Step 1: Preflight ─────────────────────────────────
    print("[1/6] Preflight checks")
    check_ffmpeg()
    hf_token = resolve_hf_token()
    print("  ffmpeg   : OK")
    print("  HF token : OK")

    # ── Step 2: Resolve paths ─────────────────────────────
    print("\n[2/6] Resolving paths")

    if input_file:
        audio_path = Path(str(input_file)).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {audio_path}\n"
                "Check the INPUT_AUDIO_FILE path at the top of the script."
            )
        if not audio_path.is_file():
            raise ValueError(f"Input path is not a file: {audio_path}")
    else:
        # Auto-detect first audio file on Desktop
        candidates = sorted(
            f for f in DEFAULT_DESKTOP.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not candidates:
            raise FileNotFoundError(
                f"No audio file found on Desktop ({DEFAULT_DESKTOP}).\n"
                "Either:\n"
                "  1. Place an audio file on the Desktop, or\n"
                "  2. Set INPUT_AUDIO_FILE at the top of this script, or\n"
                "  3. Run:  python diarization.py --input path/to/audio.mp3"
            )
        audio_path = candidates[0]
        print(f"  Auto-detected input : {audio_path.name}")

    output_dir = Path(str(output_dir_s)).expanduser().resolve() \
                 if output_dir_s else DEFAULT_DESKTOP
    output_dir.mkdir(parents=True, exist_ok=True)  # create folder if needed

    print(f"  Input  : {audio_path}")
    print(f"  Output : {output_dir}")

    # ── Step 3: Load pipeline ─────────────────────────────
    print("\n[3/6] Loading pyannote pipeline")
    pipeline = load_pipeline(hf_token)

    # ── Step 4: Prepare audio ─────────────────────────────
    print("\n[4/6] Preparing audio")
    # tempfile is always writable; temp files are auto-deleted on exit
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio, tmp_wav = prepare_audio(audio_path, tmp_dir)

        # ── Step 5: Diarise ───────────────────────────────
        print("\n[5/6] Running diarization")
        diarization_result = run_diarization(pipeline, tmp_wav, min_spk, max_spk)
    # tmp_dir and tmp_wav are automatically deleted here

    # ── Step 6: Analyse ───────────────────────────────────
    print("\n[6/6] Analysing results")

    speaker_times, speaker_segments, speaker_loudness_raw = collect_speaker_data(
        diarization_result, audio
    )

    if not speaker_times:
        print("\nNo speakers detected.")
        print("The audio may be silent, too short, or heavily degraded.")
        sys.exit(0)

    avg_loudness = compute_average_loudness(speaker_loudness_raw)

    # Speaking time = primary and most reliable signal for main speaker
    time_dominant = max(speaker_times, key=speaker_times.get)   # type: ignore[arg-type]
    loudness_dominant = (
        max(avg_loudness, key=lambda s: avg_loudness[s]["composite"])
        if avg_loudness else time_dominant
    )
    main_speaker = time_dominant

    # ── Console output ────────────────────────────────────
    print("\n=== SEGMENT TIMELINE ===")
    print(f"  {'Start':>8}  {'End':>8}  {'Speaker':<22}  {'Dur':>6}  {'RMS(dBFS)':>10}")
    print("  " + "-" * 64)

    timeline: List[Dict] = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms   = min(int(turn.end * 1000), len(audio))
        rms_db   = -100.0
        if start_ms < len(audio) and end_ms > start_ms:
            rms_db = calculate_loudness_metrics(audio[start_ms:end_ms])["rms_db"]
        timeline.append({
            "start":    turn.start,
            "end":      turn.end,
            "speaker":  speaker,
            "duration": turn.end - turn.start,
            "rms_db":   rms_db,
        })
        print(
            f"  {turn.start:8.2f}  {turn.end:8.2f}  {speaker:<22}  "
            f"{turn.end - turn.start:6.2f}  {rms_db:10.1f}"
        )
    timeline.sort(key=lambda x: x["start"])

    total_audio_s = len(audio) / 1000.0
    print("\n=== SPEAKER STATISTICS ===")
    for speaker, t in sorted(speaker_times.items()):
        pct = (t / total_audio_s) * 100
        print(f"  {speaker:<24} {t:7.1f}s   ({pct:.1f}%)")

    if avg_loudness:
        print("\n=== LOUDNESS (duration-weighted) ===")
        for speaker, m in sorted(avg_loudness.items()):
            print(
                f"  {speaker:<24}  RMS {m['avg_rms_db']:6.1f} dBFS  |  "
                f"Peak {m['avg_peak_db']:6.1f} dBFS  |  "
                f"Composite {m['composite']:.4f}"
            )

    print("\n=== DOMINANT SPEAKER ===")
    print(f"  Time-based     : {time_dominant}")
    print(f"  Loudness-based : {loudness_dominant}")
    if time_dominant == loudness_dominant:
        print("  Both methods agree")
    else:
        print(
            f"  Methods disagree — using TIME-based ({time_dominant}) as main speaker"
        )
    print(f"\n  Main speaker : {main_speaker}")

    # ── Export audio ──────────────────────────────────────
    print(f"\n=== EXPORTING AUDIO -> {output_dir} ===")

    main_out = export_speaker_audio(
        main_speaker, speaker_segments, audio, output_dir, suffix="_MAIN"
    )
    if main_out:
        print(f"  {main_speaker} (main) -> {main_out.name}")

    for speaker in sorted(speaker_segments):
        out = export_speaker_audio(
            speaker, speaker_segments, audio, output_dir, suffix="_only"
        )
        if out:
            loud = (
                f"  (composite: {avg_loudness[speaker]['composite']:.4f})"
                if speaker in avg_loudness else ""
            )
            print(f"  {speaker} -> {out.name}{loud}")

    # ── Write report ──────────────────────────────────────
    report_path = output_dir / f"{audio_path.stem}_diarization_report.txt"
    write_report(
        report_path, audio_path.name,
        speaker_times, avg_loudness,
        time_dominant, loudness_dominant, main_speaker, timeline,
    )
    print(f"\n  Report -> {report_path.name}")

    print("\n==============================================")
    print("   DONE")
    print("==============================================")
    print(f"  Main speaker  : {main_speaker}")
    print(f"  Output folder : {output_dir}\n")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
