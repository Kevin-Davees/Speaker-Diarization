# 🎙️ Speaker Diarization

> **Who spoke when?** — Automatically identify and separate speakers from any multi-speaker audio recording.

Built on [pyannote.audio](https://github.com/pyannote/pyannote-audio), this script takes a single audio file and produces:

- Individual `.wav` files per detected speaker
- A combined "main speaker" file (the speaker with the most speaking time)
- A full timeline and loudness report (`.txt`)

---

## Table of Contents

- [What It Does](#what-it-does)
- [Requirements](#requirements)
- [Installation](#installation)
- [HuggingFace Token Setup](#huggingface-token-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Input and Output Locations](#input-and-output-locations)
- [Output Files Explained](#output-files-explained)
- [Command-Line Arguments](#command-line-arguments)
- [Common Issues and Fixes](#common-issues-and-fixes)
- [How Speaker Selection Works](#how-speaker-selection-works)
- [Supported Audio Formats](#supported-audio-formats)
- [Notes on Accuracy](#notes-on-accuracy)
- [Clearing the Model Cache](#clearing-the-model-cache)
- [License](#license)

---

## What It Does

```
Input:  meeting.mp3  (3 people talking)

Output:
  speaker_00_only.wav             ← all of Speaker 00's audio
  speaker_01_only.wav             ← all of Speaker 01's audio
  speaker_02_only.wav             ← all of Speaker 02's audio
  speaker_00_MAIN.wav             ← main speaker (most speaking time)
  meeting_diarization_report.txt  ← full timeline + loudness stats
```

---

## Requirements

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.8 or higher | Runtime |
| pyannote.audio | latest | Speaker diarization model |
| pydub | latest | Audio loading and export |
| numpy | latest | Loudness calculations |
| torch | latest | Model inference (CPU or GPU) |
| torchaudio | latest | Audio tensor utilities |
| **ffmpeg** | any recent | Audio decoding — external binary, must be on PATH |

---

## Installation

### Step 1 — Get the script

Download `diarization.py` directly, or clone this repository:

```bash
git clone https://github.com/<your-repo>/speaker-diarization.git
cd speaker-diarization
```

---

### Step 2 — Install Python packages

```bash
pip install pyannote.audio pydub numpy torch torchaudio
```

**GPU users (NVIDIA CUDA):** install the GPU-enabled version of PyTorch for 5–10× faster processing:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (e.g. `cu121` for CUDA 12.1).  
Check your CUDA version: `nvidia-smi`

---

### Step 3 — Install ffmpeg

pydub requires ffmpeg to decode formats like MP3, M4A, and AAC.

**Windows:**
1. Download the build from https://ffmpeg.org/download.html
2. Extract the zip file to a folder (e.g. `C:\ffmpeg`)
3. Add the `bin\` subfolder to your system PATH:
   - Open Start → search **Environment Variables**
   - Under **System Variables**, select `Path` → Edit → New
   - Add the full path to the `bin` folder (e.g. `C:\ffmpeg\bin`)
4. Open a new terminal and verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Verify:**
```bash
ffmpeg -version
```

---

## HuggingFace Token Setup

The pyannote model is gated and requires a free HuggingFace account.

### Step 1 — Create a HuggingFace account

Sign up for free at https://huggingface.co/join

### Step 2 — Accept the model licences

You must manually accept the usage conditions before the model can be downloaded.  
Visit both links and click **"Agree and access repository"**:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

> ⚠️ **Skipping this step is the single most common reason the script fails on first run.**

### Step 3 — Generate an access token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name it anything (e.g. `diarization-script`)
4. Select **Read** access
5. Copy the generated token — it starts with `hf_`

### Step 4 — Set the token

**Recommended — environment variable (token stays out of source code):**

| Platform | Command |
|---|---|
| Windows PowerShell | `$env:HF_TOKEN="hf_xxxxxxxxxxxx"` |
| Windows CMD | `set HF_TOKEN=hf_xxxxxxxxxxxx` |
| macOS / Linux | `export HF_TOKEN=hf_xxxxxxxxxxxx` |

To make it permanent on macOS/Linux, add the export line to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export HF_TOKEN="hf_xxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

To make it permanent on Windows, add `HF_TOKEN` as a User Environment Variable via Start → Environment Variables.

**Alternative — paste directly into the script (less secure):**

Open `diarization.py` and find the `USER CONFIGURATION` section at the top. Set:
```python
HF_TOKEN_FALLBACK = "hf_xxxxxxxxxxxx"
```

> 🔒 Do not commit your token to a public git repository.

---

## Configuration

The top of `diarization.py` contains a clearly marked **USER CONFIGURATION** section. Edit these four settings before running:

```python
# ── HuggingFace token (only needed if not using environment variable) ──
HF_TOKEN_FALLBACK = ""

# ── Input audio file ──────────────────────────────────────────────────
# Full path to your audio file. Leave as None to auto-detect on Desktop.
#
# Windows : r"C:\Users\<USERNAME>\Desktop\recording.mp3"
# macOS   : "/Users/<USERNAME>/Desktop/recording.mp3"
# Linux   : "/home/<USERNAME>/Desktop/recording.mp3"
INPUT_AUDIO_FILE = None

# ── Output folder ─────────────────────────────────────────────────────
# Where speaker WAVs and report are saved. Leave as None for Desktop.
#
# Windows : r"C:\Users\<USERNAME>\Documents\output"
# macOS   : "/Users/<USERNAME>/Documents/output"
# Linux   : "/home/<USERNAME>/Documents/output"
OUTPUT_FOLDER = None

# ── Speaker count hints (optional, improves accuracy) ────────────────
MIN_SPEAKERS = None   # e.g. 2
MAX_SPEAKERS = None   # e.g. 4
```

These constants can also be overridden at runtime using command-line arguments (see [Usage](#usage)).

---

## Usage

### Default — Desktop in, Desktop out

Place an audio file on the Desktop, then run:

```bash
python diarization.py
```

The script finds the first audio file on the Desktop, processes it, and saves all outputs there.

---

### Custom input file

```bash
python diarization.py --input "path/to/recording.mp3"
```

---

### Custom input and output folder

```bash
python diarization.py --input "recording.mp3" --output "path/to/output_folder"
```

The output folder is created automatically if it does not exist.

---

### With speaker count hints

Providing these values when you know them significantly improves accuracy:

```bash
# Exactly 2 speakers
python diarization.py --input "interview.mp3" --min-speakers 2 --max-speakers 2

# Between 2 and 5 speakers
python diarization.py --input "panel.mp3" --min-speakers 2 --max-speakers 5
```

---

### Short flags

```bash
python diarization.py -i recording.mp3 -o ./output
```

---

## Input and Output Locations

| Setting | Default | How to change |
|---|---|---|
| Input file | Auto-finds first audio file on Desktop | `INPUT_AUDIO_FILE` constant or `--input` flag |
| Output folder | Desktop | `OUTPUT_FOLDER` constant or `--output` flag |
| Temp files | System temp directory | Automatic — deleted after each run |

### Desktop path by operating system

| OS | Desktop path |
|---|---|
| Windows | `C:\Users\<USERNAME>\Desktop` |
| macOS | `/Users/<USERNAME>/Desktop` |
| Linux | `/home/<USERNAME>/Desktop` |

Replace `<USERNAME>` with your actual system username.

---

## Output Files Explained

After processing `interview.mp3` with 2 detected speakers:

```
interview_diarization_report.txt    ← full analysis report
speaker_00_MAIN.wav                 ← longest speaker's isolated audio
speaker_00_only.wav                 ← all of Speaker 00's segments
speaker_01_only.wav                 ← all of Speaker 01's segments
```

### Sample report

```
SPEAKER DIARIZATION REPORT
============================================================
Source file  : interview.mp3
Speakers     : 2
Total speech : 312.4s

DOMINANT SPEAKER ANALYSIS
  Time-based dominant    : SPEAKER_00
  Loudness-based dominant: SPEAKER_00
  Selected main speaker  : SPEAKER_00  (time-based — most reliable)

SPEAKER STATISTICS
  SPEAKER_00               210.3s   (67.3%)
  SPEAKER_01               102.1s   (32.7%)

LOUDNESS METRICS  (duration-weighted averages)
  SPEAKER_00:
    Speaking time : 210.3s
    Avg RMS       : 0.0712  (-22.9 dBFS)
    Avg Peak      : 0.3840  (-8.3 dBFS)
    Composite     : 0.1963

SEGMENT TIMELINE
   Start(s)    End(s)  Speaker                Dur(s)   RMS(dBFS)
  ...
```

---

## Command-Line Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | `INPUT_AUDIO_FILE` or auto-detect on Desktop | Path to input audio file |
| `--output` | `-o` | `OUTPUT_FOLDER` or Desktop | Path to output folder |
| `--min-speakers` | — | `MIN_SPEAKERS` | Minimum number of speakers to detect |
| `--max-speakers` | — | `MAX_SPEAKERS` | Maximum number of speakers to detect |

Command-line arguments always take priority over the constants in the script.

---

## Common Issues and Fixes

### `EnvironmentError: HuggingFace token not found`

Neither the `HF_TOKEN` environment variable nor `HF_TOKEN_FALLBACK` is set.  
See [HuggingFace Token Setup](#huggingface-token-setup).

---

### `RuntimeError: All pyannote pipeline versions failed`

**Cause 1 — Licence not accepted:**  
Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the terms.

**Cause 2 — Token is wrong or expired:**  
Generate a fresh token at https://huggingface.co/settings/tokens and update `HF_TOKEN`.

**Cause 3 — No internet on first run:**  
The model (~300 MB) downloads once on first use. Ensure you have a connection.

**Cause 4 — Outdated package:**
```bash
pip install --upgrade pyannote.audio
```

---

### `EnvironmentError: ffmpeg not found`

ffmpeg is not installed or not on PATH. See [Install ffmpeg](#step-3--install-ffmpeg).

---

### `FileNotFoundError: No audio file found on Desktop`

Set `INPUT_AUDIO_FILE` at the top of the script, or pass `--input`:
```bash
python diarization.py --input "path/to/audio.mp3"
```

---

### `FileNotFoundError: Input file not found`

The path in `INPUT_AUDIO_FILE` or `--input` does not exist.  
Double-check the path, including the file extension.

---

### `RuntimeError: Failed to load audio`

- File may be corrupted or in an unsupported codec
- Try converting to WAV first:
  ```bash
  ffmpeg -i input.mp3 -ar 16000 -ac 1 converted.wav
  python diarization.py --input converted.wav
  ```

---

### Script is slow

- First run downloads the model (~300 MB) — one-time only
- CPU is slow for long files — use a GPU if available (see [Installation](#installation))
- A 10-minute file takes approximately 2–5 minutes on CPU, under 1 minute on GPU

---

### Only one speaker detected for a multi-speaker file

- Provide `--min-speakers` and `--max-speakers` hints
- Ensure audio quality is adequate (not heavily compressed or noisy)
- Check that speakers have clearly distinct turns (heavy overlap reduces accuracy)

---

### Output WAV files are silent or empty

Try converting to 16-bit mono WAV before processing:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 converted.wav
python diarization.py --input converted.wav
```

---

### `OSError: No space left on device`

Free up disk space. The model cache uses approximately 300–500 MB.  
Cache location: `~/.cache/torch/pyannote`

---

## How Speaker Selection Works

The **main speaker** exported as `*_MAIN.wav` is the speaker with the most total speaking time.

Loudness data is computed and shown in the report as supplementary information but does **not** influence which speaker is selected as main. This is intentional:

- Loudness varies with mic placement, recording setup, and natural voice volume
- Speaking time is a more consistent and meaningful signal across different recording conditions
- A quiet speaker with 8 minutes of speech is a stronger candidate for "main" than a loud speaker with 2 minutes

---

## Supported Audio Formats

Any format supported by your ffmpeg installation:

| Format | Extension |
|---|---|
| MP3 | `.mp3` |
| WAV | `.wav` |
| Apple AAC | `.m4a` |
| FLAC | `.flac` |
| OGG Vorbis | `.ogg` |
| AAC | `.aac` |
| Windows Media Audio | `.wma` |

For best results, 16-bit mono WAV at 16 kHz is the ideal input format.  
The script performs the mono conversion automatically.

---

## Notes on Accuracy

- Works best on clean recordings with minimal background noise and distinct voices
- Challenging conditions: heavy overlap, very similar voices, loud background music
- pyannote v3.1 is optimised for conversational audio (interviews, meetings, podcasts)
- Very short utterances (under 0.5 seconds) may occasionally be misattributed
- Providing `--min-speakers` and `--max-speakers` consistently improves results

---

## Clearing the Model Cache

If you suspect a corrupted model download, open `diarization.py` and uncomment:

```python
clear_model_cache()
```

Run the script once — it will re-download the model (~300 MB) — then comment the line out again.

Alternatively, delete the cache folder manually:

| OS | Cache path |
|---|---|
| Windows | `C:\Users\<USERNAME>\.cache\torch\pyannote` |
| macOS | `~/.cache/torch/pyannote` |
| Linux | `~/.cache/torch/pyannote` |

---

## Project Structure

```
speaker-diarization/
├── diarization.py    ← main script
└── README.md         ← this file
```

---

## License

MIT License — free to use, modify, and distribute.  
Model usage is subject to the [pyannote.audio licence](https://huggingface.co/pyannote/speaker-diarization-3.1).
