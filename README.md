# 🎙️ Video Voice Enhancer (GUI) — CustomTkinter + FFmpeg

Beautiful, simple desktop app to **boost dialogue clarity and loudness** in any video.  
It bundles **FFmpeg** via `imageio-ffmpeg` (no PATH needed) and gives you studio-style controls: noise reduction, high‑pass, pre‑emphasis, compression, **LUFS normalization**, and a true‑peak limiter. Choose **MP4** or **WebM** output; the app automatically reuses the original video when possible to save time.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10–3.12-blue">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey">
  <img src="https://img.shields.io/badge/GUI-CustomTkinter-4c9">
  <img src="https://img.shields.io/badge/License-Your%20Choice-informational">
</p>

---

## ✨ Features

- **One‑click enhancement** with progress bar & live log
- **Noise reduction** (speech‑friendly) with adjustable strength
- **High‑pass + pre‑emphasis** for clarity and rumble control
- **Compressor** (threshold/ratio/makeup/attack/release tuned for voice)
- **Integrated loudness** normalization to **target LUFS** (e.g., −12, −14, −16)
- **True‑peak limiter** with configurable ceiling (e.g., −1.0 dBFS)
- **“Extra Boost”** after LUFS if you still want it louder
- **Stereo (dual‑mono)** toggle for a centered voice in L/R
- **Container‑aware muxing**:
  - **MP4** (H.264 + AAC) or **WebM** (VP9/AV1 + Opus)
  - **Copies the original video stream** when compatible (no re‑encode)
  - Otherwise **transcodes** with sane defaults
- **Bundled FFmpeg** — thanks to `imageio-ffmpeg`, no system install required
- **Dark/Light themes + accent color** (blue/green/dark‑blue)

> The GUI file is **`voice_enhancer_gui.py`**. Drop it in a virtualenv and run!

---

## 🖼️ Screenshots / Demo

Add your own screenshots here:

```
docs/
  ├─ screenshot_dark.png
  └─ demo.mp4
```

Example Markdown you can use in your README once you add files:

```md
![App screenshot]
```
<img width="991" height="821" alt="image" src="https://github.com/user-attachments/assets/e1920a5b-4c2d-4e7a-a015-880c5f29a466" />
---

## 🔧 Installation

> **Python 3.10–3.12 recommended.** Some audio/DSP wheels may lag behind the newest Python releases.

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Upgrade packaging tools
python -m pip install -U pip setuptools wheel

# 3) Install dependencies
pip install customtkinter numpy scipy librosa soundfile pyloudnorm noisereduce imageio-ffmpeg
```

That’s it. `imageio-ffmpeg` will **download a local FFmpeg binary** automatically on first use.

---

## ▶️ Usage

```bash
python voice_enhancer_gui.py
```

1. **Input Video** → Pick an `.mp4`, `.mov`, `.mkv`, `.webm`, etc.  
2. **Output** → Choose **.mp4** or **.webm** (auto‑named `*_enhanced.ext`).  
3. **Audio Tab** → Tune **Noise Reduction**, **Target Loudness (LUFS)**, **Ceiling**, **Extra Boost**, and **Stereo**.  
4. **Container Tab** → Pick **transcode speed/quality** and container.  
5. **Appearance** → Dark/Light mode and accent.  
6. Press **Enhance** ▶ and watch the **Log**. When finished, click **Open Output**.

---

## 🎚️ What the app does (signal chain)

The audio pipeline is designed for **spoken voice**:

1. **DC removal + High‑Pass** (default 80 Hz) — removes rumble/handling noise.  
2. **Noise Reduction** (non‑stationary, speech‑safe) — stronger values reduce more noise but can add artifacts.  
3. **Pre‑emphasis** (0.95) — gently brightens speech.  
4. **Compression** — tames peaks and raises quieter syllables.  
5. **LUFS Normalization** — sets overall loudness (e.g., **−12 LUFS** ≈ louder YouTube‑style speech, **−14** typical streaming, **−16** more conservative).  
6. **True‑Peak Limiter** — caps the absolute peak at the selected **Ceiling** (e.g., **−1.0 dBFS**).  
7. **Extra Boost (dB)** — optional final gain after LUFS, then the limiter catches any overs.

**Container‑aware muxing:**  
- If your input video codec is **MP4‑safe** (H.264/H.265/MPEG‑4), we **copy the video stream** and only re‑encode the audio to **AAC**.  
- If your output is **WebM** and the input isn’t VP8/VP9/AV1, we **transcode** video to **VP9** (quality driven by the **Speed** you choose) and audio to **Opus**.  
- Audio bitrate defaults to **160 kb/s** (tweakable in code).

---

## 🛠️ Controls Explained

| Control | Why it matters | Tips |
|---|---|---|
| **Noise Reduction** | Reduces background hiss/fan/AC noise | If you hear artifacts or “bubbling,” lower it |
| **Target Loudness (LUFS)** | Overall speech loudness | −12 = louder, −14 = typical web, −16 = conservative |
| **True Peak Ceiling (dBFS)** | Peak safety for players | −1.0 is a safe cross‑platform choice |
| **Extra Boost (dB)** | Final push after LUFS | Use sparingly; limiter will catch overs |
| **Stereo (dual‑mono)** | Duplicates mono voice to L/R | Keeps dialog centered in both channels |
| **Sample Rate** | Processing sample rate | 48 kHz is standard for video |
| **Speed (Container tab)** | Transcode quality/speed | “medium” = better quality, “fast” = quicker |

---

## 📦 Project Structure

```
.
├─ voice_enhancer_gui.py   # The GUI app (run this)
├─ README.md               # You are here
└─ docs/                   # (optional) screenshots, demo video, etc.
```

---

## 🧪 Programmatic usage (optional)

You can reuse the DSP core in your own scripts. For example:

```python
from voice_enhancer_gui import enhance_audio, mux_auto, extract_audio
import librosa, soundfile as sf

# 1) Extract & load mono audio at 48 kHz
extract_audio("input.mp4", "raw.wav", sr=48000)
y, sr = librosa.load("raw.wav", sr=48000, mono=True)

# 2) Enhance
y = enhance_audio(y, sr, target_lufs=-14.0, ceiling=-1.0)

# 3) Save & mux back
sf.write("enhanced.wav", y, sr, subtype="PCM_16")
mux_auto("input.mp4", "enhanced.wav", "output_enhanced.mp4", stereo=True)
```

---

## ❓ FAQ / Troubleshooting

- **“FFmpeg failed …”**  
  Ensure the input path exists and isn’t DRM‑protected. The log will show the exact ffmpeg args used.  
  `imageio-ffmpeg` provides an FFmpeg binary automatically—no PATH setup required.

- **“ModuleNotFoundError: …”**  
  Activate your virtualenv and re‑install requirements. Use Python 3.10–3.12.

- **Output is still too quiet/too loud**  
  Adjust **Target LUFS** (e.g., −12 is louder than −16). Use **Extra Boost** carefully and keep **Ceiling** around −1.0 dBFS.

- **Artifacts after noise reduction**  
  Lower the **Noise Reduction** slider until artifacts are acceptable.

- **Slow processing**  
  If the app has to transcode video, it will be slower. If possible, choose an output container that allows **copying** the original video stream.

---

## 🗺️ Roadmap (ideas)

- Batch mode (process multiple files)
- Adjustable high‑pass / compressor controls in the GUI
- Voice activity detection & auto‑ducking music
- Optional speech‑to‑text for subtitle drafts (Whisper)
- Presets for YouTube/Podcast/TikTok

Open an issue or PR if you want any of these!

---

## 🧾 Changelog

- **August 23, 2025 — v1.0**: First public release (GUI, LUFS normalize, limiter, noise reduction, container‑aware muxing).

---

## 🙏 Credits

- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [librosa](https://librosa.org/)
- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)
- [noisereduce](https://github.com/timsainb/noisereduce)
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)
- [FFmpeg](https://ffmpeg.org/)

---


## 📜 License

Choose a license that fits your needs (e.g., MIT, Apache‑2.0). If you’re unsure, MIT is a good default.
