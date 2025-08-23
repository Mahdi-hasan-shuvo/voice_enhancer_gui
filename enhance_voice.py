import argparse, os, subprocess, tempfile, math, sys, json, shlex
from pathlib import Path
import numpy as np
import librosa, soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, filtfilt
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

# ---------- FFmpeg helpers ----------
def run_ffmpeg(args: list):
    try:
        subprocess.run([FFMPEG_BIN] + args, check=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed with args:\n", " ".join(shlex.quote(a) for a in args))
        raise

def ffprobe_codec(path: str, stream_type='v'):
    """Return codec_name for v:0 or a:0 using ffmpeg JSON output."""
    cmd = [
        FFMPEG_BIN, "-hide_banner", "-v", "error",
        "-select_streams", f"{stream_type}:0",
        "-show_entries", "stream=codec_name",
        "-of", "json", "-i", path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if streams and "codec_name" in streams[0]:
            return streams[0]["codec_name"]
    except Exception:
        pass
    return None

def is_mp4_compatible_video(codec: str):
    return codec in {"h264", "hevc", "h265", "mpeg4"}

def is_webm_compatible_video(codec: str):
    return codec in {"vp8", "vp9", "av1"}

# ---------- DSP chain ----------
def highpass(y, sr, cutoff=80.0, order=2):
    nyq = 0.5 * sr
    b, a = butter(order, cutoff/nyq, btype="highpass")
    return filtfilt(b, a, y)

def preemphasis(y, coef=0.95):
    return np.append(y[0], y[1:] - coef * y[:-1])

def compressor(y, sr, threshold_db=-22.0, ratio=3.0, makeup_db=4.0,
               attack_ms=8.0, release_ms=120.0):
    atk = math.exp(-1.0 / (sr * (attack_ms/1000.0)))
    rel = math.exp(-1.0 / (sr * (release_ms/1000.0)))
    env = 0.0
    out = np.zeros_like(y)
    for i, x in enumerate(y):
        level = abs(x)
        env = (atk * env + (1 - atk) * level) if level > env else (rel * env + (1 - rel) * level)
        x_db = 20.0 * math.log10(max(env, 1e-9))
        if x_db > threshold_db:
            y_db = threshold_db + (x_db - threshold_db) / ratio
            gain_db = (y_db - x_db) + makeup_db
        else:
            gain_db = makeup_db
        g = 10.0 ** (gain_db / 20.0)
        out[i] = x * g
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out

def lufs_normalize(y, sr, target=-16.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y.astype(np.float64))
    y_norm = pyln.normalize.loudness(y.astype(np.float64), loudness, target)
    return y_norm.astype(np.float32)

def peak_limit(y, ceiling_dbfs=-1.5):
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak = np.max(np.abs(y))
    if peak > ceiling and peak > 0:
        y = y * (ceiling / peak)
    return y

def enhance_audio(y, sr,
                  hp_cut=80.0,
                  nr_strength=0.8,
                  preemp=0.95,
                  thresh=-22.0, ratio=3.0, makeup=4.0,
                  atk=8.0, rel=120.0,
                  target_lufs=-16.0,
                  ceiling=-1.5):
    y = y - np.mean(y)                       # DC
    y = highpass(y, sr, cutoff=hp_cut)       # Rumble cut
    y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=nr_strength,
                        time_constant_s=0.4, freq_mask_smooth_hz=500, n_jobs=1)
    y = preemphasis(y, coef=preemp)          # Clarity
    y = compressor(y, sr, threshold_db=thresh, ratio=ratio, makeup_db=makeup,
                   attack_ms=atk, release_ms=rel)
    y = lufs_normalize(y, sr, target=target_lufs)
    y = peak_limit(y, ceiling_dbfs=ceiling)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

# ---------- I/O ----------
def extract_audio(in_video, out_wav, sr=48000):
    run_ffmpeg([
        "-y", "-i", in_video,
        "-ac", "1", "-ar", str(sr),
        out_wav
    ])

def mux_auto(in_video, in_wav, out_video, a_bitrate="160k", speed="medium", stereo=False, dry_run=False):
    """
    Choose a safe container/codec combo automatically:
      - .mp4 -> H.264 video (copy if already MP4-compatible), AAC audio
      - .webm -> VP8/VP9/AV1 video (copy if compatible), Opus audio
      - stereo=True will set -ac 2 for encoded audio
    """
    out_ext = Path(out_video).suffix.lower()
    vcodec_in = ffprobe_codec(in_video, 'v') or ""
    ac_args = ["-ac", "2"] if stereo else []

    if dry_run:
        print(f"[dry-run] input video codec: {vcodec_in}")
        print(f"[dry-run] output: {out_video}")

    if out_ext == ".webm":
        # WebM mandates VP8/VP9/AV1 video + Opus/Vorbis audio
        if is_webm_compatible_video(vcodec_in):
            # copy video, encode audio to Opus
            args = [
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "libopus", "-b:a", a_bitrate, *ac_args,
                "-shortest",
                out_video
            ]
            if dry_run:
                print("[dry-run] action: copy video → webm, encode audio → Opus")
            else:
                run_ffmpeg(args)
        else:
            # transcode video to VP9
            crf = "30" if speed == "fast" else "28"
            cpu_used = "6" if speed == "fast" else "4"
            args = [
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", crf, "-row-mt", "1",
                "-cpu-used", cpu_used,
                "-pix_fmt", "yuv420p",
                "-c:a", "libopus", "-b:a", a_bitrate, *ac_args,
                "-shortest",
                out_video
            ]
            if dry_run:
                print("[dry-run] action: transcode video → VP9, encode audio → Opus")
            else:
                run_ffmpeg(args)
    else:
        # default to MP4
        if is_mp4_compatible_video(vcodec_in):
            # copy video, encode audio to AAC
            args = [
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", a_bitrate, *ac_args,
                "-shortest", "-movflags", "+faststart",
                out_video
            ]
            if dry_run:
                print("[dry-run] action: copy video → mp4, encode audio → AAC")
            else:
                run_ffmpeg(args)
        else:
            # transcode video to H.264
            crf = "20" if speed == "fast" else "18"
            preset = "faster" if speed == "fast" else "medium"
            args = [
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", preset, "-crf", crf,
                "-c:a", "aac", "-b:a", a_bitrate, *ac_args,
                "-shortest", "-movflags", "+faststart",
                out_video
            ]
            if dry_run:
                print("[dry-run] action: transcode video → H.264, encode audio → AAC")
            else:
                print("Video codec incompatible with MP4. Transcoding to H.264…")
                run_ffmpeg(args)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Enhance voice in a video and remux with proper container/codec.")
    ap.add_argument("input_video", help="Path to input video")
    ap.add_argument("-o", "--output", help="Output path (.mp4 or .webm). Defaults to *_enhanced.mp4")
    ap.add_argument("--sr", type=int, default=48000, help="Audio sample rate (default 48000)")
    ap.add_argument("--nr", type=float, default=0.8, help="Noise reduction strength (0-1)")
    ap.add_argument("--lufs", type=float, default=-16.0, help="Target loudness in LUFS (e.g., -12 is louder)")
    ap.add_argument("--ceiling", type=float, default=-1.5, help="Peak ceiling dBFS (e.g., -1.0 is louder)")
    ap.add_argument("--boost_db", type=float, default=0.0, help="Extra post-normalize gain in dB (0–6)")
    ap.add_argument("--stereo", action="store_true", help="Duplicate mono to stereo on output")
    ap.add_argument("--speed", choices=["fast","medium"], default="medium",
                    help="Transcode speed/quality for VP9/H.264 when needed")
    ap.add_argument("--dry-run", action="store_true", help="Print what will happen and exit")
    ap.add_argument("--show-ffmpeg", action="store_true", help="Print bundled ffmpeg path and exit")
    args = ap.parse_args()

    if args.show_ffmpeg:
        print(FFMPEG_BIN)
        sys.exit(0)

    in_path = args.input_video
    out_path = args.output or str(Path(in_path).with_name(Path(in_path).stem + "_enhanced.mp4"))

    if args.dry_run:
        vcodec = ffprobe_codec(in_path, 'v')
        print(f"[dry-run] ffmpeg: {FFMPEG_BIN}")
        print(f"[dry-run] input: {in_path}")
        print(f"[dry-run] detected video codec: {vcodec}")
        print(f"[dry-run] output: {out_path}")

    with tempfile.TemporaryDirectory() as td:
        raw_wav = str(Path(td) / "raw.wav")
        enh_wav = str(Path(td) / "enhanced.wav")

        if not args.dry_run:
            print("Extracting audio…")
            extract_audio(in_path, raw_wav, sr=args.sr)

            print("Loading & enhancing…")
            y, sr = librosa.load(raw_wav, sr=args.sr, mono=True)
            y = enhance_audio(
                y, sr,
                nr_strength=args.nr,
                target_lufs=args.lufs,
                ceiling=args.ceiling
            )

            # Optional extra loudness push after normalize (into limiter)
            if args.boost_db != 0.0:
                g = 10.0 ** (args.boost_db / 20.0)
                y = np.clip(y * g, -1.0, 1.0)

            # Write mono or dual-mono stereo
            if args.stereo:
                y_st = np.stack([y, y], axis=0)  # (2, n)
                sf.write(enh_wav, y_st.T, sr, subtype="PCM_16")
            else:
                sf.write(enh_wav, y, sr, subtype="PCM_16")

        print("Muxing enhanced audio back to video…")
        mux_auto(in_path, enh_wav, out_path, a_bitrate="160k", speed=args.speed, stereo=args.stereo, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"✅ Done: {out_path}")
    else:
        print("✅ Dry-run complete.")

if __name__ == "__main__":
    main()
