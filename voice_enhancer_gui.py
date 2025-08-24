# voice_enhancer_gui.py
# GUI voice enhancer for video using customtkinter + imageio-ffmpeg
# ---------------------------------------------------------------
# Features:
# - Drag/drop style file picking (button), pretty sliders & switches
# - Auto container/codec handling: MP4(H.264+AAC) or WebM(VP8/VP9/AV1+Opus)
# - Bundled ffmpeg via imageio-ffmpeg (no PATH changes)
# - Noise reduction, high-pass, preemphasis, compression, LUFS normalize, limiter
# - "Louder" controls: target LUFS, ceiling, extra boost, stereo (dual-mono)
# - Background processing thread, progress bar, log console
# - Dark/Light theme + accent color switch

import os, sys, math, json, shlex, subprocess, threading, tempfile, traceback
from pathlib import Path
import numpy as np

# Audio/DSP
import librosa, soundfile as sf
import pyloudnorm as pyln
import noisereduce as nr
from scipy.signal import butter, filtfilt
import imageio_ffmpeg

# GUI
import customtkinter as ctk
from tkinter import filedialog, messagebox

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

# ------------------------- DSP helpers -------------------------
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
    meter = pyln.Meter(sr)  # ITU-R BS.1770
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

# ------------------------- FFmpeg helpers -------------------------
def run_ffmpeg(args: list):
    try:
        subprocess.run([FFMPEG_BIN] + args, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("FFmpeg failed:\n" + " ".join(shlex.quote(a) for a in args)) from e

def ffprobe_codec(path: str, stream_type='v'):
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
        return None
    return None

def is_mp4_compatible_video(codec: str):
    return codec in {"h264", "hevc", "h265", "mpeg4"}

def is_webm_compatible_video(codec: str):
    return codec in {"vp8", "vp9", "av1"}

def extract_audio(in_video, out_wav, sr=48000):
    run_ffmpeg(["-y", "-i", in_video, "-ac", "1", "-ar", str(sr), out_wav])

def mux_auto(in_video, in_wav, out_video, a_bitrate="160k", speed="medium", stereo=False, log=print):
    out_ext = Path(out_video).suffix.lower()
    vcodec_in = ffprobe_codec(in_video, 'v') or ""
    ac_args = ["-ac", "2"] if stereo else []

    if out_ext == ".webm":
        if is_webm_compatible_video(vcodec_in):
            log("Mux: copy video (WebM) + encode audio (Opus)")
            run_ffmpeg([
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "libopus", "-b:a", a_bitrate, *ac_args,
                "-shortest",
                out_video
            ])
        else:
            log("Transcoding video → VP9 for WebM…")
            crf = "30" if speed == "fast" else "28"
            cpu_used = "6" if speed == "fast" else "4"
            run_ffmpeg([
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", crf, "-row-mt", "1",
                "-cpu-used", cpu_used,
                "-pix_fmt", "yuv420p",
                "-c:a", "libopus", "-b:a", a_bitrate, *ac_args,
                "-shortest",
                out_video
            ])
    else:
        # default MP4
        if is_mp4_compatible_video(vcodec_in):
            log("Mux: copy video (MP4) + encode audio (AAC)")
            run_ffmpeg([
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", a_bitrate, *ac_args,
                "-shortest", "-movflags", "+faststart",
                out_video
            ])
        else:
            log("Transcoding video → H.264 for MP4…")
            crf = "20" if speed == "fast" else "18"
            preset = "faster" if speed == "fast" else "medium"
            run_ffmpeg([
                "-y", "-i", in_video, "-i", in_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", preset, "-crf", crf,
                "-c:a", "aac", "-b:a", a_bitrate, *ac_args,
                "-shortest", "-movflags", "+faststart",
                out_video
            ])

# ------------------------- GUI -------------------------
class VoiceEnhancerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Voice Enhancer")
        self.geometry("980x780")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")  # you can change to "green", "dark-blue", etc.

        self.processing_thread = None
        self.stop_flag = False

        # State vars
        self.input_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.container_choice = ctk.StringVar(value=".mp4")
        self.samplerate = ctk.StringVar(value="48000")
        self.nr_strength = ctk.DoubleVar(value=0.80)
        self.lufs = ctk.DoubleVar(value=-12.0)     # louder default
        self.ceiling = ctk.DoubleVar(value=-1.0)   # tighter ceiling
        self.boost_db = ctk.DoubleVar(value=2.0)   # extra push
        self.stereo = ctk.BooleanVar(value=True)   # dual-mono by default
        self.speed = ctk.StringVar(value="medium") # transcode quality
        self.theme = ctk.StringVar(value="Dark")
        self.accent = ctk.StringVar(value="blue")

        self._build_ui()

    # ---------- UI Layout ----------
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, corner_radius=16)
        header.pack(fill="x", padx=14, pady=(14,8))
        title = ctk.CTkLabel(header, text="🎙️ Video Voice Enhancer", font=ctk.CTkFont(size=22, weight="bold"))
        title.pack(side="left", padx=12, pady=10)

        ffm = ctk.CTkLabel(header, text=f"FFmpeg: {Path(FFMPEG_BIN).name}", font=ctk.CTkFont(size=12))
        ffm.pack(side="right", padx=12)

        # Main content
        content = ctk.CTkFrame(self, corner_radius=16)
        content.pack(fill="both", expand=True, padx=14, pady=8)

        left = ctk.CTkFrame(content, corner_radius=16)
        left.pack(side="left", fill="both", expand=True, padx=(12,8), pady=12)

        right = ctk.CTkFrame(content, corner_radius=16, width=340)
        right.pack(side="right", fill="y", padx=(8,12), pady=12)

        # Left: file selectors + logs
        file_box = ctk.CTkFrame(left, corner_radius=12)
        file_box.pack(fill="x", padx=12, pady=(12,8))

        ctk.CTkLabel(file_box, text="Input Video", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        input_entry = ctk.CTkEntry(file_box, textvariable=self.input_path, placeholder_text="Choose a video file…")
        input_entry.grid(row=1, column=0, padx=10, pady=(0,10), sticky="we")
        browse_in = ctk.CTkButton(file_box, text="Browse…", command=self.select_input)
        browse_in.grid(row=1, column=1, padx=10, pady=(0,10))
        file_box.grid_columnconfigure(0, weight=1)

        out_box = ctk.CTkFrame(left, corner_radius=12)
        out_box.pack(fill="x", padx=12, pady=(0,8))

        ctk.CTkLabel(out_box, text="Output", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        container_menu = ctk.CTkOptionMenu(out_box, values=[".mp4", ".webm"], variable=self.container_choice, command=self._auto_name_output)
        container_menu.grid(row=0, column=1, padx=10, pady=8, sticky="e")

        output_entry = ctk.CTkEntry(out_box, textvariable=self.output_path, placeholder_text="Auto: <input>_enhanced.<ext>")
        output_entry.grid(row=1, column=0, padx=10, pady=(0,10), sticky="we")
        browse_out = ctk.CTkButton(out_box, text="Save as…", command=self.select_output)
        browse_out.grid(row=1, column=1, padx=10, pady=(0,10))
        out_box.grid_columnconfigure(0, weight=1)

        # Log area
        log_box = ctk.CTkFrame(left, corner_radius=12)
        log_box.pack(fill="both", expand=True, padx=12, pady=(8,12))
        ctk.CTkLabel(log_box, text="Log", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.log_text = ctk.CTkTextbox(log_box, height=260)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.configure(state="disabled")

        # Progress + buttons
        action_box = ctk.CTkFrame(left, corner_radius=12)
        action_box.pack(fill="x", padx=12, pady=(0,12))
        self.progress = ctk.CTkProgressBar(action_box)
        self.progress.pack(fill="x", padx=12, pady=(14,8))
        self.progress.set(0.0)

        btns = ctk.CTkFrame(action_box)
        btns.pack(fill="x", padx=8, pady=(0,12))
        self.start_btn = ctk.CTkButton(btns, text="▶ Enhance", command=self.start_processing)
        self.start_btn.pack(side="left", padx=8, pady=8)
        self.cancel_btn = ctk.CTkButton(btns, text="⛔ Cancel", state="disabled", fg_color="#8b1e1e", hover_color="#a22727", command=self.cancel_processing)
        self.cancel_btn.pack(side="left", padx=8, pady=8)
        self.open_btn = ctk.CTkButton(btns, text="📂 Open Output", state="disabled", command=self.open_output_folder)
        self.open_btn.pack(side="right", padx=8, pady=8)

        # Right: controls
        tabs = ctk.CTkTabview(right, corner_radius=12)
        tabs.pack(fill="both", expand=True, padx=12, pady=12)
        t_audio = tabs.add("Audio")
        t_encode = tabs.add("Container")
        t_theme = tabs.add("Appearance")

        # Audio tab
        self._slider(t_audio, "Noise Reduction", self.nr_strength, 0.0, 1.0, step=0.01, row=0, tip="Lower if artifacts, raise if noisy")
        self._slider(t_audio, "Target Loudness (LUFS)", self.lufs, -24, -10, step=0.5, row=1, tip="−12 is louder, −14 is broadcast-ish")
        self._slider(t_audio, "True Peak Ceiling (dBFS)", self.ceiling, -3.0, -0.5, step=0.1, row=2, tip="−1.0 safe for web players")
        self._slider(t_audio, "Extra Boost (dB)", self.boost_db, 0.0, 10.0, step=0.5, row=3, tip="Applied after LUFS, into limiter")
        self._switch(t_audio, "Stereo (duplicate mono to L/R)", self.stereo, row=4)

        ctk.CTkLabel(t_audio, text="Sample Rate").grid(row=5, column=0, sticky="w", padx=12, pady=(12,0))
        ctk.CTkOptionMenu(t_audio, variable=self.samplerate, values=["48000", "44100"]).grid(row=5, column=1, padx=12, pady=(12,0), sticky="e")
        t_audio.grid_columnconfigure(0, weight=1)
        t_audio.grid_columnconfigure(1, weight=0)

        # Container tab
        ctk.CTkLabel(t_encode, text="Transcode Speed / Quality").grid(row=0, column=0, sticky="w", padx=12, pady=(12,6))
        ctk.CTkOptionMenu(t_encode, variable=self.speed, values=["medium", "fast"]).grid(row=0, column=1, padx=12, pady=(12,6), sticky="e")

        ctk.CTkLabel(t_encode, text="Container (also on left)").grid(row=1, column=0, sticky="w", padx=12, pady=6)
        ctk.CTkOptionMenu(t_encode, variable=self.container_choice, values=[".mp4", ".webm"], command=self._auto_name_output).grid(row=1, column=1, padx=12, pady=6, sticky="e")

        hint = ("• MP4 → H.264 video + AAC audio (copies video if already MP4-safe)\n"
                "• WebM → VP8/VP9/AV1 video + Opus audio (copies video if already WebM-safe)")
        ctk.CTkLabel(t_encode, text=hint, justify="left").grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(6,12))
        t_encode.grid_columnconfigure(0, weight=1)
        t_encode.grid_columnconfigure(1, weight=0)

        # Appearance tab
        ctk.CTkLabel(t_theme, text="Mode").grid(row=0, column=0, sticky="w", padx=12, pady=12)
        ctk.CTkOptionMenu(t_theme, variable=self.theme, values=["Dark", "Light"], command=self._apply_theme).grid(row=0, column=1, padx=12, pady=12, sticky="e")
        ctk.CTkLabel(t_theme, text="Accent Color").grid(row=1, column=0, sticky="w", padx=12, pady=12)
        ctk.CTkOptionMenu(t_theme, variable=self.accent, values=["blue", "green", "dark-blue"], command=self._apply_accent).grid(row=1, column=1, padx=12, pady=12, sticky="e")
        t_theme.grid_columnconfigure(0, weight=1)
        t_theme.grid_columnconfigure(1, weight=0)

    def _slider(self, parent, label, var, a, b, step, row, tip=""):
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=row, column=0, columnspan=2, sticky="we", padx=12, pady=(10,4))
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text=label, anchor="w").grid(row=0, column=0, sticky="w", padx=12, pady=(10,0))
        value_label = ctk.CTkLabel(frame, text=f"{var.get():.2f}")
        value_label.grid(row=0, column=1, sticky="e", padx=12, pady=(10,0))
        slider = ctk.CTkSlider(frame, from_=a, to=b, number_of_steps=int((b-a)/step) if step>0 else 0,
                               command=lambda v: value_label.configure(text=f"{float(v):.2f}"),
                               variable=var)
        slider.grid(row=1, column=0, columnspan=2, sticky="we", padx=12, pady=(6,12))
        if tip:
            ctk.CTkLabel(frame, text=tip, text_color=("gray40","gray70")).grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0,8))

    def _switch(self, parent, label, var, row):
        sw = ctk.CTkSwitch(parent, text=label, variable=var)
        sw.grid(row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(12,6))

    def _apply_theme(self, _=None):
        ctk.set_appearance_mode(self.theme.get())

    def _apply_accent(self, _=None):
        ctk.set_default_color_theme(self.accent.get())

    # ---------- File dialog helpers ----------
    def select_input(self):
        path = filedialog.askopenfilename(title="Select input video",
                                          filetypes=[("Video files","*.mp4 *.mov *.mkv *.webm *.m4v *.avi *.mts *.m2ts"), ("All files","*.*")])
        if path:
            self.input_path.set(path)
            self._auto_name_output()

    def select_output(self):
        ext = self.container_choice.get()
        path = filedialog.asksaveasfilename(defaultextension=ext,
                                            filetypes=[("MP4","*.mp4"),("WebM","*.webm"),("All files","*.*")])
        if path:
            self.output_path.set(path)

    def _auto_name_output(self, *_):
        ip = self.input_path.get().strip()
        if not ip:
            return
        ext = self.container_choice.get()
        stem = Path(ip).with_suffix("").name + "_enhanced"
        out = str(Path(ip).with_name(stem + ext))
        # Only auto-fill if user hasn't customized
        if not self.output_path.get() or self.output_path.get().endswith((".mp4",".webm")) and "_enhanced" in Path(self.output_path.get()).stem:
            self.output_path.set(out)

    # ---------- Logging ----------
    def log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    # ---------- Buttons ----------
    def start_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            return
        ip = self.input_path.get().strip()
        if not ip:
            messagebox.showwarning("Input required", "Please choose an input video file.")
            return
        if not os.path.isfile(ip):
            messagebox.showerror("Not found", f"Input file not found:\n{ip}")
            return
        op = self.output_path.get().strip()
        if not op:
            self._auto_name_output()
            op = self.output_path.get().strip()

        self.stop_flag = False
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.progress.set(0.0)
        self.progress.start()

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.log(f"FFmpeg: {FFMPEG_BIN}")
        self.log(f"Input:  {ip}")
        self.log(f"Output: {op}")

        # Launch background thread
        self.processing_thread = threading.Thread(target=self._process_job, args=(ip, op), daemon=True)
        self.processing_thread.start()
        self.after(250, self._poll_thread_done)

    def cancel_processing(self):
        # Soft cancel (stops after current step)
        self.stop_flag = True
        self.log("Cancel requested. Finishing current step…")

    def _poll_thread_done(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.after(300, self._poll_thread_done)
        else:
            self.progress.stop()
            self.progress.set(1.0)
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

    def open_output_folder(self):
        op = self.output_path.get().strip()
        if not op:
            return
        folder = str(Path(op).resolve().parent)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])
        except Exception as e:
            messagebox.showinfo("Open Folder", folder)

    # ---------- Worker ----------
    def _process_job(self, in_path, out_path):
        try:
            sr = int(self.samplerate.get())
            nr_strength = float(self.nr_strength.get())
            target_lufs = float(self.lufs.get())
            ceiling = float(self.ceiling.get())
            boost = float(self.boost_db.get())
            stereo = bool(self.stereo.get())
            speed = self.speed.get()

            with tempfile.TemporaryDirectory() as td:
                raw_wav = str(Path(td) / "raw.wav")
                enh_wav = str(Path(td) / "enhanced.wav")

                self.log("Extracting audio…")
                extract_audio(in_path, raw_wav, sr=sr)
                if self.stop_flag:
                    self.log("Cancelled before processing."); return

                self.log("Loading & enhancing… (this can take a bit)")
                y, sr_loaded = librosa.load(raw_wav, sr=sr, mono=True)
                y = enhance_audio(
                    y, sr_loaded,
                    nr_strength=nr_strength,
                    target_lufs=target_lufs,
                    ceiling=ceiling
                )

                if boost != 0.0:
                    self.log(f"Applying extra boost: {boost:.1f} dB")
                    g = 10.0 ** (boost / 20.0)
                    y = np.clip(y * g, -1.0, 1.0)

                if stereo:
                    y = np.stack([y, y], axis=0).T  # (n, 2)
                sf.write(enh_wav, y, sr_loaded, subtype="PCM_16")
                if self.stop_flag:
                    self.log("Cancelled before muxing."); return

                self.log("Muxing enhanced audio back to video…")
                mux_auto(in_path, enh_wav, out_path, a_bitrate="160k", speed=speed, stereo=stereo, log=self.log)

            self.log("✅ Done.")
            self.open_btn.configure(state="normal")
        except Exception as e:
            self.log("❌ Error:\n" + "".join(traceback.format_exception_only(type(e), e)).strip())
            tb = traceback.format_exc(limit=3)
            self.log(tb)

# ------------------------- Run -------------------------
if __name__ == "__main__":
    app = VoiceEnhancerApp()
    app.mainloop()
