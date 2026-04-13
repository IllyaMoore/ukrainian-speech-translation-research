"""
Split audio into 3-5 minute segments at natural pauses.
"""

import soundfile as sf
import numpy as np
from pathlib import Path
import argparse
import subprocess
import tempfile
import os

NATIVE_FORMATS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def convert_to_wav(input_path, output_path):
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("imageio-ffmpeg не встановлено. pip install imageio-ffmpeg")

    cmd = [
        ffmpeg, '-i', str(input_path),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', '-y',
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg помилка: {result.stderr}")


def detect_silence(samples, sample_rate, min_silence_len_ms=500, silence_thresh_db=-40):
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)

    window_size = int(sample_rate * 0.01)
    n_windows = len(samples) // window_size

    rms = np.array([
        np.sqrt(np.mean(samples[i*window_size:(i+1)*window_size]**2))
        for i in range(n_windows)
    ])

    thresh_linear = 10 ** (silence_thresh_db / 20)
    is_silent = rms < thresh_linear

    silences = []
    in_silence = False
    start = 0
    min_windows = int(min_silence_len_ms / 10)

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start = i
            in_silence = True
        elif not silent and in_silence:
            if i - start >= min_windows:
                silences.append((start * 10, i * 10))
            in_silence = False

    if in_silence and n_windows - start >= min_windows:
        silences.append((start * 10, n_windows * 10))

    return silences


def find_split_points(silences, total_ms, min_dur_ms=180_000, max_dur_ms=300_000):
    pause_midpoints = [(start + end) // 2 for start, end in silences]

    split_points = []
    last_cut = 0

    for mid in pause_midpoints:
        if mid - last_cut < min_dur_ms:
            continue
        split_points.append(mid)
        last_cut = mid

    if split_points and (total_ms - split_points[-1]) < 60_000:
        split_points.pop()

    return split_points


def split_audio(input_path, output_dir="segments", min_min=3, max_min=5,
                silence_thresh=-40, min_silence_len=500):
    p = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    temp_wav = None
    if p.suffix.lower() not in NATIVE_FORMATS:
        print(f"Конвертація {p.name} в wav...")
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        convert_to_wav(p, temp_wav.name)
        audio_path = temp_wav.name
    else:
        audio_path = str(p)

    print(f"Завантаження {p.name}...")
    samples, sample_rate = sf.read(audio_path)
    total_ms = int(len(samples) / sample_rate * 1000)
    print(f"Тривалість: {total_ms / 1000 / 60:.1f} хв")

    min_ms = min_min * 60 * 1000
    max_ms = max_min * 60 * 1000

    print("Пошук пауз...")
    silences = detect_silence(samples, sample_rate, min_silence_len, silence_thresh)
    print(f"Знайдено {len(silences)} пауз")

    points = find_split_points(silences, total_ms, min_ms, max_ms)

    boundaries = [0] + points + [total_ms]
    n = len(boundaries) - 1
    print(f"Буде {n} фрагментів\n")

    stem = p.stem

    for i in range(n):
        start_ms = boundaries[i]
        end_ms = boundaries[i + 1]

        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = int(end_ms * sample_rate / 1000)

        segment = samples[start_sample:end_sample]
        dur = (end_ms - start_ms) / 1000

        fname = f"{stem}_seg{i+1:03d}.wav"
        fpath = out / fname
        sf.write(str(fpath), segment, sample_rate)

        s_min, s_sec = divmod(start_ms // 1000, 60)
        e_min, e_sec = divmod(end_ms // 1000, 60)
        print(f"  [{i+1}/{n}] {fname}  "
              f"{s_min:02d}:{s_sec:02d} -> {e_min:02d}:{e_sec:02d}  "
              f"({dur:.1f}с)")

    if temp_wav:
        os.unlink(temp_wav.name)

    print(f"\nЗбережено в ./{out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Розрізка аудіо на фрагменти 3-5 хв по паузах")
    parser.add_argument("input", help="Шлях до аудіофайлу")
    parser.add_argument("-o", "--output", default="segments")
    parser.add_argument("--min", type=int, default=3)
    parser.add_argument("--max", type=int, default=5)
    parser.add_argument("--thresh", type=int, default=-40)
    parser.add_argument("--pause", type=int, default=500)
    args = parser.parse_args()

    split_audio(args.input, args.output, args.min, args.max, args.thresh, args.pause)
