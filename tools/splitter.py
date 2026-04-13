import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

NATIVE = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def to_wav(src, dst):
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError("pip install imageio-ffmpeg")
    r = subprocess.run(
        [ff, '-i', str(src), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '44100', '-ac', '1', '-y', str(dst)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr}")


def detect_silence(samples, sr, min_silence_ms=500, thresh_db=-40):
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    win = int(sr * 0.01)
    n = len(samples) // win
    rms = np.array([
        np.sqrt(np.mean(samples[i*win:(i+1)*win]**2)) for i in range(n)
    ])
    silent = rms < 10 ** (thresh_db / 20)
    min_w = min_silence_ms // 10

    silences, in_sil, start = [], False, 0
    for i, s in enumerate(silent):
        if s and not in_sil:
            start, in_sil = i, True
        elif not s and in_sil:
            if i - start >= min_w:
                silences.append((start * 10, i * 10))
            in_sil = False
    if in_sil and n - start >= min_w:
        silences.append((start * 10, n * 10))
    return silences


def split_points(silences, total_ms, min_dur_ms, max_dur_ms):
    mids = [(s + e) // 2 for s, e in silences]
    points, last = [], 0
    for m in mids:
        if m - last < min_dur_ms:
            continue
        points.append(m)
        last = m
    # не залишаємо хвостовий шматок коротший за хвилину
    if points and (total_ms - points[-1]) < 60_000:
        points.pop()
    return points


def split_audio(input_path, output_dir="segments",
                min_min=3, max_min=5,
                silence_thresh=-40, min_silence_len=500):
    p = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tmp = None
    if p.suffix.lower() not in NATIVE:
        print(f"Конвертація {p.name}...")
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        to_wav(p, tmp.name)
        path = tmp.name
    else:
        path = str(p)

    samples, sr = sf.read(path)
    total_ms = int(len(samples) / sr * 1000)
    print(f"{p.name}: {total_ms/1000/60:.1f} хв")

    silences = detect_silence(samples, sr, min_silence_len, silence_thresh)
    print(f"{len(silences)} пауз")

    points = split_points(silences, total_ms, min_min*60_000, max_min*60_000)
    bounds = [0] + points + [total_ms]
    n = len(bounds) - 1
    print(f"{n} фрагментів\n")

    stem = p.stem
    for i in range(n):
        s_ms, e_ms = bounds[i], bounds[i+1]
        s_smp = int(s_ms * sr / 1000)
        e_smp = int(e_ms * sr / 1000)
        seg = samples[s_smp:e_smp]
        fname = f"{stem}_seg{i+1:03d}.wav"
        sf.write(str(out / fname), seg, sr)
        sm, ss = divmod(s_ms // 1000, 60)
        em, es = divmod(e_ms // 1000, 60)
        print(f"  [{i+1}/{n}] {fname}  "
              f"{sm:02d}:{ss:02d} -> {em:02d}:{es:02d}  ({(e_ms-s_ms)/1000:.1f}с)")

    if tmp:
        os.unlink(tmp.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Розрізка аудіо по паузах")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="segments")
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--thresh", type=int, default=-40)
    ap.add_argument("--pause", type=int, default=500)
    a = ap.parse_args()
    split_audio(a.input, a.output, a.min, a.max, a.thresh, a.pause)
