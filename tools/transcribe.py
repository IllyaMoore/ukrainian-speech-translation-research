import argparse
from pathlib import Path
from faster_whisper import WhisperModel


def transcribe_files(input_dir="segments", output_dir="Translations",
                     model_size="large-v3", language="uk"):
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(inp.glob("*.wav"))
    if not files:
        print(f"Немає wav в {inp}")
        return

    print(f"{len(files)} файлів, модель {model_size}")
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    for i, ap in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {ap.name}")
        segments, _ = model.transcribe(
            str(ap),
            language=language,
            beam_size=5, best_of=5, temperature=0,
            condition_on_previous_text=True,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments)
        outp = out / f"{ap.stem}.txt"
        outp.write_text(text, encoding="utf-8")
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"    -> {outp.name} ({len(text)} симв.) \"{preview}\"")


def merge_transcriptions(output_dir="Translations"):
    out = Path(output_dir)
    groups = {}
    for f in sorted(out.glob("*.txt")):
        base = f.stem.rsplit("_seg", 1)[0] if "_seg" in f.stem else f.stem
        groups.setdefault(base, []).append(f)

    for base, segs in groups.items():
        if len(segs) <= 1:
            continue
        segs.sort()
        merged = "\n\n".join(s.read_text(encoding="utf-8") for s in segs)
        path = out / f"{base}_full.txt"
        path.write_text(merged, encoding="utf-8")
        print(f"  {path.name} ({len(segs)} сегм.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",  default="segments")
    p.add_argument("-o", "--output", default="Translations")
    p.add_argument("-m", "--model",  default="large-v3")
    p.add_argument("-l", "--language", default="uk")
    p.add_argument("--merge", action="store_true")
    a = p.parse_args()

    transcribe_files(a.input, a.output, a.model, a.language)
    if a.merge:
        merge_transcriptions(a.output)
