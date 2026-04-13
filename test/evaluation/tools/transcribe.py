"""
Transcribe audio files using Whisper ASR.
"""

import argparse
from pathlib import Path
from faster_whisper import WhisperModel


def transcribe_files(input_dir="segments", output_dir="Translations",
                     model_size="large-v3", language="uk"):
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(inp.glob("*.wav"))
    if not audio_files:
        print(f"Не знайдено wav файлів в {inp}")
        return

    print(f"Знайдено {len(audio_files)} файлів")
    print(f"Завантаження моделі {model_size}...")

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    print(f"Модель завантажено\n")

    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {audio_path.name}...")

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            best_of=5,
            temperature=0,
            condition_on_previous_text=True,
            vad_filter=True,
        )

        text_parts = [segment.text.strip() for segment in segments]
        full_text = " ".join(text_parts)

        output_path = out / f"{audio_path.stem}.txt"
        output_path.write_text(full_text, encoding="utf-8")

        preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
        print(f"    -> {output_path.name} ({len(full_text)} symbols)")
        print(f"    \"{preview}\"\n")

    print(f"Готово! Транскрипції збережено в ./{out}/")


def merge_transcriptions(output_dir="Translations"):
    out = Path(output_dir)
    files = sorted(out.glob("*.txt"))
    groups = {}

    for f in files:
        name = f.stem
        base = name.rsplit("_seg", 1)[0] if "_seg" in name else name
        if base not in groups:
            groups[base] = []
        groups[base].append(f)

    print(f"\nОб'єднання сегментів...")

    for base, segment_files in groups.items():
        if len(segment_files) <= 1:
            continue

        segment_files.sort()
        texts = [sf.read_text(encoding="utf-8") for sf in segment_files]
        merged = "\n\n".join(texts)

        merged_path = out / f"{base}_full.txt"
        merged_path.write_text(merged, encoding="utf-8")
        print(f"  {merged_path.name} ({len(segment_files)} сегментів)")

    print("Готово!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Транскрипція аудіо через Whisper")
    parser.add_argument("-i", "--input", default="segments")
    parser.add_argument("-o", "--output", default="Translations")
    parser.add_argument("-m", "--model", default="large-v3")
    parser.add_argument("-l", "--language", default="uk")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    transcribe_files(args.input, args.output, args.model, args.language)

    if args.merge:
        merge_transcriptions(args.output)
