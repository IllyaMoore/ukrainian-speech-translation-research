"""
ASR + MT pipeline: Whisper (UK) -> OPUS-MT (UK -> EN).
Outputs JSON with texts, timestamps, and latency metrics.
"""


import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer



@dataclass
class Segment:
    id: int
    start: float
    end: float
    text_uk: str
    text_en: str


@dataclass
class ProcessingResult:
    audio_file: str
    duration_sec: float
    segments: list
    full_text_uk: str
    full_text_en: str
    asr_time_sec: float
    mt_time_sec: float
    total_time_sec: float
    whisper_model: str
    mt_model: str
    language: str


class Pipeline:

    def __init__(
        self,
        whisper_model: str = "large-v3",
        mt_model: str = "Helsinki-NLP/opus-mt-uk-en",
        device: str = "auto"
    ):
        self.whisper_model_name = whisper_model
        self.mt_model_name = mt_model

        print(f"Завантаження Whisper {whisper_model}...")
        self.asr = WhisperModel(whisper_model, device=device, compute_type="auto")

        print(f"Завантаження MT {mt_model}...")
        self.tokenizer = MarianTokenizer.from_pretrained(mt_model)
        self.mt = MarianMTModel.from_pretrained(mt_model)

        import torch
        if torch.cuda.is_available():
            self.mt = self.mt.cuda()
            self.mt_device = "cuda"
        else:
            self.mt_device = "cpu"

        print("Моделі завантажено\n")

    def translate(self, text: str, max_length: int = 512) -> str:
        if not text.strip():
            return ""

        sentences = self._split_text(text, max_length=400)
        translated_parts = []

        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            if self.mt_device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self.mt.generate(**inputs, max_length=max_length)
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_parts.append(translated)

        return " ".join(translated_parts)

    def _split_text(self, text: str, max_length: int = 400) -> list:
        if len(text) <= max_length:
            return [text]

        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        result = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) < max_length:
                current += " " + sentence if current else sentence
            else:
                if current:
                    result.append(current.strip())
                current = sentence

        if current:
            result.append(current.strip())

        return result if result else [text[:max_length]]

    def process_audio(self, audio_path: str, language: str = "uk") -> ProcessingResult:
        audio_path = Path(audio_path)
        print(f"Обробка: {audio_path.name}")

        t0 = time.perf_counter()

        asr_segments, info = self.asr.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            best_of=5,
            temperature=0,
            condition_on_previous_text=True,
            vad_filter=True,
            word_timestamps=False,
        )

        segments_list = []
        for seg in asr_segments:
            segments_list.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })

        asr_time = time.perf_counter() - t0
        print(f"  ASR: {asr_time:.2f}s ({len(segments_list)} сегментів)")

        full_text_uk = " ".join([s["text"] for s in segments_list])

        t0 = time.perf_counter()

        result_segments = []
        for seg in segments_list:
            text_en = self.translate(seg["text"])
            result_segments.append(Segment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text_uk=seg["text"],
                text_en=text_en
            ))

        mt_time = time.perf_counter() - t0
        print(f"  MT:  {mt_time:.2f}s")

        full_text_en = " ".join([s.text_en for s in result_segments])

        total_time = asr_time + mt_time
        duration = info.duration if hasattr(info, 'duration') else 0

        print(f"  Total: {total_time:.2f}s (audio: {duration:.1f}s, RTF: {total_time/duration:.2f}x)\n")

        return ProcessingResult(
            audio_file=audio_path.name,
            duration_sec=duration,
            segments=[asdict(s) for s in result_segments],
            full_text_uk=full_text_uk,
            full_text_en=full_text_en,
            asr_time_sec=asr_time,
            mt_time_sec=mt_time,
            total_time_sec=total_time,
            whisper_model=self.whisper_model_name,
            mt_model=self.mt_model_name,
            language=language
        )


def process_corpus(
    input_dir: str = "segments",
    output_dir: str = "results",
    whisper_model: str = "large-v3",
    mt_model: str = "Helsinki-NLP/opus-mt-uk-en",
    language: str = "uk"
):
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "asr").mkdir(exist_ok=True)
    (out / "mt").mkdir(exist_ok=True)

    audio_files = sorted(inp.glob("*.wav"))
    if not audio_files:
        print(f"Не знайдено wav файлів в {inp}")
        return

    print(f"Знайдено {len(audio_files)} файлів\n")

    pipeline = Pipeline(whisper_model, mt_model)

    all_results = []
    total_asr = 0
    total_mt = 0
    total_duration = 0

    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}]", end=" ")

        result = pipeline.process_audio(str(audio_path), language)
        all_results.append(asdict(result))

        total_asr += result.asr_time_sec
        total_mt += result.mt_time_sec
        total_duration += result.duration_sec

        stem = audio_path.stem
        (out / "asr" / f"{stem}.txt").write_text(result.full_text_uk, encoding="utf-8")
        (out / "mt" / f"{stem}.txt").write_text(result.full_text_en, encoding="utf-8")
        (out / f"{stem}.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    summary = {
        "total_files": len(audio_files),
        "total_audio_duration_sec": total_duration,
        "total_asr_time_sec": total_asr,
        "total_mt_time_sec": total_mt,
        "total_processing_time_sec": total_asr + total_mt,
        "avg_asr_rtf": total_asr / total_duration if total_duration > 0 else 0,
        "avg_mt_time_per_file_sec": total_mt / len(audio_files),
        "overall_rtf": (total_asr + total_mt) / total_duration if total_duration > 0 else 0,
        "whisper_model": whisper_model,
        "mt_model": mt_model,
        "files": [r["audio_file"] for r in all_results]
    }

    (out / "corpus_results.json").write_text(
        json.dumps({"summary": summary, "results": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("=" * 50)
    print(f"ПІДСУМОК:")
    print(f"  Файлів: {len(audio_files)}")
    print(f"  Загальна тривалість аудіо: {total_duration/60:.1f} хв")
    print(f"  ASR час: {total_asr:.1f}s (RTF: {total_asr/total_duration:.2f}x)")
    print(f"  MT час:  {total_mt:.1f}s")
    print(f"  Загальний RTF: {(total_asr + total_mt)/total_duration:.2f}x")
    print(f"\nРезультати збережено в ./{out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASR + MT pipeline з часовими мітками та латентністю")
    parser.add_argument("-i", "--input", default="segments",
                        help="Папка з аудіо (за замовч. segments)")
    parser.add_argument("-o", "--output", default="results",
                        help="Папка для результатів (за замовч. results)")
    parser.add_argument("-m", "--model", default="large-v3",
                        help="Whisper модель (за замовч. large-v3)")
    parser.add_argument("--mt", default="Helsinki-NLP/opus-mt-uk-en",
                        help="MT модель (за замовч. Helsinki-NLP/opus-mt-uk-en)")
    parser.add_argument("-l", "--language", default="uk",
                        help="Мова аудіо (за замовч. uk)")
    args = parser.parse_args()

    process_corpus(args.input, args.output, args.model, args.mt, args.language)
