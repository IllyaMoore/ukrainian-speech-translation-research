import argparse
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
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
    def __init__(self, whisper_model="large-v3",
                 mt_model="Helsinki-NLP/opus-mt-uk-en", device="auto"):
        self.whisper_model_name = whisper_model
        self.mt_model_name = mt_model

        print(f"Whisper {whisper_model}...")
        self.asr = WhisperModel(whisper_model, device=device, compute_type="auto")

        print(f"MT {mt_model}...")
        self.tokenizer = MarianTokenizer.from_pretrained(mt_model)
        self.mt = MarianMTModel.from_pretrained(mt_model)

        self.mt_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.mt_device == "cuda":
            self.mt = self.mt.cuda()
        print()

    def _split(self, text, max_len=400):
        if len(text) <= max_len:
            return [text]
        parts, cur = [], ""
        for s in re.split(r'(?<=[.!?])\s+', text):
            if len(cur) + len(s) < max_len:
                cur = (cur + " " + s).strip() if cur else s
            else:
                if cur:
                    parts.append(cur)
                cur = s
        if cur:
            parts.append(cur)
        return parts or [text[:max_len]]

    def translate(self, text, max_length=512):
        if not text.strip():
            return ""
        out = []
        for chunk in self._split(text, 400):
            inp = self.tokenizer(chunk, return_tensors="pt",
                                 padding=True, truncation=True, max_length=max_length)
            if self.mt_device == "cuda":
                inp = {k: v.cuda() for k, v in inp.items()}
            gen = self.mt.generate(**inp, max_length=max_length)
            out.append(self.tokenizer.decode(gen[0], skip_special_tokens=True))
        return " ".join(out)

    def process_audio(self, audio_path, language="uk"):
        audio_path = Path(audio_path)
        print(f"{audio_path.name}")

        t0 = time.perf_counter()
        asr_segments, info = self.asr.transcribe(
            str(audio_path),
            language=language,
            beam_size=5, best_of=5, temperature=0,
            condition_on_previous_text=True,
            vad_filter=True, word_timestamps=False,
        )
        segs = [{"id": s.id, "start": s.start, "end": s.end,
                 "text": s.text.strip()} for s in asr_segments]
        asr_time = time.perf_counter() - t0
        print(f"  ASR: {asr_time:.2f}s, {len(segs)} сегм.")

        t0 = time.perf_counter()
        result_segs = [Segment(id=s["id"], start=s["start"], end=s["end"],
                               text_uk=s["text"], text_en=self.translate(s["text"]))
                       for s in segs]
        mt_time = time.perf_counter() - t0
        print(f"  MT:  {mt_time:.2f}s")

        full_uk = " ".join(s["text"] for s in segs)
        full_en = " ".join(s.text_en for s in result_segs)
        duration = getattr(info, 'duration', 0) or 0
        total = asr_time + mt_time
        if duration:
            print(f"  RTF: {total/duration:.2f}x\n")

        return ProcessingResult(
            audio_file=audio_path.name,
            duration_sec=duration,
            segments=[asdict(s) for s in result_segs],
            full_text_uk=full_uk,
            full_text_en=full_en,
            asr_time_sec=asr_time,
            mt_time_sec=mt_time,
            total_time_sec=total,
            whisper_model=self.whisper_model_name,
            mt_model=self.mt_model_name,
            language=language,
        )


def process_corpus(input_dir="segments", output_dir="results",
                   whisper_model="large-v3",
                   mt_model="Helsinki-NLP/opus-mt-uk-en",
                   language="uk"):
    inp = Path(input_dir)
    out = Path(output_dir)
    (out / "asr").mkdir(parents=True, exist_ok=True)
    (out / "mt").mkdir(parents=True, exist_ok=True)

    files = sorted(inp.glob("*.wav"))
    if not files:
        print(f"Немає wav в {inp}")
        return
    print(f"{len(files)} файлів\n")

    pl = Pipeline(whisper_model, mt_model)

    all_results = []
    total_asr = total_mt = total_dur = 0.0

    for i, ap in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", end=" ")
        r = pl.process_audio(str(ap), language)
        all_results.append(asdict(r))
        total_asr += r.asr_time_sec
        total_mt  += r.mt_time_sec
        total_dur += r.duration_sec

        stem = ap.stem
        (out / "asr" / f"{stem}.txt").write_text(r.full_text_uk, encoding="utf-8")
        (out / "mt"  / f"{stem}.txt").write_text(r.full_text_en, encoding="utf-8")
        (out / f"{stem}.json").write_text(
            json.dumps(asdict(r), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "total_files": len(files),
        "total_audio_duration_sec": total_dur,
        "total_asr_time_sec": total_asr,
        "total_mt_time_sec": total_mt,
        "total_processing_time_sec": total_asr + total_mt,
        "avg_asr_rtf": total_asr / total_dur if total_dur else 0,
        "avg_mt_time_per_file_sec": total_mt / len(files),
        "overall_rtf": (total_asr + total_mt) / total_dur if total_dur else 0,
        "whisper_model": whisper_model,
        "mt_model": mt_model,
        "files": [r["audio_file"] for r in all_results],
    }
    (out / "corpus_results.json").write_text(
        json.dumps({"summary": summary, "results": all_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(files)} файлів, {total_dur/60:.1f} хв аудіо")
    print(f"ASR {total_asr:.1f}s, MT {total_mt:.1f}s, "
          f"RTF {(total_asr+total_mt)/total_dur:.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",  default="segments")
    p.add_argument("-o", "--output", default="results")
    p.add_argument("-m", "--model",  default="large-v3")
    p.add_argument("--mt", default="Helsinki-NLP/opus-mt-uk-en")
    p.add_argument("-l", "--language", default="uk")
    a = p.parse_args()
    process_corpus(a.input, a.output, a.model, a.mt, a.language)
