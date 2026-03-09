"""
LLM-based translation via GPT-4o, Claude, and Gemini APIs.

Usage:
    python llm_translate.py --model gpt4o
    python llm_translate.py --model claude
    python llm_translate.py --model gemini
    python llm_translate.py --model all

Requires env variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""

import argparse
import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai


@dataclass
class TranslationResult:
    segment_name: str
    source_text: str
    translated_text: str
    model: str
    tokens_input: int
    tokens_output: int
    time_sec: float


SYSTEM_PROMPT = """You are a professional translator specializing in Ukrainian to English translation of political and diplomatic speech.

Your task is to translate the provided Ukrainian text to English with high accuracy.

Guidelines:
1. Preserve proper names accurately (e.g., "Рустем Умєров" → "Rustem Umerov", "Кирило Буданов" → "Kyrylo Budanov")
2. Use standard English transliterations for Ukrainian place names (e.g., "Київ" → "Kyiv", "Дніпро" → "Dnipro", "Кривий Ріг" → "Kryvyi Rih")
3. Translate military terminology accurately (e.g., "ГУР" → "HUR/Defense Intelligence of Ukraine", "ДСНС" → "State Emergency Service")
4. Maintain the formal register appropriate for political discourse
5. Preserve the structure and meaning of the original text

Provide only the English translation, without any explanations or notes."""


class LLMTranslator:

    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None, google_key: Optional[str] = None):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_model = None

        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        if anthropic_key:
            self.anthropic_client = Anthropic(api_key=anthropic_key)
        if google_key:
            genai.configure(api_key=google_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

    def translate_gpt4o(self, text: str) -> TranslationResult:
        if not self.openai_client:
            raise ValueError("OpenAI API key not provided")

        t0 = time.perf_counter()

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate the following Ukrainian text to English:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=4096
        )

        elapsed = time.perf_counter() - t0

        return TranslationResult(
            segment_name="",
            source_text=text,
            translated_text=response.choices[0].message.content.strip(),
            model="gpt-4o",
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            time_sec=round(elapsed, 2)
        )

    def translate_claude(self, text: str) -> TranslationResult:
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not provided")

        t0 = time.perf_counter()

        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Translate the following Ukrainian text to English:\n\n{text}"}
            ]
        )

        elapsed = time.perf_counter() - t0

        return TranslationResult(
            segment_name="",
            source_text=text,
            translated_text=response.content[0].text.strip(),
            model="claude-sonnet-4-20250514",
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            time_sec=round(elapsed, 2)
        )

    def translate_gemini(self, text: str) -> TranslationResult:
        if not self.gemini_model:
            raise ValueError("Google API key not provided")

        t0 = time.perf_counter()

        prompt = f"{SYSTEM_PROMPT}\n\nTranslate the following Ukrainian text to English:\n\n{text}"
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=4096
            )
        )

        elapsed = time.perf_counter() - t0

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        return TranslationResult(
            segment_name="",
            source_text=text,
            translated_text=response.text.strip(),
            model="gemini-2.0-flash",
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            time_sec=round(elapsed, 2)
        )


def process_corpus(
    model: str,
    input_dir: Path,
    output_dir: Path,
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    google_key: Optional[str] = None
) -> list:
    translator = LLMTranslator(openai_key, anthropic_key, google_key)

    asr_files = sorted(input_dir.glob("*.txt"))
    print(f"Знайдено {len(asr_files)} ASR файлів")

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0

    for i, asr_file in enumerate(asr_files, 1):
        segment_name = asr_file.stem
        text = asr_file.read_text(encoding="utf-8").strip()

        print(f"[{i}/{len(asr_files)}] {segment_name}...", end=" ", flush=True)

        try:
            if model == "gpt4o":
                result = translator.translate_gpt4o(text)
            elif model == "claude":
                result = translator.translate_claude(text)
            elif model == "gemini":
                result = translator.translate_gemini(text)
            else:
                raise ValueError(f"Unknown model: {model}")

            result.segment_name = segment_name
            results.append(result)

            total_input_tokens += result.tokens_input
            total_output_tokens += result.tokens_output
            total_time += result.time_sec

            print(f"OK ({result.time_sec:.1f}s, {result.tokens_input}+{result.tokens_output} tokens)")

            output_file = output_dir / f"{segment_name}.txt"
            output_file.write_text(result.translated_text, encoding="utf-8")

            time.sleep(0.5)

        except Exception as e:
            print(f"ПОМИЛКА: {e}")
            continue

    print("\n" + "=" * 50)
    print(f"ПІДСУМОК ({model}):")
    print(f"  Оброблено сегментів: {len(results)}")
    print(f"  Загальний час: {total_time:.1f}s")
    print(f"  Токенів (вхід): {total_input_tokens}")
    print(f"  Токенів (вихід): {total_output_tokens}")
    print(f"  Результати збережено в: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM переклад для дослідження")
    parser.add_argument("-m", "--model", choices=["gpt4o", "claude", "gemini", "all"], default="all")
    parser.add_argument("-i", "--input", default=None)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    input_dir = Path(args.input) if args.input else base_dir / "results" / "asr"
    output_base = Path(args.output) if args.output else base_dir / "results"

    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
    GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY")

    models_to_run = ["gpt4o", "claude", "gemini"] if args.model == "all" else [args.model]

    all_results = {}

    for model in models_to_run:
        print("\n" + "=" * 60)
        print(f"ПЕРЕКЛАД ЧЕРЕЗ {model.upper()}")
        print("=" * 60 + "\n")

        output_dir = output_base / f"mt_{model}"
        output_dir.mkdir(parents=True, exist_ok=True)

        openai_key = OPENAI_KEY if model == "gpt4o" else None
        anthropic_key = ANTHROPIC_KEY if model == "claude" else None
        google_key = GOOGLE_KEY if model == "gemini" else None

        results = process_corpus(model, input_dir, output_dir, openai_key, anthropic_key, google_key)
        all_results[model] = [asdict(r) for r in results]

    report_file = output_base / "llm_translation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nЗагальний звіт збережено: {report_file}")


if __name__ == "__main__":
    main()
