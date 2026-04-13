"""LLM-переклад UA->EN через GPT-4o / Claude / Gemini.

  python llm_translate.py --model gpt4o
  python llm_translate.py --model all

Потребує: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

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
1. Preserve proper names accurately (e.g., "Рустем Умєров" -> "Rustem Umerov", "Кирило Буданов" -> "Kyrylo Budanov")
2. Use standard English transliterations for Ukrainian place names (e.g., "Київ" -> "Kyiv", "Дніпро" -> "Dnipro", "Кривий Ріг" -> "Kryvyi Rih")
3. Translate military terminology accurately (e.g., "ГУР" -> "HUR/Defense Intelligence of Ukraine", "ДСНС" -> "State Emergency Service")
4. Maintain the formal register appropriate for political discourse
5. Preserve the structure and meaning of the original text

Provide only the English translation, without any explanations or notes."""

USER_TEMPLATE = "Translate the following Ukrainian text to English:\n\n{text}"


class LLMTranslator:
    def __init__(self, openai_key=None, anthropic_key=None, google_key=None):
        self.openai = OpenAI(api_key=openai_key) if openai_key else None
        self.anthropic = Anthropic(api_key=anthropic_key) if anthropic_key else None
        self.gemini = None
        if google_key:
            genai.configure(api_key=google_key)
            self.gemini = genai.GenerativeModel('gemini-2.0-flash')

    def gpt4o(self, text):
        if not self.openai:
            raise ValueError("no OpenAI key")
        t0 = time.perf_counter()
        r = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=text)},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return TranslationResult(
            segment_name="",
            source_text=text,
            translated_text=r.choices[0].message.content.strip(),
            model="gpt-4o",
            tokens_input=r.usage.prompt_tokens,
            tokens_output=r.usage.completion_tokens,
            time_sec=round(time.perf_counter() - t0, 2),
        )

    def claude(self, text):
        if not self.anthropic:
            raise ValueError("no Anthropic key")
        t0 = time.perf_counter()
        r = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(text=text)}],
        )
        return TranslationResult(
            segment_name="",
            source_text=text,
            translated_text=r.content[0].text.strip(),
            model="claude-sonnet-4-20250514",
            tokens_input=r.usage.input_tokens,
            tokens_output=r.usage.output_tokens,
            time_sec=round(time.perf_counter() - t0, 2),
        )

    def gemini_translate(self, text):
        if not self.gemini:
            raise ValueError("no Google key")
        t0 = time.perf_counter()
        r = self.gemini.generate_content(
            f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(text=text)}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3, max_output_tokens=4096),
        )
        u = r.usage_metadata
        return TranslationResult(
            segment_name="",
            source_text=text,
            translated_text=r.text.strip(),
            model="gemini-2.0-flash",
            tokens_input=u.prompt_token_count if u else 0,
            tokens_output=u.candidates_token_count if u else 0,
            time_sec=round(time.perf_counter() - t0, 2),
        )

    def call(self, model, text):
        return {
            "gpt4o":  self.gpt4o,
            "claude": self.claude,
            "gemini": self.gemini_translate,
        }[model](text)


def process_corpus(model, input_dir, output_dir, **keys):
    tr = LLMTranslator(**keys)
    files = sorted(input_dir.glob("*.txt"))
    print(f"{len(files)} ASR файлів")

    results = []
    tot_in = tot_out = tot_t = 0

    for i, f in enumerate(files, 1):
        name = f.stem
        text = f.read_text(encoding="utf-8").strip()
        print(f"[{i}/{len(files)}] {name}...", end=" ", flush=True)

        try:
            r = tr.call(model, text)
        except Exception as e:
            print(f"FAIL: {e}")
            continue

        r.segment_name = name
        results.append(r)
        tot_in  += r.tokens_input
        tot_out += r.tokens_output
        tot_t   += r.time_sec

        print(f"OK ({r.time_sec:.1f}s, {r.tokens_input}+{r.tokens_output} tok)")
        (output_dir / f"{name}.txt").write_text(r.translated_text, encoding="utf-8")
        time.sleep(0.5)

    print(f"\n{model}: {len(results)} сегм., {tot_t:.1f}s, "
          f"tokens {tot_in}/{tot_out} -> {output_dir}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model",
                   choices=["gpt4o", "claude", "gemini", "all"], default="all")
    p.add_argument("-i", "--input",  default=None)
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    base = Path(__file__).parent.parent
    input_dir  = Path(args.input)  if args.input  else base / "results" / "asr"
    output_base = Path(args.output) if args.output else base / "results"

    keys_for = {
        "gpt4o":  {"openai_key":   os.environ.get("OPENAI_API_KEY")},
        "claude": {"anthropic_key": os.environ.get("ANTHROPIC_API_KEY")},
        "gemini": {"google_key":   os.environ.get("GOOGLE_API_KEY")},
    }

    models = ["gpt4o", "claude", "gemini"] if args.model == "all" else [args.model]

    all_results = {}
    for m in models:
        print(f"\n--- {m.upper()} ---")
        out = output_base / f"mt_{m}"
        out.mkdir(parents=True, exist_ok=True)
        rs = process_corpus(m, input_dir, out, **keys_for[m])
        all_results[m] = [asdict(r) for r in rs]

    report = output_base / "llm_translation_report.json"
    report.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n-> {report}")


if __name__ == "__main__":
    main()
