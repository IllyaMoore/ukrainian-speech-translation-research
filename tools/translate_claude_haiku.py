import os
import time
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE = Path(__file__).resolve().parent.parent
ASR_DIR = BASE / "data" / "results" / "asr"
OUT_DIR = BASE / "data" / "results" / "mt_claude_haiku"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a professional translator specializing in Ukrainian to English translation of political and diplomatic speech.

Your task is to translate the provided Ukrainian text to English with high accuracy.

Guidelines:
1. Preserve proper names accurately (e.g., "Рустем Умєров" to "Rustem Umerov", "Кирило Буданов" to "Kyrylo Budanov")
2. Use standard English transliterations for Ukrainian place names (e.g., "Київ" to "Kyiv", "Дніпро" to "Dnipro", "Кривий Ріг" to "Kryvyi Rih")
3. Translate military terminology accurately (e.g., "ГУР" to "HUR/Defense Intelligence of Ukraine", "ДСНС" to "State Emergency Service")
4. Maintain the formal register appropriate for political discourse
5. Preserve the structure and meaning of the original text

Provide only the English translation, without any explanations or notes."""


def main():
    key = os.environ.get("CLAUDE_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("No CLAUDE_KEY in environment")

    client = Anthropic(api_key=key)
    files = sorted(ASR_DIR.glob("*_seg*.txt"))
    print(f"Files: {len(files)}, model: {MODEL}")

    total_t = 0.0
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                        "content": f"Translate the following Ukrainian text to English:\n\n{text}"}]
        )
        elapsed = time.perf_counter() - t0
        total_t += elapsed
        translated = resp.content[0].text.strip()
        (OUT_DIR / f.name).write_text(translated, encoding="utf-8")
        print(f"{f.stem}: {elapsed:.1f}s, in={resp.usage.input_tokens}, "
              f"out={resp.usage.output_tokens}")

    print(f"\nTotal time: {total_t:.1f}s")


if __name__ == "__main__":
    main()
