"""
Standalone UA→EN translator using Helsinki-NLP/opus-mt-tc-big-zle-en (240M params).

Багатомовний варіант OPUS-MT: East Slavic (UK/RU/BE) → English,
натренований на OPUS-2023.

Usage:
    pip install -r requirements.txt
    python translate.py --input-dir ./ua --output-dir ./en
    python translate.py --text "Привіт, світе"
    echo "Привіт" | python translate.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List

import torch
from transformers import MarianMTModel, MarianTokenizer


MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-zle-en"
MAX_CHARS = 400
MAX_LENGTH = 512


def split_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) < max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks or [text[:max_chars]]


def translate(text: str, tok, mdl, device: str, num_beams: int = 1) -> str:
    if not text.strip():
        return ""
    out = []
    for chunk in split_text(text):
        inputs = tok(chunk, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LENGTH)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl.generate(**inputs, max_length=MAX_LENGTH, num_beams=num_beams)
        out.append(tok.decode(outputs[0], skip_special_tokens=True))
    return " ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=f"UA→EN via {MODEL_ID}")
    parser.add_argument("--input-dir", type=Path,
                        help="Folder with UA .txt files")
    parser.add_argument("--output-dir", type=Path, default=Path("translations"),
                        help="Output folder for EN translations")
    parser.add_argument("--text", help="Single UA text to translate")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam size (1=greedy, faster; 4+=higher quality)")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...", file=sys.stderr)
    t_load = time.perf_counter()
    tok = MarianTokenizer.from_pretrained(MODEL_ID)
    mdl = MarianMTModel.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        mdl = mdl.cuda()
    mdl.eval()
    print(f"  Loaded in {time.perf_counter() - t_load:.1f}s, device={device}",
          file=sys.stderr)

    if args.text:
        print(translate(args.text, tok, mdl, device, args.num_beams))
        return 0

    if args.input_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(args.input_dir.glob("*.txt"))
        total = 0.0
        for f in files:
            text = f.read_text(encoding="utf-8").strip()
            t = time.perf_counter()
            en = translate(text, tok, mdl, device, args.num_beams)
            dt = time.perf_counter() - t
            total += dt
            (args.output_dir / f.name).write_text(en, encoding="utf-8")
            print(f"  {f.name}: {dt:.2f}s", file=sys.stderr)
        if files:
            print(f"\n{len(files)} files in {total:.1f}s "
                  f"(avg {total/len(files):.2f}s/file)", file=sys.stderr)
        return 0

    text = sys.stdin.read().strip()
    if text:
        print(translate(text, tok, mdl, device, args.num_beams))
    return 0


if __name__ == "__main__":
    sys.exit(main())
