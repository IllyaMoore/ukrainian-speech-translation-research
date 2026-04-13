import time
import re
from pathlib import Path
from typing import List

import torch
from transformers import MarianMTModel, MarianTokenizer


BASE = Path(__file__).resolve().parent.parent

VARIANTS = {
    "baseline_v2": (BASE / "data" / "results" / "asr",               BASE / "data" / "results" / "mt_baseline_v2"),
    "exact":       (BASE / "data" / "results" / "asr_corrected",     BASE / "data" / "results" / "mt_exact"),
    "llm_corr":    (BASE / "data" / "results" / "asr_corrected_llm", BASE / "data" / "results" / "mt_llm_corr"),
    "phonetic":    (BASE / "data" / "results" / "asr_phonetic",      BASE / "data" / "results" / "mt_phonetic"),
}

MT_MODEL = "Helsinki-NLP/opus-mt-uk-en"


def load_mt():
    tok = MarianTokenizer.from_pretrained(MT_MODEL)
    mdl = MarianMTModel.from_pretrained(MT_MODEL)
    if torch.cuda.is_available():
        mdl = mdl.cuda()
        device = "cuda"
    else:
        device = "cpu"
    mdl.eval()
    return tok, mdl, device


def split_text(text: str, max_chars: int = 400) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r'(?<=[.!?])\s+', text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) < max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out or [text[:max_chars]]


def translate_text(text: str, tok, mdl, device: str, max_length: int = 512) -> str:
    if not text.strip():
        return ""
    parts = split_text(text)
    out_parts = []
    for p in parts:
        inputs = tok(p, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl.generate(**inputs, max_length=max_length)
        out_parts.append(tok.decode(outputs[0], skip_special_tokens=True))
    return " ".join(out_parts)


def main():
    tok, mdl, device = load_mt()

    for variant, (in_dir, out_dir) in VARIANTS.items():
        if not in_dir.exists():
            print(f"skip {variant}: {in_dir} not found")
            continue
        out_dir.mkdir(exist_ok=True, parents=True)
        files = sorted(in_dir.glob("*_seg*.txt"))
        print(f"[{variant}] {len(files)} files -> {out_dir}/")

        t0 = time.perf_counter()
        for f in files:
            text = f.read_text(encoding="utf-8").strip()
            translated = translate_text(text, tok, mdl, device)
            (out_dir / f.name).write_text(translated, encoding="utf-8")
        print(f"  elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
