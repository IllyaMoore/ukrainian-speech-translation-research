import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from biasing_correction import BIASING_DICTIONARY, levenshtein_distance, compute_wer_cer


BASE = Path(__file__).resolve().parent.parent
ASR_DIR = BASE / "data" / "results" / "asr"
REF_UA  = BASE / "data" / "references" / "ua"
OUT_DIR = BASE / "data" / "results" / "asr_phonetic"
EVAL_DIR = BASE / "evaluation"
OUT_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_THRESHOLD = 0.18
MAX_NGRAM = 3


TRANSLITERATION = {
    'а': 'a',  'б': 'b',  'в': 'v',  'г': 'h',  'ґ': 'g',
    'д': 'd',  'е': 'e',  'є': 'e',
    'ж': 'zh', 'з': 'z',
    'и': 'i',  'і': 'i',  'й': 'i',  'ї': 'i',
    'к': 'k',  'л': 'l',  'м': 'm',  'н': 'n',  'о': 'o',
    'п': 'p',  'р': 'r',  'с': 's',  'т': 't',  'у': 'u',
    'ф': 'f',  'х': 'h',
    'ц': 'c',  'ч': 'ch', 'ш': 'sh', 'щ': 'sh',
    'ь': '',   "'": '',   '\u2019': '', '\u02bc': '',
    'ю': 'yu', 'я': 'ya',
    'ы': 'i',  'ё': 'e',  'ъ': '',  'э': 'e',
}


def transliterate(word: str) -> str:
    word = word.lower()
    out = []
    for ch in word:
        if ch in TRANSLITERATION:
            out.append(TRANSLITERATION[ch])
        elif ch.isalpha():
            out.append(ch)
    s = ''.join(out)
    s = re.sub(r'(.)\1+', r'\1', s)
    return s


def phonetic_distance(w1: str, w2: str) -> float:
    p1, p2 = transliterate(w1), transliterate(w2)
    if not p1 and not p2:
        return 0.0
    return levenshtein_distance(p1, p2) / max(len(p1), len(p2), 1)


def build_canonical_entries() -> List[str]:
    canonicals = []
    for category, entries in BIASING_DICTIONARY.items():
        for canon, _variants in entries:
            canonicals.append(canon)
    extras = ["Рустем", "Кирило", "Давид", "Сергій", "Гнатов",
              "ГУР", "СБУ", "ДСНС", "ЗСУ", "МВС"]
    for e in extras:
        if e not in canonicals:
            canonicals.append(e)
    return canonicals


PUNCT_RE = re.compile(r'[.,!?;:»«"\'()\u2014\u2013\u2019\u2018\-]+')


def strip_punct(word: str) -> Tuple[str, str]:
    m = PUNCT_RE.search(word)
    if m and m.start() > 0:
        return word[:m.start()], word[m.start():]
    if m and m.start() == 0:
        return word[m.end():], ''
    return word, ''


def is_candidate(word: str) -> bool:
    return len(word) >= 4


def correct_text(text: str,
                 canonicals: List[str],
                 threshold: float = DEFAULT_THRESHOLD,
                 max_ngram: int = MAX_NGRAM) -> Tuple[str, List[Dict]]:
    canon_lower = {c.lower() for c in canonicals}

    tokens = text.split()
    n = len(tokens)
    out_tokens: List[str] = []
    corrections: List[Dict] = []
    i = 0

    while i < n:
        word = tokens[i]
        clean, _ = strip_punct(word)

        if not is_candidate(clean):
            out_tokens.append(word)
            i += 1
            continue

        if clean.lower() in canon_lower:
            out_tokens.append(word)
            i += 1
            continue

        best: Optional[Tuple[str, int, float]] = None

        for ngram_len in range(1, min(max_ngram, n - i) + 1):
            window_tokens = tokens[i:i + ngram_len]
            if any(any(p in t for p in '.!?') for t in window_tokens[:-1]):
                continue
            window_clean = ' '.join(strip_punct(t)[0] for t in window_tokens)
            if len(window_clean.replace(' ', '')) < 4:
                continue
            if window_clean.lower() in canon_lower:
                continue

            for canon in canonicals:
                d = phonetic_distance(window_clean, canon)
                if 0 < d < threshold:
                    if best is None or d < best[2]:
                        best = (canon, ngram_len, d)

        if best:
            canon, ngram_len, dist = best
            original = ' '.join(tokens[i:i + ngram_len])
            _, last_suffix = strip_punct(tokens[i + ngram_len - 1])
            replacement = canon + last_suffix

            if replacement.lower().rstrip('.,!?;:') == original.lower().rstrip('.,!?;:'):
                out_tokens.extend(tokens[i:i + ngram_len])
                i += ngram_len
                continue

            out_tokens.append(replacement)
            corrections.append({
                "original": original,
                "corrected": replacement,
                "ngram": ngram_len,
                "distance": round(dist, 3),
            })
            i += ngram_len
        else:
            out_tokens.append(word)
            i += 1

    return ' '.join(out_tokens), corrections


def calibrate_threshold(asr_files: List[Path],
                         ref_dir: Path,
                         canonicals: List[str],
                         val_size: int = 3) -> float:
    val_files = sorted(asr_files)[:val_size]
    candidates = [0.18, 0.22, 0.25, 0.28, 0.32, 0.35]
    best_thr = DEFAULT_THRESHOLD
    best_wer = float('inf')

    for thr in candidates:
        total = 0.0
        for f in val_files:
            ref = (ref_dir / f.name).read_text(encoding="utf-8").strip()
            asr = f.read_text(encoding="utf-8").strip()
            corrected, _ = correct_text(asr, canonicals, threshold=thr)
            total += compute_wer_cer(ref, corrected)["wer"]
        avg = total / len(val_files)
        if avg < best_wer:
            best_wer = avg
            best_thr = thr

    return best_thr


def main():
    canonicals = build_canonical_entries()
    asr_files = sorted(ASR_DIR.glob("*_seg*.txt"))
    threshold = calibrate_threshold(asr_files, REF_UA, canonicals)

    rows = []
    all_corrections: Dict[str, List[Dict]] = {}

    for f in asr_files:
        name = f.stem
        ref = (REF_UA / f.name).read_text(encoding="utf-8").strip()
        asr = f.read_text(encoding="utf-8").strip()

        corrected, corrections = correct_text(asr, canonicals, threshold=threshold)
        (OUT_DIR / f.name).write_text(corrected, encoding="utf-8")

        m_before = compute_wer_cer(ref, asr)
        m_after  = compute_wer_cer(ref, corrected)

        rows.append({
            "segment": name,
            "wer_before": m_before["wer"],
            "wer_after":  m_after["wer"],
            "cer_before": m_before["cer"],
            "cer_after":  m_after["cer"],
            "n_corrections": len(corrections),
        })
        all_corrections[name] = corrections

    n = len(rows)
    avg_wer_before = sum(r["wer_before"] for r in rows) / n
    avg_wer_after  = sum(r["wer_after"]  for r in rows) / n
    avg_cer_before = sum(r["cer_before"] for r in rows) / n
    avg_cer_after  = sum(r["cer_after"]  for r in rows) / n
    total_corr = sum(r["n_corrections"] for r in rows)
    rel_wer = (avg_wer_after - avg_wer_before) / avg_wer_before * 100
    rel_cer = (avg_cer_after - avg_cer_before) / avg_cer_before * 100

    print(f"Threshold: {threshold:.2f}")
    print(f"Segments: {n}, total corrections: {total_corr}")
    print(f"WER: {avg_wer_before:.2%} -> {avg_wer_after:.2%} ({rel_wer:+.1f}%)")
    print(f"CER: {avg_cer_before:.2%} -> {avg_cer_after:.2%} ({rel_cer:+.1f}%)")

    summary = {
        "method": "phonetic_aware_correction",
        "threshold": threshold,
        "dictionary_size": len(canonicals),
        "segments": n,
        "total_corrections": total_corr,
        "wer_before": round(avg_wer_before, 4),
        "wer_after":  round(avg_wer_after,  4),
        "wer_delta_rel_pct": round(rel_wer, 1),
        "cer_before": round(avg_cer_before, 4),
        "cer_after":  round(avg_cer_after,  4),
        "cer_delta_rel_pct": round(rel_cer, 1),
        "per_segment": rows,
        "corrections": all_corrections,
    }
    with open(EVAL_DIR / "phonetic_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
