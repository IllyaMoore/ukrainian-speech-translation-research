import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer

from biasing_correction import BIASING_DICTIONARY, compute_wer_cer
from phonetic_correction import (
    transliterate, phonetic_distance, strip_punct, build_canonical_entries
)


BASE = Path(__file__).resolve().parent.parent
ASR_DIR = BASE / "data" / "results" / "asr"
REF_EN  = BASE / "data" / "references" / "en"
REF_UA  = BASE / "data" / "references" / "ua"
OUT_MT  = BASE / "data" / "results" / "mt_placeholder"
OUT_ASR = BASE / "data" / "results" / "asr_placeholder"
EVAL_DIR = BASE / "evaluation"
OUT_MT.mkdir(exist_ok=True, parents=True)
OUT_ASR.mkdir(exist_ok=True, parents=True)

MT_MODEL = "Helsinki-NLP/opus-mt-uk-en"
PHONETIC_THRESHOLD = 0.18

COMMON_WORDS_BLACKLIST = {
    "україни", "україна", "україні", "україну", "україн",
    "українці", "українки", "українці-українки",
    "українська", "українські", "українських",
    "росія", "росії", "росією", "росію",
    "америка", "америки", "америкою",
}


UK_TO_EN_DICT: Dict[str, str] = {
    "Рустем Умєров":      "Rustem Umerov",
    "Умєров":             "Umerov",
    "Кирило Буданов":     "Kyrylo Budanov",
    "Буданов":            "Budanov",
    "Будановим":          "Budanov",
    "Давид Арахамія":     "Davyd Arakhamia",
    "Арахамія":           "Arakhamia",
    "Сергій Кислиця":     "Serhiy Kyslytsya",
    "Кислиця":            "Kyslytsya",
    "Скібіцький":         "Skibitskyi",
    "Гнатов":             "Hnatov",
    "Хмара":              "Khmara",
    "Клименко":           "Klymenko",
    "Клименком":          "Klymenko",
    "Емманюель":          "Emmanuel",
    "Трамп":              "Trump",
    "Трампом":            "Trump",
    "Зеленський":         "Zelensky",
    "Федоров":            "Fedorov",
    "Федорова":           "Fedorov",
    "Кривий Ріг":         "Kryvyi Rih",
    "Київ":               "Kyiv",
    "Харків":             "Kharkiv",
    "Дніпро":             "Dnipro",
    "Одеса":              "Odesa",
    "Одесі":              "Odesa",
    "Запоріжжя":          "Zaporizhzhia",
    "Сумщина":            "Sumy region",
    "Чернігівщина":       "Chernihiv region",
    "Полтавська":         "Poltava",
    "Полтавській":        "Poltava",
    "Одещина":            "Odesa region",
    "Дніпровщина":        "Dnipro region",
    "Дніпровщини":        "Dnipro region",
    "Венесуели":          "Venezuela",
    "Балтії":             "Baltic states",
    "Емірати":            "Emirates",
    "Еміратах":           "Emirates",
    "окремого штурмового батальйону": "Separate Assault Battalion",
    "Національної":       "National",
    "Національна":        "National",
    "Хартія":             "Khartia",
    "ГУР":                "HUR",
    "СБУ":                "SBU",
    "ДСНС":               "SES of Ukraine",
    "ЗСУ":                "Armed Forces of Ukraine",
    "МВС":                "Ministry of Internal Affairs",
    "Укрзалізниці":       "Ukrzaliznytsia",
    "Укренерго":          "Ukrenergo",
    "Харківобленерго":    "Kharkivoblenerho",
    "Patriot PAC-3":      "Patriot PAC-3",
    "Patriot":            "Patriot",
}


def find_entities(text: str,
                   dict_uk: Dict[str, str],
                   canonicals: List[str]) -> List[Tuple[int, int, str, str]]:
    tokens = text.split()
    n = len(tokens)
    occupied = [False] * n
    matches: List[Tuple[int, int, str, str]] = []

    sorted_uk_keys = sorted(dict_uk.keys(), key=lambda s: -len(s.split()))

    for uk_key in sorted_uk_keys:
        en_val = dict_uk[uk_key]
        key_tokens = uk_key.split()
        L = len(key_tokens)
        for i in range(n - L + 1):
            if any(occupied[i:i + L]):
                continue
            window_tokens = tokens[i:i + L]
            window_clean = ' '.join(strip_punct(t)[0] for t in window_tokens)
            if window_clean.lower() == uk_key.lower():
                matches.append((i, i + L, uk_key, en_val))
                for k in range(i, i + L):
                    occupied[k] = True

    canonical_uk_forms = list(dict_uk.keys())

    i = 0
    while i < n:
        if occupied[i]:
            i += 1
            continue
        clean, _ = strip_punct(tokens[i])
        if len(clean) < 4 or clean.lower() in COMMON_WORDS_BLACKLIST:
            i += 1
            continue

        best = None
        for ngram_len in range(1, min(3, n - i) + 1):
            if any(occupied[i:i + ngram_len]):
                continue
            window_tokens = tokens[i:i + ngram_len]
            if any(any(p in t for p in '.!?') for t in window_tokens[:-1]):
                continue
            window_clean = ' '.join(strip_punct(t)[0] for t in window_tokens)
            wc_low = window_clean.lower()
            if len(wc_low.replace(' ', '')) < 4 or wc_low in COMMON_WORDS_BLACKLIST:
                continue

            for canon_uk in canonical_uk_forms:
                d = phonetic_distance(window_clean, canon_uk)
                if 0 < d < PHONETIC_THRESHOLD:
                    if best is None or d < best[2]:
                        best = (canon_uk, ngram_len, d)

        if best:
            canon_uk, L, dist = best
            matches.append((i, i + L, ' '.join(tokens[i:i + L]),
                             dict_uk[canon_uk]))
            for k in range(i, i + L):
                occupied[k] = True
            i += L
        else:
            i += 1

    matches.sort(key=lambda m: m[0])
    return matches


def insert_codeswitch(text: str,
                       entities: List[Tuple[int, int, str, str]]
                      ) -> Tuple[str, List[Tuple[str, str]]]:
    if not entities:
        return text, []

    tokens = text.split()
    out_tokens: List[str] = []
    i = 0
    ent_idx = 0
    log: List[Tuple[str, str]] = []

    while i < len(tokens):
        if ent_idx < len(entities) and entities[ent_idx][0] == i:
            start, end, uk_form, en_form = entities[ent_idx]
            _, last_suffix = strip_punct(tokens[end - 1])
            out_tokens.append(en_form + last_suffix)
            log.append((uk_form, en_form))
            i = end
            ent_idx += 1
        else:
            out_tokens.append(tokens[i])
            i += 1

    return ' '.join(out_tokens), log


def check_entity_survival(translated: str,
                            log: List[Tuple[str, str]]) -> int:
    survived = 0
    for _uk, en in log:
        first_word = en.split()[0]
        if first_word and first_word in translated:
            survived += 1
    return survived


REPAIR_THRESHOLD = 0.30
MIN_REPAIR_LEN = 4

PUNCT_LATIN_RE = re.compile(r'[.,!?;:"\'()\u2014\u2013\-]+')


def _strip_latin_punct(token: str) -> Tuple[str, str]:
    m = PUNCT_LATIN_RE.search(token)
    if m and m.start() > 0:
        return token[:m.start()], token[m.start():]
    if m and m.start() == 0:
        return token[m.end():], ''
    return token, ''


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                              prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _norm_lev(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    return _levenshtein(s1.lower(), s2.lower()) / max(len(s1), len(s2), 1)


def repair_translated(translated: str,
                       log: List[Tuple[str, str]],
                       threshold: float = REPAIR_THRESHOLD
                      ) -> Tuple[str, int]:
    if not log:
        return translated, 0

    n_repairs = 0
    out = translated

    en_forms = []
    seen = set()
    for _uk, en in log:
        if en not in seen:
            en_forms.append(en)
            seen.add(en)

    en_forms.sort(key=lambda s: -len(s))

    for en in en_forms:
        if len(en) < MIN_REPAIR_LEN:
            continue
        if en in out:
            continue

        en_word_count = len(en.split())
        tokens = out.split()
        n = len(tokens)

        best = None

        for ngram_len in (en_word_count, en_word_count + 1, en_word_count - 1):
            if ngram_len < 1 or ngram_len > n:
                continue
            for i in range(n - ngram_len + 1):
                window_tokens = tokens[i:i + ngram_len]
                clean_parts = [_strip_latin_punct(t)[0] for t in window_tokens]
                window_clean = ' '.join(clean_parts)
                if len(window_clean) < MIN_REPAIR_LEN:
                    continue
                if window_clean and en and \
                   window_clean[0].lower() != en[0].lower() and \
                   _levenshtein(window_clean[0].lower(), en[0].lower()) > 1:
                    continue
                d = _norm_lev(window_clean, en)
                if d < threshold:
                    if best is None or d < best[2]:
                        best = (i, i + ngram_len, d)

        if best:
            i, j, _d = best
            _, last_suffix = _strip_latin_punct(tokens[j - 1])
            new_tokens = tokens[:i] + [en + last_suffix] + tokens[j:]
            out = ' '.join(new_tokens)
            n_repairs += 1

    return out, n_repairs


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


def translate_chunked(text: str, tok, mdl, device: str) -> str:
    out_parts = []
    for p in split_text(text):
        inputs = tok(p, return_tensors="pt", padding=True, truncation=True,
                     max_length=512)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl.generate(**inputs, max_length=512)
        out_parts.append(tok.decode(outputs[0], skip_special_tokens=True))
    return " ".join(out_parts)


def main():
    tok, mdl, device = load_mt()
    asr_files = sorted(ASR_DIR.glob("*_seg*.txt"))
    summary_rows = []

    for f in asr_files:
        name = f.stem
        asr_text = f.read_text(encoding="utf-8").strip()

        entities = find_entities(asr_text, UK_TO_EN_DICT,
                                  build_canonical_entries())

        cs_text, log = insert_codeswitch(asr_text, entities)
        (OUT_ASR / f.name).write_text(cs_text, encoding="utf-8")

        translated_raw = translate_chunked(cs_text, tok, mdl, device)
        n_survived_pre = check_entity_survival(translated_raw, log)

        translated, n_repairs = repair_translated(translated_raw, log)
        n_survived_post = check_entity_survival(translated, log)

        (OUT_MT / f.name).write_text(translated, encoding="utf-8")

        summary_rows.append({
            "segment": name,
            "n_entities_found": len(entities),
            "n_survived_pre_repair":  n_survived_pre,
            "n_repairs_applied":      n_repairs,
            "n_survived_post_repair": n_survived_post,
            "survival_pre":  round(n_survived_pre  / len(entities), 3) if entities else 1.0,
            "survival_post": round(n_survived_post / len(entities), 3) if entities else 1.0,
        })

        print(f"{name}: found={len(entities)}, "
              f"pre={n_survived_pre}/{len(entities)}, "
              f"repairs={n_repairs}, "
              f"post={n_survived_post}/{len(entities)}")

    total_entities = sum(r["n_entities_found"] for r in summary_rows)
    total_pre  = sum(r["n_survived_pre_repair"]  for r in summary_rows)
    total_post = sum(r["n_survived_post_repair"] for r in summary_rows)
    total_rep  = sum(r["n_repairs_applied"]      for r in summary_rows)
    survival_pre  = total_pre  / total_entities if total_entities else 0.0
    survival_post = total_post / total_entities if total_entities else 0.0

    print(f"\nTotal entities: {total_entities}")
    print(f"Survived pre-repair:  {total_pre}/{total_entities}  ({survival_pre:.1%})")
    print(f"Repairs applied:      {total_rep}")
    print(f"Survived post-repair: {total_post}/{total_entities}  ({survival_post:.1%})")

    summary = {
        "method": "entity_codeswitching_pipeline_with_repair",
        "dict_size": len(UK_TO_EN_DICT),
        "phonetic_threshold": PHONETIC_THRESHOLD,
        "repair_threshold": REPAIR_THRESHOLD,
        "total_entities":         total_entities,
        "total_survived_pre":     total_pre,
        "total_repairs_applied":  total_rep,
        "total_survived_post":    total_post,
        "survival_rate_pre":      round(survival_pre, 4),
        "survival_rate_post":     round(survival_post, 4),
        "per_segment": summary_rows,
    }
    with open(EVAL_DIR / "placeholder_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
