import json
import re
from pathlib import Path

import pandas as pd
import spacy


BASE = Path(__file__).resolve().parent.parent
REF_EN = BASE / "data" / "references" / "en"
EVAL = BASE / "evaluation"

VARIANTS = {
    "baseline":        BASE / "data" / "results" / "mt_baseline_v2",
    "exact":           BASE / "data" / "results" / "mt_exact",
    "phonetic":        BASE / "data" / "results" / "mt_phonetic",
    "llm_corr":        BASE / "data" / "results" / "mt_llm_corr",
    "placeholder":     BASE / "data" / "results" / "mt_placeholder",
    "placeholder_ext": BASE / "data" / "results" / "mt_placeholder_ext",
    "claude_haiku":    BASE / "data" / "results" / "mt_claude_haiku",
    "claude":          BASE / "data" / "results" / "mt_claude",
    "gemini":          BASE / "data" / "results" / "mt_gemini",
    "llm_mt":          BASE / "data" / "results" / "mt_gpt4o",
}

NLP = spacy.load("en_core_web_sm")
KEEP_LABELS = {"PERSON", "GPE", "LOC", "ORG", "NORP"}


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


def extract(text: str) -> set:
    doc = NLP(text)
    return {normalize(e.text) for e in doc.ents
            if e.label_ in KEEP_LABELS and len(e.text.strip()) > 1}


def main():
    refs = {f.stem: f.read_text(encoding="utf-8").strip()
            for f in sorted(REF_EN.glob("*_seg*.txt"))}
    ref_ents = {n: extract(t) for n, t in refs.items()}

    all_rows = []
    for var, path in VARIANTS.items():
        if not path.exists():
            continue
        for f in sorted(path.glob("*_seg*.txt")):
            name = f.stem
            if name not in ref_ents:
                continue
            sys_ents = extract(f.read_text(encoding="utf-8").strip())
            ref_e = ref_ents[name]
            if not ref_e:
                continue
            matched = sys_ents & ref_e
            p = len(matched) / len(sys_ents) if sys_ents else 0.0
            r = len(matched) / len(ref_e)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            all_rows.append({
                "variant": var, "segment": name,
                "n_sys": len(sys_ents), "n_ref": len(ref_e),
                "n_match": len(matched),
                "precision": p, "recall": r, "f1": f1,
            })

    df = pd.DataFrame(all_rows)
    print("=" * 60)
    print("ENTITY F1 за варіантами")
    print("=" * 60)
    print(f"\n{'Variant':<14} {'F1':>7} {'Prec':>7} {'Recall':>7} "
          f"{'avg_match':>10} {'avg_ref':>9}")
    print("-" * 60)
    summary = {}
    for var in VARIANTS:
        sub = df[df.variant == var]
        if sub.empty:
            continue
        f1 = sub.f1.mean()
        p = sub.precision.mean()
        r = sub.recall.mean()
        m = sub.n_match.mean()
        nr = sub.n_ref.mean()
        print(f"{var:<14} {f1:>6.3f}  {p:>6.3f}  {r:>6.3f}  "
              f"{m:>9.2f}  {nr:>8.2f}")
        summary[var] = {
            "f1_mean":   round(float(f1), 4),
            "f1_std":    round(float(sub.f1.std()), 4),
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "avg_match": round(float(m), 2),
            "avg_ref":   round(float(nr), 2),
        }

    df.to_csv(EVAL / "intervention_entity_f1.csv", index=False)
    with open(EVAL / "intervention_entity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nЗбережено: {EVAL}/intervention_entity_*")


if __name__ == "__main__":
    main()
