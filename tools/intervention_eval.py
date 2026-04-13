import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy import stats

import sacrebleu
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

for resource, name in [('corpora/wordnet', 'wordnet'),
                        ('corpora/wordnet', 'omw-1.4'),
                        ('tokenizers/punkt_tab', 'punkt'),
                        ('tokenizers/punkt_tab', 'punkt_tab')]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name)


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

VARIANT_LABELS = {
    "baseline":        "OPUS-MT (baseline ASR)",
    "exact":           "OPUS-MT (exact-match)",
    "phonetic":        "OPUS-MT (phonetic, A)",
    "llm_corr":        "OPUS-MT (LLM-corrected)",
    "placeholder":     "OPUS-MT (code-switching 52 dict)",
    "placeholder_ext": "OPUS-MT (code-switching 176 dict)",
    "claude_haiku":    "Claude Haiku 4.5 direct",
    "claude":          "Claude Sonnet 4 direct",
    "gemini":          "Gemini 2.0 Flash direct",
    "llm_mt":          "GPT-4o direct",
}


def load_segments(path: Path) -> Dict[str, str]:
    out = {}
    for f in sorted(path.glob("*_seg*.txt")):
        out[f.stem] = f.read_text(encoding="utf-8").strip()
    return out


def metrics_for_pair(hyp: str, ref: str):
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
    P, R, F1 = bert_score([hyp], [ref], lang="en",
                          rescale_with_baseline=True, verbose=False)
    bertf1 = float(F1[0])
    ref_tokens = word_tokenize(ref.lower())
    hyp_tokens = word_tokenize(hyp.lower())
    met = meteor_score([ref_tokens], hyp_tokens)
    return bleu, bertf1, met


def main():
    print("=" * 68)
    print("ІНТЕРВЕНЦІЙНИЙ ЕКСПЕРИМЕНТ: ПОРІВНЯННЯ ВАРІАНТІВ MT")
    print("=" * 68)

    refs = load_segments(REF_EN)
    print(f"\nРеференсних сегментів EN: {len(refs)}\n")

    all_results: Dict[str, List[Dict]] = {}

    for var, path in VARIANTS.items():
        if not path.exists():
            print(f"SKIP {var}: {path}")
            continue
        hyps = load_segments(path)
        common = sorted(set(refs.keys()) & set(hyps.keys()))
        rows = []
        print(f"\n[{var}] {VARIANT_LABELS[var]}: {len(common)} сегментів")
        for name in common:
            bleu, bf1, met = metrics_for_pair(hyps[name], refs[name])
            rows.append({"segment": name, "bleu": bleu,
                         "bert_f1": bf1, "meteor": met})
        all_results[var] = rows
        df = pd.DataFrame(rows)
        print(f"  BLEU:    {df.bleu.mean():6.2f} ± {df.bleu.std():.2f}")
        print(f"  BERTScore-F1: {df.bert_f1.mean():.4f} ± {df.bert_f1.std():.4f}")
        print(f"  METEOR:  {df.meteor.mean():.4f} ± {df.meteor.std():.4f}")

    # ---------- Зведена таблиця ----------
    print("\n" + "=" * 68)
    print("ЗВЕДЕНА ПОРІВНЯЛЬНА ТАБЛИЦЯ")
    print("=" * 68)
    print(f"{'Variant':<32} {'BLEU':>9} {'BERT-F1':>9} {'METEOR':>9} "
          f"{'ΔBLEU':>10}")
    print("-" * 72)

    base_bleu = pd.DataFrame(all_results["baseline"]).bleu.mean()

    summary = {}
    for var in VARIANTS:
        if var not in all_results:
            continue
        df = pd.DataFrame(all_results[var])
        bleu_mean = float(df.bleu.mean())
        bf1_mean  = float(df.bert_f1.mean())
        met_mean  = float(df.meteor.mean())
        delta     = bleu_mean - base_bleu
        rel       = (delta / base_bleu * 100) if base_bleu else 0
        print(f"{VARIANT_LABELS[var]:<32} {bleu_mean:>9.2f} "
              f"{bf1_mean:>9.4f} {met_mean:>9.4f} "
              f"{delta:>+7.2f} ({rel:+5.1f}%)")
        summary[var] = {
            "label":    VARIANT_LABELS[var],
            "bleu":     round(bleu_mean, 2),
            "bert_f1":  round(bf1_mean, 4),
            "meteor":   round(met_mean, 4),
            "bleu_std": round(float(df.bleu.std()), 2),
            "delta_bleu_vs_baseline":     round(delta, 2),
            "rel_delta_bleu_vs_baseline": round(rel, 1),
            "n":        int(len(df)),
        }

    # ---------- Wilcoxon signed-rank (парне порівняння з baseline) ----------
    print("\n" + "-" * 68)
    print("WILCOXON SIGNED-RANK (vs baseline, парне на 12 сегментах)")
    print("-" * 68)
    base_df = pd.DataFrame(all_results["baseline"]).set_index("segment")
    wilcoxon = {}
    for var in VARIANTS:
        if var == "baseline" or var not in all_results:
            continue
        var_df = pd.DataFrame(all_results[var]).set_index("segment")
        common = sorted(set(base_df.index) & set(var_df.index))
        if len(common) < 4:
            continue
        b = base_df.loc[common, "bleu"].values
        v = var_df.loc[common, "bleu"].values
        try:
            stat, p = stats.wilcoxon(v, b, alternative="greater")
            print(f"  {var:<14} W={stat:6.1f}  p={p:.4f}  "
                  f"(n_pairs={len(common)})")
            wilcoxon[var] = {"W": float(stat), "p": round(float(p), 4),
                              "n_pairs": len(common)}
        except Exception as e:
            print(f"  {var}: wilcoxon failed ({e})")

    # ---------- Збереження ----------
    out = {"summary": summary, "wilcoxon_vs_baseline": wilcoxon,
           "per_segment": {var: rows for var, rows in all_results.items()}}
    with open(EVAL / "intervention_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    rows_csv = []
    for var, rows in all_results.items():
        for r in rows:
            rows_csv.append({"variant": var, **r})
    pd.DataFrame(rows_csv).to_csv(EVAL / "intervention_segments.csv", index=False)

    print(f"\nЗбережено:")
    print(f"  {EVAL / 'intervention_results.json'}")
    print(f"  {EVAL / 'intervention_segments.csv'}")


if __name__ == "__main__":
    main()
