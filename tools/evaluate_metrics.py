import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd

import jiwer
import sacrebleu
from bert_score import score as bert_score
import nltk

for resource, name in [('corpora/wordnet', 'wordnet'),
                        ('corpora/wordnet', 'omw-1.4'),
                        ('tokenizers/punkt', 'punkt'),
                        ('tokenizers/punkt', 'punkt_tab')]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name)

from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


@dataclass
class ASRMetrics:
    segment_name: str
    wer: float
    cer: float
    words_ref: int
    words_hyp: int
    substitutions: int
    deletions: int
    insertions: int


@dataclass
class MTMetrics:
    segment_name: str
    bleu: float
    bert_precision: float
    bert_recall: float
    bert_f1: float
    meteor: float


def normalize(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()


def asr_metrics(reference, hypothesis, name):
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    return ASRMetrics(
        segment_name=name,
        wer=round(w.wer, 4),
        cer=round(c.cer, 4),
        words_ref=len(ref.split()),
        words_hyp=len(hyp.split()),
        substitutions=w.substitutions,
        deletions=w.deletions,
        insertions=w.insertions,
    )


def mt_metrics(reference, hypothesis, name):
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
    P, R, F1 = bert_score([hypothesis], [reference],
                          lang="en", rescale_with_baseline=True, verbose=False)
    met = meteor_score([word_tokenize(reference.lower())],
                       word_tokenize(hypothesis.lower()))
    return MTMetrics(
        segment_name=name,
        bleu=round(bleu, 2),
        bert_precision=round(P[0].item(), 4),
        bert_recall=round(R[0].item(), 4),
        bert_f1=round(F1[0].item(), 4),
        meteor=round(met, 4),
    )


def load_texts(folder):
    return {f.stem: f.read_text(encoding="utf-8").strip()
            for f in folder.glob("*_seg*.txt")}


def load_results(folder):
    out = {}
    for jf in folder.glob("*_seg*.json"):
        with open(jf, encoding="utf-8") as fh:
            out[jf.stem] = json.load(fh)
    return out


def stats_block(df, cols, decimals):
    out = {}
    for c, d in zip(cols, decimals):
        out[c] = {
            "mean":   round(df[c].mean(),   d),
            "std":    round(df[c].std(),    d),
            "min":    round(df[c].min(),    d),
            "max":    round(df[c].max(),    d),
            "median": round(df[c].median(), d),
        }
    return out


def main():
    base = Path(__file__).parent.parent
    ref_ua = load_texts(base / "Translations UA")
    ref_en = load_texts(base / "Translations EN")
    pipeline = load_results(base / "results")

    out_dir = base / "evaluation"
    out_dir.mkdir(exist_ok=True)

    common = sorted(set(ref_ua) & set(ref_en) & set(pipeline))
    print(f"Сегментів для оцінки: {len(common)}")
    if not common:
        return

    asr_rows, mt_rows = [], []

    print("\nASR (WER/CER):")
    for name in common:
        m = asr_metrics(ref_ua[name], pipeline[name]["full_text_uk"], name)
        asr_rows.append(m)
        print(f"  {name}: WER={m.wer:.2%}, CER={m.cer:.2%}")

    print("\nMT (BLEU/BERTScore/METEOR):")
    for name in common:
        print(f"  {name}...", end=" ", flush=True)
        m = mt_metrics(ref_en[name], pipeline[name]["full_text_en"], name)
        mt_rows.append(m)
        print(f"BLEU={m.bleu:.1f}, F1={m.bert_f1:.3f}, METEOR={m.meteor:.3f}")

    asr_df = pd.DataFrame([asdict(m) for m in asr_rows])
    mt_df  = pd.DataFrame([asdict(m) for m in mt_rows])

    asr_df.to_csv(out_dir / "asr_metrics.csv", index=False)
    mt_df.to_csv(out_dir / "mt_metrics.csv", index=False)

    summary = {
        "asr": {
            **stats_block(asr_df, ["wer", "cer"], [4, 4]),
            "total_words_ref":     int(asr_df["words_ref"].sum()),
            "total_substitutions": int(asr_df["substitutions"].sum()),
            "total_deletions":     int(asr_df["deletions"].sum()),
            "total_insertions":    int(asr_df["insertions"].sum()),
        },
        "mt": stats_block(mt_df, ["bleu", "bert_f1", "meteor"], [2, 4, 4]),
        "segments_evaluated": len(common),
        "segments": common,
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detailed = {
        "asr_metrics": [asdict(m) for m in asr_rows],
        "mt_metrics":  [asdict(m) for m in mt_rows],
    }
    (out_dir / "evaluation_detailed.json").write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWER mean={asr_df['wer'].mean():.2%}, CER mean={asr_df['cer'].mean():.2%}")
    print(f"BLEU mean={mt_df['bleu'].mean():.2f}, "
          f"F1 mean={mt_df['bert_f1'].mean():.4f}, "
          f"METEOR mean={mt_df['meteor'].mean():.4f}")


if __name__ == "__main__":
    main()
