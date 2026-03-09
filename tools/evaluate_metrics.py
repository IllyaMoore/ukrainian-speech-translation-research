"""
Compute ASR and MT quality metrics: WER, CER, BLEU, BERTScore, METEOR.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import pandas as pd

import jiwer
import sacrebleu
from bert_score import score as bert_score
import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_asr_metrics(reference: str, hypothesis: str, name: str) -> ASRMetrics:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    wer_output = jiwer.process_words(ref_norm, hyp_norm)
    cer_output = jiwer.process_characters(ref_norm, hyp_norm)

    return ASRMetrics(
        segment_name=name,
        wer=round(wer_output.wer, 4),
        cer=round(cer_output.cer, 4),
        words_ref=len(ref_norm.split()),
        words_hyp=len(hyp_norm.split()),
        substitutions=wer_output.substitutions,
        deletions=wer_output.deletions,
        insertions=wer_output.insertions
    )


def compute_mt_metrics(reference: str, hypothesis: str, name: str) -> MTMetrics:
    bleu_result = sacrebleu.sentence_bleu(hypothesis, [reference])

    P, R, F1 = bert_score(
        [hypothesis], [reference],
        lang="en", rescale_with_baseline=True, verbose=False
    )

    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    meteor = meteor_score([ref_tokens], hyp_tokens)

    return MTMetrics(
        segment_name=name,
        bleu=round(bleu_result.score, 2),
        bert_precision=round(P[0].item(), 4),
        bert_recall=round(R[0].item(), 4),
        bert_f1=round(F1[0].item(), 4),
        meteor=round(meteor, 4)
    )


def load_reference_texts(translations_dir: Path) -> Dict[str, str]:
    texts = {}
    for txt_file in translations_dir.glob("*_seg*.txt"):
        texts[txt_file.stem] = txt_file.read_text(encoding="utf-8").strip()
    return texts


def load_pipeline_results(results_dir: Path) -> Dict[str, dict]:
    results = {}
    for json_file in results_dir.glob("*_seg*.json"):
        with open(json_file, encoding="utf-8") as f:
            results[json_file.stem] = json.load(f)
    return results


def main():
    base_dir = Path(__file__).parent.parent

    translations_ua = base_dir / "Translations UA"
    translations_en = base_dir / "Translations EN"
    results_dir = base_dir / "results"
    output_dir = base_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ОБЧИСЛЕННЯ МЕТРИК ЯКОСТІ ASR ТА MT")
    print("=" * 60)

    print("\nЗавантаження даних...")
    ref_ua = load_reference_texts(translations_ua)
    ref_en = load_reference_texts(translations_en)
    pipeline_results = load_pipeline_results(results_dir)

    print(f"  Референсних UA текстів: {len(ref_ua)}")
    print(f"  Референсних EN текстів: {len(ref_en)}")
    print(f"  Результатів pipeline: {len(pipeline_results)}")

    common_segments = set(ref_ua.keys()) & set(ref_en.keys()) & set(pipeline_results.keys())
    print(f"  Спільних сегментів для оцінки: {len(common_segments)}")

    if not common_segments:
        print("ПОМИЛКА: Немає спільних сегментів!")
        return

    print("\n" + "-" * 60)
    print("ОБЧИСЛЕННЯ ASR МЕТРИК (WER, CER)")
    print("-" * 60)

    asr_metrics_list = []
    for name in sorted(common_segments):
        metrics = compute_asr_metrics(ref_ua[name], pipeline_results[name]["full_text_uk"], name)
        asr_metrics_list.append(metrics)
        print(f"  {name}: WER={metrics.wer:.2%}, CER={metrics.cer:.2%}")

    print("\n" + "-" * 60)
    print("ОБЧИСЛЕННЯ MT МЕТРИК (BLEU, BERTScore, METEOR)")
    print("-" * 60)

    mt_metrics_list = []
    for name in sorted(common_segments):
        print(f"  Обчислення для {name}...", end=" ", flush=True)
        metrics = compute_mt_metrics(ref_en[name], pipeline_results[name]["full_text_en"], name)
        mt_metrics_list.append(metrics)
        print(f"BLEU={metrics.bleu:.1f}, BERTScore-F1={metrics.bert_f1:.3f}, METEOR={metrics.meteor:.3f}")

    print("\n" + "=" * 60)
    print("ЗВЕДЕНІ РЕЗУЛЬТАТИ")
    print("=" * 60)

    asr_df = pd.DataFrame([asdict(m) for m in asr_metrics_list])
    print("\n--- ASR МЕТРИКИ ---")
    print(f"WER:  mean={asr_df['wer'].mean():.2%}, std={asr_df['wer'].std():.2%}, "
          f"min={asr_df['wer'].min():.2%}, max={asr_df['wer'].max():.2%}")
    print(f"CER:  mean={asr_df['cer'].mean():.2%}, std={asr_df['cer'].std():.2%}, "
          f"min={asr_df['cer'].min():.2%}, max={asr_df['cer'].max():.2%}")

    mt_df = pd.DataFrame([asdict(m) for m in mt_metrics_list])
    print("\n--- MT МЕТРИКИ ---")
    print(f"BLEU:        mean={mt_df['bleu'].mean():.2f}, std={mt_df['bleu'].std():.2f}, "
          f"min={mt_df['bleu'].min():.2f}, max={mt_df['bleu'].max():.2f}")
    print(f"BERTScore-F1: mean={mt_df['bert_f1'].mean():.4f}, std={mt_df['bert_f1'].std():.4f}, "
          f"min={mt_df['bert_f1'].min():.4f}, max={mt_df['bert_f1'].max():.4f}")
    print(f"METEOR:      mean={mt_df['meteor'].mean():.4f}, std={mt_df['meteor'].std():.4f}, "
          f"min={mt_df['meteor'].min():.4f}, max={mt_df['meteor'].max():.4f}")

    print("\n" + "-" * 60)
    print("ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
    print("-" * 60)

    asr_df.to_csv(output_dir / "asr_metrics.csv", index=False)
    mt_df.to_csv(output_dir / "mt_metrics.csv", index=False)
    print(f"  Збережено: {output_dir / 'asr_metrics.csv'}")
    print(f"  Збережено: {output_dir / 'mt_metrics.csv'}")

    summary = {
        "asr": {
            "wer": {
                "mean": round(asr_df['wer'].mean(), 4),
                "std": round(asr_df['wer'].std(), 4),
                "min": round(asr_df['wer'].min(), 4),
                "max": round(asr_df['wer'].max(), 4),
                "median": round(asr_df['wer'].median(), 4)
            },
            "cer": {
                "mean": round(asr_df['cer'].mean(), 4),
                "std": round(asr_df['cer'].std(), 4),
                "min": round(asr_df['cer'].min(), 4),
                "max": round(asr_df['cer'].max(), 4),
                "median": round(asr_df['cer'].median(), 4)
            },
            "total_words_ref": int(asr_df['words_ref'].sum()),
            "total_substitutions": int(asr_df['substitutions'].sum()),
            "total_deletions": int(asr_df['deletions'].sum()),
            "total_insertions": int(asr_df['insertions'].sum())
        },
        "mt": {
            "bleu": {
                "mean": round(mt_df['bleu'].mean(), 2),
                "std": round(mt_df['bleu'].std(), 2),
                "min": round(mt_df['bleu'].min(), 2),
                "max": round(mt_df['bleu'].max(), 2),
                "median": round(mt_df['bleu'].median(), 2)
            },
            "bert_f1": {
                "mean": round(mt_df['bert_f1'].mean(), 4),
                "std": round(mt_df['bert_f1'].std(), 4),
                "min": round(mt_df['bert_f1'].min(), 4),
                "max": round(mt_df['bert_f1'].max(), 4),
                "median": round(mt_df['bert_f1'].median(), 4)
            },
            "meteor": {
                "mean": round(mt_df['meteor'].mean(), 4),
                "std": round(mt_df['meteor'].std(), 4),
                "min": round(mt_df['meteor'].min(), 4),
                "max": round(mt_df['meteor'].max(), 4),
                "median": round(mt_df['meteor'].median(), 4)
            }
        },
        "segments_evaluated": len(common_segments),
        "segments": sorted(list(common_segments))
    }

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Збережено: {output_dir / 'evaluation_summary.json'}")

    detailed = {
        "asr_metrics": [asdict(m) for m in asr_metrics_list],
        "mt_metrics": [asdict(m) for m in mt_metrics_list]
    }
    with open(output_dir / "evaluation_detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    print(f"  Збережено: {output_dir / 'evaluation_detailed.json'}")

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
