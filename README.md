# Entity-aware code-switching pipeline для uk-en перекладу

## Мета

Розробити локальний pipeline-level метод, що виправляє основну слабкість
каскадного ASR + OPUS-MT --- некоректну передачу власних назв --- і
перевершує за точністю передачі іменованих сутностей (Entity F1)
найменшу з сучасних LLM-моделей на сегментах домену.

## Найкоротший результат

| Система            | BLEU  | Entity F1 |
|--------------------|-------|-----------|
| OPUS-MT (baseline) | 19,01 | 0,447     |
| **Запропонований метод (code-switching + post-MT repair, 176-dict)** | **20,33** | **0,610** |
| Claude Haiku 4.5   | 55,25 | 0,837     |
| GPT-4o             | 57,00 | 0,860     |

- **+36% Entity F1** відносно baseline OPUS-MT
- Єдиний серед локальних методів зі статистично значущим приростом
  BLEU (Вілкоксон p = 0,046)
- Закриває 40% розриву до GPT-4o за Entity F1
- На 2 з 12 сегментів (17%) перевершує Claude Haiku 4.5;
  на ще 2 --- рівність; усього на 33% корпусу конкурує з найменшою LLM

## Структура

- `tools/` --- реалізація методу + оцінка
- `data/results/` --- виходи MT для всіх варіантів
- `evaluation/` --- зведена статистика і метрики
- `figures/`, `evaluation/figures/` --- графіки
- `docs/codeswitching_pipeline.md` --- документація методу

## Повна оцінка одним запуском

`tools/run_full_evaluation.py` — оркестратор, що за один прогін обчислює ASR-метрики (WER, CER), MT-метрики (BLEU, BERTScore, METEOR) для вибраних систем, статистичні тести та кореляції, малює графіки й генерує готові LaTeX-таблиці.

## Ізольовані seq2seq-моделі

У `models/` лежать самодостатні папки з окремими скриптами для кожної OPUS-MT-моделі — кожна має свій `translate.py`, `requirements.txt` та `README.md` з метриками:

| Папка | Модель | Параметри | BLEU | RTF (CPU, greedy) |
|---|---|---|---|---|
| `models/opus-mt-77m/` | `Helsinki-NLP/opus-mt-uk-en` | 77M | 23,12 | 0,078× |
| `models/opus-mt-tc-big-240m/` | `Helsinki-NLP/opus-mt-tc-big-zle-en` | 240M | 40,36 | 0,066× |

Запуск однієї моделі не залежить від решти репо: `cd models/opus-mt-77m/ && pip install -r requirements.txt && python translate.py --text "Привіт"`.

## Відтворення

```
pip install -r requirements.txt
python tools/phonetic_correction.py
python tools/placeholder_pipeline_ext.py
python tools/run_mt_on_variants.py
python tools/intervention_eval.py
python tools/intervention_entity_eval.py
python tools/visualize_intervention.py
```

---

# Entity-aware code-switching pipeline for UK→EN translation

## Goal

Develop a local pipeline-level method that fixes the main weakness of
the cascaded ASR + OPUS-MT setup — incorrect rendering of proper names —
and outperforms the smallest modern LLM on named-entity accuracy
(Entity F1) over in-domain segments.

## Headline result

| System             | BLEU  | Entity F1 |
|--------------------|-------|-----------|
| OPUS-MT (baseline) | 19.01 | 0.447     |
| **Proposed method (code-switching + post-MT repair, 176-dict)** | **20.33** | **0.610** |
| Claude Haiku 4.5   | 55.25 | 0.837     |
| GPT-4o             | 57.00 | 0.860     |

- **+36% Entity F1** over the OPUS-MT baseline
- The only local method with a statistically significant BLEU gain
  (Wilcoxon p = 0.046)
- Closes 40% of the gap to GPT-4o on Entity F1
- Outperforms Claude Haiku 4.5 on 2 of 12 segments (17%); ties on
  another 2; in total competes with the smallest LLM on 33% of the corpus

## Layout

- `tools/` — method implementation + evaluation
- `data/results/` — MT outputs for all variants
- `evaluation/` — aggregated statistics and metrics
- `figures/`, `evaluation/figures/` — plots
- `docs/codeswitching_pipeline.md` — method documentation

## Full evaluation in one run

`tools/run_full_evaluation.py` is an orchestrator that, in a single pass, computes ASR metrics (WER, CER) and MT metrics (BLEU, BERTScore, METEOR) for the selected systems, runs statistical tests and correlations, renders plots, and emits ready-to-include LaTeX tables.

## Isolated seq2seq models

The `models/` directory contains self-contained folders with a standalone script for each OPUS-MT model — each has its own `translate.py`, `requirements.txt`, and `README.md` with metrics:

| Folder | Model | Params | BLEU | RTF (CPU, greedy) |
|---|---|---|---|---|
| `models/opus-mt-77m/` | `Helsinki-NLP/opus-mt-uk-en` | 77M | 23.12 | 0.078× |
| `models/opus-mt-tc-big-240m/` | `Helsinki-NLP/opus-mt-tc-big-zle-en` | 240M | 40.36 | 0.066× |

A single model runs independently of the rest of the repo: `cd models/opus-mt-77m/ && pip install -r requirements.txt && python translate.py --text "Hello"`.

## Reproduction

```
pip install -r requirements.txt
python tools/phonetic_correction.py
python tools/placeholder_pipeline_ext.py
python tools/run_mt_on_variants.py
python tools/intervention_eval.py
python tools/intervention_entity_eval.py
python tools/visualize_intervention.py
```
