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
