# OPUS-MT 77M — baseline

Модель `Helsinki-NLP/opus-mt-uk-en` — класичний seq2seq Transformer (~77M параметрів, Marian-архітектура, ~300 MB ваг) для прямого українсько-англійського перекладу.

## Метрики (12 сегментів політичного дискурсу, ~33 хв аудіо)

| Метрика | Значення |
|---|---|
| BLEU | **23,12** |
| BERTScore F1 | 0,397 |
| METEOR | 0,452 |
| MT time (CPU, greedy) | 157 s |
| RTF (MT / audio) | 0,078× |
| Real-time запас | у 12,8× швидше за real-time |

## Запуск

```bash
pip install -r requirements.txt

# Папка з .txt файлами UA → папка EN
python translate.py --input-dir ./ua --output-dir ./en

# Один рядок
python translate.py --text "Привіт, світе"

# Stdin
echo "Привіт, світе" | python translate.py

# Якість замість швидкості
python translate.py --input-dir ./ua --num-beams 4
```

При першому запуску HuggingFace завантажує ~300 MB ваг у локальний кеш.

---

# OPUS-MT 77M — baseline (English)

`Helsinki-NLP/opus-mt-uk-en` — classic seq2seq Transformer (~77M params, Marian architecture, ~300 MB weights) for direct Ukrainian → English translation.

## Metrics (12 political-speech segments, ~33 min of audio)

| Metric | Value |
|---|---|
| BLEU | **23.12** |
| BERTScore F1 | 0.397 |
| METEOR | 0.452 |
| MT time (CPU, greedy) | 157 s |
| RTF (MT / audio) | 0.078× |
| Real-time headroom | 12.8× faster than real-time |

## Usage

```bash
pip install -r requirements.txt

# Folder of UA .txt files → EN folder
python translate.py --input-dir ./ua --output-dir ./en

# Single string
python translate.py --text "Hello, world"

# Stdin
echo "Hello, world" | python translate.py

# Quality over speed
python translate.py --input-dir ./ua --num-beams 4
```

First run downloads ~300 MB of weights to the local HuggingFace cache.
