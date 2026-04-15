# OPUS-MT TC-Big 240M

Модель `Helsinki-NLP/opus-mt-tc-big-zle-en` — seq2seq Transformer нового покоління (~240M параметрів, Marian-архітектура, ~900 MB ваг). Натренована на OPUS-2023 з усіх східнослов'янських мов (укр/рос/біл) у англійську; у ZLE-групі автоматично розпізнає вхідну мову.

## Метрики (12 сегментів політичного дискурсу, ~33 хв аудіо)

| Режим | BLEU | METEOR | BERTScore F1 | MT time | RTF |
|---|---|---|---|---|---|
| greedy (`--num-beams 1`, default) | **40,36** | 0,618 | — | 133 s | **0,066×** |
| beam=4 (`--num-beams 4`, quality) | 42,09 | 0,627 | 0,583 | 312 s | 0,156× |

Для довідки — baseline `opus-mt-uk-en` (77M): BLEU 23,12. TC-Big у греді-режимі дає **+17 BLEU** при **нижчому RTF** — тобто одночасно кращий і швидший за real-time.

## Запуск

```bash
pip install -r requirements.txt

# Папка з .txt файлами UA → папка EN (швидкий режим)
python translate.py --input-dir ./ua --output-dir ./en

# Один рядок
python translate.py --text "Привіт, світе"

# Stdin
echo "Привіт, світе" | python translate.py

# Якість важливіша за швидкість
python translate.py --input-dir ./ua --num-beams 4
```

При першому запуску HuggingFace завантажує ~900 MB ваг у локальний кеш.

---

# OPUS-MT TC-Big 240M (English)

`Helsinki-NLP/opus-mt-tc-big-zle-en` — a next-generation seq2seq Transformer (~240M params, Marian architecture, ~900 MB weights). Trained on OPUS-2023 from all East-Slavic languages (UK/RU/BE) into English; the ZLE group auto-detects the source language.

## Metrics (12 political-speech segments, ~33 min of audio)

| Mode | BLEU | METEOR | BERTScore F1 | MT time | RTF |
|---|---|---|---|---|---|
| greedy (`--num-beams 1`, default) | **40.36** | 0.618 | — | 133 s | **0.066×** |
| beam=4 (`--num-beams 4`, quality) | 42.09 | 0.627 | 0.583 | 312 s | 0.156× |

For reference, the 77M baseline `opus-mt-uk-en` scores BLEU 23.12. TC-Big in greedy mode is **+17 BLEU** at a **lower RTF** — simultaneously more accurate and faster than real-time.

## Usage

```bash
pip install -r requirements.txt

# Folder of UA .txt files → EN folder (fast mode)
python translate.py --input-dir ./ua --output-dir ./en

# Single string
python translate.py --text "Hello, world"

# Stdin
echo "Hello, world" | python translate.py

# Quality over speed
python translate.py --input-dir ./ua --num-beams 4
```

First run downloads ~900 MB of weights to the local HuggingFace cache.
