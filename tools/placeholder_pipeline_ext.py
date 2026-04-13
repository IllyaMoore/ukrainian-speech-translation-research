import json
from pathlib import Path

import torch

from placeholder_pipeline import (
    find_entities_in_text as find_entities,
    insert_codeswitch, check_entity_survival,
    translate_chunked, load_mt, repair_translated,
    PHONETIC_THRESHOLD, REPAIR_THRESHOLD,
)
from phonetic_correction import build_canonical_entries


BASE = Path(__file__).resolve().parent.parent
ASR_DIR = BASE / "data" / "results" / "asr"
OUT_MT  = BASE / "data" / "results" / "mt_placeholder_ext"
OUT_ASR = BASE / "data" / "results" / "asr_placeholder_ext"
OUT_MT.mkdir(exist_ok=True, parents=True)
OUT_ASR.mkdir(exist_ok=True, parents=True)
EVAL_DIR = BASE / "evaluation"


UK_TO_EN_DICT_EXT = {
    "Рустем Умєров":       "Rustem Umerov",
    "Рустема":             "Rustem",
    "Умєров":              "Umerov",
    "Умєрова":             "Umerov",
    "Кирило Буданов":      "Kyrylo Budanov",
    "Кирилом Будановим":   "Kyrylo Budanov",
    "Буданов":             "Budanov",
    "Будановим":           "Budanov",
    "Давид Арахамія":      "Davyd Arakhamia",
    "Арахамія":            "Arakhamia",
    "Сергій Кислиця":      "Serhiy Kyslytsya",
    "Кислиця":             "Kyslytsya",
    "Скібіцький":          "Skibitskyi",
    "Скібіцького":         "Skibitskyi",
    "Андрій Гнатов":       "Andrii Hnatov",
    "Гнатов":              "Hnatov",
    "Гнатова":             "Hnatov",
    "Хмара":               "Khmara",
    "Хмари":               "Khmara",
    "Євгеній Хмара":       "Yevhenii Khmara",
    "Михайло Федоров":     "Mykhailo Fedorov",
    "Михайла Федорова":    "Mykhailo Fedorov",
    "Федоров":             "Fedorov",
    "Федорова":            "Fedorov",
    "Юлія Свириденко":     "Yuliia Svyrydenko",
    "Юлією Свириденко":    "Yuliia Svyrydenko",
    "Свириденко":          "Svyrydenko",
    "Мар'ян Кушнір":       "Maryan Kushnir",
    "Мар'яну Кушніру":     "Maryan Kushnir",
    "Владислав Ржковський": "Vladyslav Rzhkovskyi",
    "Владиславе":          "Vladyslav",
    "Тата Кеплер":         "Tata Kepler",
    "Татою Кеплер":        "Tata Kepler",
    "Василь Малюк":        "Vasyl Maliuk",
    "Василю Малюку":       "Vasyl Maliuk",
    "Малюку":              "Maliuk",
    "Денис Клименко":      "Denys Klymenko",
    "Денисом Клименком":   "Denys Klymenko",
    "Клименко":            "Klymenko",
    "Клименком":           "Klymenko",
    "Василь Козак":        "Vasyl Kozak",
    "Василем Козаком":     "Vasyl Kozak",
    "Козак":               "Kozak",
    "Олександр Поклад":    "Oleksandr Poklad",
    "Олександром Покладом": "Oleksandr Poklad",
    "Поклад":              "Poklad",
    "Андрій Сибіга":       "Andrii Sybiha",
    "Андрія Сибіги":       "Andrii Sybiha",
    "Сибіга":              "Sybiha",
    "Христя Фріланд":      "Chrystia Freeland",
    "Христя":              "Chrystia",
    "Дмитро Кулеба":       "Dmytro Kuleba",
    "Дмитром Кулебою":     "Dmytro Kuleba",
    "Кулеба":              "Kuleba",
    "Кір Стармер":         "Keir Starmer",
    "Кіром Стармером":     "Keir Starmer",
    "Стармер":             "Starmer",
    "Емманюель":           "Emmanuel",
    "Трамп":               "Trump",
    "Трампа":              "Trump",
    "Трампом":             "Trump",
    "Зеленський":          "Zelensky",
    # Географія України
    "Київ":                "Kyiv",
    "Києві":               "Kyiv",
    "Києва":               "Kyiv",
    "Києву":               "Kyiv",
    "Київщині":            "Kyiv region",
    "Київщина":            "Kyiv region",
    "Харків":              "Kharkiv",
    "Харкові":             "Kharkiv",
    "Харкова":             "Kharkiv",
    "Харківщина":          "Kharkiv region",
    "Харківщині":          "Kharkiv region",
    "Дніпро":              "Dnipro",
    "Дніпрі":              "Dnipro",
    "Дніпра":              "Dnipro",
    "Дніпровщина":         "Dnipropetrovsk region",
    "Дніпровщині":         "Dnipropetrovsk region",
    "Дніпровщини":         "Dnipropetrovsk region",
    "Кривий Ріг":          "Kryvyi Rih",
    "Кривому Розі":        "Kryvyi Rih",
    "Кривого Рогу":        "Kryvyi Rih",
    "Чернігів":            "Chernihiv",
    "Чернігові":           "Chernihiv",
    "Чернігівщина":        "Chernihiv region",
    "Чернігівщині":        "Chernihiv region",
    "Сумщина":             "Sumy region",
    "Сумщині":             "Sumy region",
    "Одеса":               "Odesa",
    "Одесі":               "Odesa",
    "Одещина":             "Odesa region",
    "Херсон":              "Kherson",
    "Херсоні":             "Kherson",
    "Запоріжжя":           "Zaporizhzhia",
    "Запоріжжі":           "Zaporizhzhia",
    "Полтавська":          "Poltava",
    "Полтавській":         "Poltava",
    "Полтавщина":          "Poltava region",
    "Кіровоградщина":      "Kirovohrad region",
    "Кіровоградщині":      "Kirovohrad region",
    "Миколаївщина":        "Mykolaiv region",
    "Миколаївщині":        "Mykolaiv region",
    "Вінницькій":          "Vinnytsia",
    "Чернівецькій":        "Chernivtsi",
    "Привозу":             "Privoz",
    "Привозом":            "Privoz",
    "Покровський":         "Pokrovsk",
    "Покровському":        "Pokrovsk",
    "Краматорський":       "Kramatorsk",
    "Куп'янський":         "Kupiansk",
    "Еміратах":            "Emirates",
    "Емірати":             "Emirates",
    # Країни
    "Франції":             "France",
    "Франція":             "France",
    "Британії":            "Britain",
    "Британія":            "Britain",
    "Норвегії":            "Norway",
    "Норвегія":            "Norway",
    "Австрії":             "Austria",
    "Італії":              "Italy",
    "Італія":              "Italy",
    "Німеччини":           "Germany",
    "Німеччина":           "Germany",
    "Литва":               "Lithuania",
    "Литви":               "Lithuania",
    "Польща":              "Poland",
    "Польщі":              "Poland",
    "Японії":              "Japan",
    "Японія":              "Japan",
    "Європа":              "Europe",
    "Європі":              "Europe",
    "Європою":             "Europe",
    "Європи":              "Europe",
    "Іран":                "Iran",
    "Ірані":               "Iran",
    "Канади":              "Canada",
    "Москва":              "Moscow",
    "Москву":              "Moscow",
    "Венесуели":           "Venezuela",
    "Балтії":              "Baltic states",
    "Євросоюз":            "European Union",
    "Сполучених Штатів Америки": "United States of America",
    "Сполучених Штатів":   "United States",
    # Організації та абревіатури
    "СБУ":                 "SBU",
    "ГУР":                 "HUR",
    "МВС":                 "Ministry of Internal Affairs",
    "ДСНС":                "State Emergency Service",
    "ЗСУ":                 "Armed Forces of Ukraine",
    "Укрзалізниці":        "Ukrzaliznytsia",
    "Укренерго":           "Ukrenergo",
    "Харківобленерго":     "Kharkivoblenergo",
    "Служба безпеки України": "Security Service of Ukraine",
    "Національної гвардії": "National Guard",
    "Національної гвардії України": "National Guard of Ukraine",
    "Радіо Свобода":       "Radio Liberty",
    "Конгресі Сполучених Штатів Америки": "United States Congress",
    "Альфи":               "Alpha",
    "Альфа":               "Alpha",
    # Військові підрозділи
    "окремого штурмового батальйону": "Separate Assault Battalion",
    "бригади морської піхоти":        "Marine Brigade",
    "бригади безпілотних систем":     "Unmanned Systems Brigade",
    "бригади оперативного призначення": "Operational Purpose Brigade",
    "механізованої бригади":          "Mechanized Brigade",
    "окремої механізованої бригади":  "Separate Mechanized Brigade",
    "Червона Калина":                 "Chervona Kalyna",
    "Хартія":                         "Khartiia",
    "Воля й Поле":                    "Will and Field",
    # Зброя, програми
    "Patriot PAC-3":        "Patriot PAC-3",
    "Patriot":              "Patriot",
    "PEARL":                "PEARL",
    "шахеди":               "Shaheds",
    "шахедів":              "Shaheds",
    "РСЗВ":                 "MLRS",
    # Події/місця
    "Давос":                "Davos",
    "Давосу":               "Davos",
    "Омар":                 "Omar",
}


def main():
    tok, mdl, device = load_mt()
    canonicals = build_canonical_entries()
    asr_files = sorted(ASR_DIR.glob("*_seg*.txt"))
    summary_rows = []

    print(f"Розширений словник: {len(UK_TO_EN_DICT_EXT)} записів")
    print(f"ASR файлів: {len(asr_files)}\n")

    for f in asr_files:
        name = f.stem
        asr_text = f.read_text(encoding="utf-8").strip()

        entities = find_entities(asr_text, UK_TO_EN_DICT_EXT, canonicals)

        cs_text, log = insert_codeswitch(asr_text, entities)
        (OUT_ASR / f.name).write_text(cs_text, encoding="utf-8")

        translated_raw = translate_chunked(cs_text, tok, mdl, device)
        n_survived_pre = check_entity_survival(translated_raw, log)

        translated, n_repairs = repair_translated(translated_raw, log)
        n_survived_post = check_entity_survival(translated, log)

        (OUT_MT / f.name).write_text(translated, encoding="utf-8")

        summary_rows.append({
            "segment": name,
            "n_entities_found":       len(entities),
            "n_survived_pre_repair":  n_survived_pre,
            "n_repairs_applied":      n_repairs,
            "n_survived_post_repair": n_survived_post,
            "survival_pre":  round(n_survived_pre  / len(entities), 3) if entities else 1.0,
            "survival_post": round(n_survived_post / len(entities), 3) if entities else 1.0,
        })

        print(f"{name}: found={len(entities)}, pre={n_survived_pre}, "
              f"repairs={n_repairs}, post={n_survived_post}")

    total_entities = sum(r["n_entities_found"]       for r in summary_rows)
    total_pre      = sum(r["n_survived_pre_repair"]  for r in summary_rows)
    total_post     = sum(r["n_survived_post_repair"] for r in summary_rows)
    total_rep      = sum(r["n_repairs_applied"]      for r in summary_rows)
    s_pre  = total_pre  / total_entities if total_entities else 0.0
    s_post = total_post / total_entities if total_entities else 0.0

    print(f"\nTotal entities: {total_entities}")
    print(f"Survived pre-repair:  {total_pre}/{total_entities}  ({s_pre:.1%})")
    print(f"Repairs applied:      {total_rep}")
    print(f"Survived post-repair: {total_post}/{total_entities}  ({s_post:.1%})")

    summary = {
        "method": "entity_codeswitching_pipeline_with_repair_extended_dict",
        "dict_size": len(UK_TO_EN_DICT_EXT),
        "phonetic_threshold": PHONETIC_THRESHOLD,
        "repair_threshold":  REPAIR_THRESHOLD,
        "total_entities":        total_entities,
        "total_survived_pre":    total_pre,
        "total_repairs_applied": total_rep,
        "total_survived_post":   total_post,
        "survival_rate_pre":     round(s_pre, 4),
        "survival_rate_post":    round(s_post, 4),
        "per_segment": summary_rows,
    }
    with open(EVAL_DIR / "placeholder_ext_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
