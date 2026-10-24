#!/usr/bin/env python3
"""
Скрипт постобработки и фильтрации распознанных паспортов РФ.
Запуск:
    python postprocess.py <input_predictions.jsonl> <output_answers.jsonl>
"""
import os
import sys
import json
import csv
from datetime import datetime, timedelta

FIELDS = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru"
]

RU_TO_MRZ = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2",
    "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "1",
    "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8", " ": "<", "-": "<"
}

WEIGHTS = [7, 3, 1]

def ru_to_mrz(text):
    return "".join(RU_TO_MRZ.get(c, "") for c in str(text).upper())

def norm(s):
    return " ".join(str(s or "").split()).upper()

def add_years(d, years):
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)

def load_vocab():
    vocab = {
        "ОТДЕЛ", "ОТДЕЛЕНИЕМ", "ОТДЕЛОМ", "УПРАВЛЕНИЕМ", "УПРАВЛЕНИЕ",
        "ГЛАВНЫМ", "ГУ", "МВД", "УМВД", "ОМВД", "УФМС", "ОФМС",
        "ПО", "ВОПРОСАМ", "МИГРАЦИИ", "РОССИИ", "В", "И", "РАЙОНЕ", "РАЙОНА",
        "Г.", "ГОРОДЕ", "ГОРОДА", "ГОР.", "ОБЛАСТИ", "РЕСПУБЛИКЕ", "КРАЕ", "ОКРУГЕ",
        "АВТОНОМНОМ", "АО", "ОБЛ.", "Р-НЕ", "П.", "ПОСЕЛКЕ", "С.", "СЕЛЕ", "Д.", "ДЕРЕВНЕ",
        "ТП", "ОВД", "РОВД", "УВД", "ОМ", "МП", "ПУНКТ", "ПУНКТОМ", "ТЕРРИТОРИАЛЬНЫЙ",
        "ТЕРРИТОРИАЛЬНЫМ", "МЕЖРАЙОННЫЙ", "МЕЖРАЙОННЫМ"
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "reference", "subdivisions.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                words = row["name"].upper().replace(",", " ").replace("\"", " ").split()
                for w in words:
                    cleaned = w.strip(".,;:() ")
                    if cleaned:
                        vocab.add(cleaned)
    return vocab

VALID_VOCAB = load_vocab()

def is_passport_valid(birth_str, issue_str, check_date=datetime(2026, 10, 1)):
    try:
        b_date = datetime.strptime(str(birth_str).strip(), "%d.%m.%Y")
        i_date = datetime.strptime(str(issue_str).strip(), "%d.%m.%Y")
    except Exception:
        return False
    if i_date > check_date or i_date < add_years(b_date, 14):
        return False
    deadline_20 = add_years(b_date, 20) + timedelta(days=90)
    deadline_45 = add_years(b_date, 45) + timedelta(days=90)
    if check_date >= deadline_45 and i_date < add_years(b_date, 45):
        return False
    if check_date >= deadline_20 and i_date < add_years(b_date, 20):
        return False
    return True

def calculate_check_digit(text):
    total = 0
    for i, char in enumerate(text):
        val = int(char) if char.isdigit() else (ord(char.upper()) - ord("A") + 10 if char.isalpha() else 0)
        total += val * WEIGHTS[i % 3]
    return str(total % 10)

def parse_mrz_date(s, is_birth=True):
    if len(s) != 6 or not s.isdigit():
        return None
    yy = int(s[0:2])
    year = (2000 + yy if yy <= 26 else 1900 + yy) if is_birth else (2000 + yy if yy <= 30 else 1900 + yy)
    return f"{s[4:6]}.{s[2:4]}.{year}"

def parse_mrz_line2(line2):
    if not line2 or len(line2) < 44:
        return None
    data = {}
    if (calculate_check_digit(line2[0:9]) == line2[9] and 
            line2[0:3].isdigit() and line2[28].isdigit() and line2[3:9].isdigit()):
        data["Licence_number"] = f"{line2[0:2]} {line2[2]}{line2[28]} {line2[3:9]}"
    if calculate_check_digit(line2[13:19]) == line2[19]:
        data["Birth_date"] = parse_mrz_date(line2[13:19], is_birth=True)
    if line2[20] in ("M", "F"):
        data["Sex_ru"] = "МУЖ." if line2[20] == "M" else "ЖЕН."
    if line2[29:35].isdigit():
        data["Issue_date"] = parse_mrz_date(line2[29:35], is_birth=False)
    if line2[35:41].isdigit():
        data["Issue_organisation_code"] = f"{line2[35:38]}-{line2[38:41]}"
    return data

def process_document(doc, min_conf_th=0.92):
    doc_id = doc.get("doc_id")
    quality = doc.get("quality") or {}
    if quality.get("glare", 0) > 0.35:
        return {"doc_id": doc_id, "decision": "reject"}

    viz = doc.get("viz") or {}
    fields = {}
    min_char_conf = 1.0

    for fld in FIELDS:
        f_data = viz.get(fld)
        if not f_data or not f_data.get("lines"):
            return {"doc_id": doc_id, "decision": "reject"}
        lines_text = []
        for l in f_data["lines"]:
            lines_text.append(l.get("text", "").strip())
            for c_conf in l.get("char_conf", []):
                if c_conf < min_char_conf:
                    min_char_conf = c_conf
        fields[fld] = " ".join(lines_text)

    if min_char_conf < min_conf_th:
        return {"doc_id": doc_id, "decision": "reject"}

    mrz = doc.get("mrz")
    if not mrz or not mrz.get("lines") or len(mrz["lines"]) < 2:
        return {"doc_id": doc_id, "decision": "reject"}

    mrz_data = parse_mrz_line2(mrz["lines"][1]["text"])
    if not mrz_data or len(mrz_data) != 5:
        return {"doc_id": doc_id, "decision": "reject"}

    # Сверка кода подразделения и даты выдачи между VIZ и MRZ
    viz_code = norm(fields["Issue_organisation_code"]).replace(" ", "")
    if len(viz_code) == 6 and viz_code.isdigit():
        viz_code = f"{viz_code[:3]}-{viz_code[3:]}"
    if viz_code != mrz_data["Issue_organisation_code"]:
        return {"doc_id": doc_id, "decision": "reject"}
    if norm(fields["Issue_date"]) != mrz_data["Issue_date"]:
        return {"doc_id": doc_id, "decision": "reject"}

    # Применяем эталонные значения из MRZ для структурированных полей
    fields.update(mrz_data)

    # Кросс-валидация ФИО через строку MRZ 1
    l1 = mrz["lines"][0].get("text", "")
    last_mrz = ru_to_mrz(fields["Last_name_ru"])
    first_mrz = ru_to_mrz(fields["First_name_ru"])
    mid_mrz = ru_to_mrz(fields.get("Middle_name_ru", ""))
    expected_l1 = ("PNRUS" + last_mrz + "<<" + first_mrz + "<" + mid_mrz).rstrip("<")
    clean_l1 = l1.rstrip("<")
    if not (clean_l1 == expected_l1 or clean_l1.startswith(expected_l1)):
        return {"doc_id": doc_id, "decision": "reject"}

    # Проверка действительности паспорта на 01.10.2026
    if not is_passport_valid(fields.get("Birth_date"), fields.get("Issue_date")):
        return {"doc_id": doc_id, "decision": "reject"}

    # Проверка на потерянные строки в многострочных полях
    org = fields["Issue_organization_ru"].strip()
    if any(org.endswith(" " + prep) for prep in ["В", "ПО", "НА", "И", "ОТ", "ПРИ", "Р-НЕ", "Г."]):
        return {"doc_id": doc_id, "decision": "reject"}
    if any(org.endswith(" " + adj) for adj in ["ТЮМЕНСКОЙ", "КАБАРДИНО-БАЛКАРСКОЙ", "РЕСПУБЛИКЕ"]):
        return {"doc_id": doc_id, "decision": "reject"}
    if any(org.startswith(pref + " ") for pref in ["Р-НА", "ОБЛ.", "КРАЯ", "РЕСП.", "РОССИИ ПО"]):
        return {"doc_id": doc_id, "decision": "reject"}
    if "ОМВД " in org and "ОМВД РОССИИ" not in org and not org.endswith("ОМВД"):
        return {"doc_id": doc_id, "decision": "reject"}

    # Проверка предлогов после слова РОССИИ
    org_words = org.split()
    for i, w in enumerate(org_words):
        if w == "РОССИИ" and i + 1 < len(org_words):
            if org_words[i+1] not in ("ПО", "В", "ПРИ", "И", "НА", "ОТ", "ПОО", "ЯМАЛО-НЕНЕЦКОМУ"):
                return {"doc_id": doc_id, "decision": "reject"}

    # Нормализация точки в сокращении ОБЛ. в месте рождения
    bp = fields["Birth_place_ru"].strip()
    if bp.endswith(" ОБЛ"):
        bp = bp + "."
        fields["Birth_place_ru"] = bp

    if any(bp.endswith(" " + prep) for prep in ["В", "ПО", "НА", "И", "ОТ"]):
        return {"doc_id": doc_id, "decision": "reject"}
    if any(bp.startswith(pref + " ") for pref in ["Р-НА", "ОБЛ.", "КРАЯ", "РЕСП."]):
        return {"doc_id": doc_id, "decision": "reject"}
    if bp.endswith(" ОБЛ.") and not any(p in bp for p in ["Г.", "ГОР.", "С.", "Д.", "ДЕР.", "ПОС.", "П.", "Р-Н", "Р-НА", "РАЙОН"]):
        return {"doc_id": doc_id, "decision": "reject"}
    if bp.endswith(("ОГО", "ЕГО", "СКОГО", "ЦКОГО")):
        return {"doc_id": doc_id, "decision": "reject"}
    if bp.endswith(("СКИЙ", "ЦКИЙ")) and not any(bp.startswith(p + " ") for p in ["ХУТ.", "ГОР.", "Г.", "ПОС."]):
        return {"doc_id": doc_id, "decision": "reject"}
    if len(bp) < 16 and any(bp.startswith(p + " ") for p in ["С.", "П.", "Д."]) and not any(r in bp for r in ["ОБЛ", "КР", "РЕСП", "Р-Н", "РАЙОН"]):
        return {"doc_id": doc_id, "decision": "reject"}

    # Словарная проверка органа выдачи
    org_words = org.split()
    for w in org_words:
        clean_w = w.strip(".,;:()№\" ").upper()
        if clean_w and not clean_w.isdigit() and clean_w not in VALID_VOCAB and w not in VALID_VOCAB:
            return {"doc_id": doc_id, "decision": "reject"}

    return {
        "doc_id": doc_id,
        "decision": "accept",
        "fields": fields
    }

def main():
    if len(sys.argv) < 3:
        print(f"Использование: python {sys.argv[0]} <input_predictions.jsonl> <output_answers.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    total = 0
    accepted = 0

    with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            doc = json.loads(line)
            res = process_document(doc)
            out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            total += 1
            if res["decision"] == "accept":
                accepted += 1

    print(f"Обработано документов: {total}")
    print(f"Принято: {accepted} ({accepted / total:.1%})")
    print(f"Отклонено: {total - accepted} ({(total - accepted) / total:.1%})")

if __name__ == "__main__":
    main()
