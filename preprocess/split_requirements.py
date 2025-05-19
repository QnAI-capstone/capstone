import json
import os

# 매핑: 단과대 이름 → 파일명 접두사
prefix_map = {
    "서강대학교 전인(全人) 교양 교육과정": "01_jeonin",
    "인문대학": "02_hum",
    "사회과학대학": "03_social",
    "자연과학대학": "04_natural",
    "공과대학": "05_eng",
    "소프트웨어융합대학": "06_sw",
    "경제대학": "07_eco",
    "경영대학": "08_business",
    "지식융합미디어대학": "09_media",
    "연계전공": "10_joint",
    "로욜라국제대학": "11_loyola",
    "마이크로 교육과정": "12_micro",
    "교직과정": "13_teach"
}

# 경로
input_path = "data/전공이수요건_2025.json"
output_dir = "data/requirement"
os.makedirs(output_dir, exist_ok=True)

# 원본 로드
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

root = data.get("2025 서강대학교 요람", {})

# 저장
for college_name, college_data in root.items():
    prefix = prefix_map.get(college_name)
    if not prefix:
        print(f"⚠️ 매핑되지 않은 단과대: {college_name}")
        continue

    filename = f"2025_req_{prefix}.json"
    filepath = os.path.join(output_dir, filename)

    # 전체 구조 유지: "2025 서강대학교 요람": { 단과대학: 내용 }
    wrapped = {
        "2025 서강대학교 요람": {
            college_name: college_data
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(wrapped, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {filepath}")
