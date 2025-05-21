import json
import os
from datetime import datetime

# 파일 경로
INPUT_PATH = os.path.join("data/raw", "acasupp.json")
OUTPUT_DIR = os.path.join("data", "acasupport")

# 출력 디렉토리가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JSON 파일 로드
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 게시글의 날짜에서 시간 제거: YYYY.MM.DD만 남기기
def format_date(date_str):
    try:
        return date_str[:10]  # "2024.07.08 11:01:51" → "2024.07.08"
    except:
        return None

# 필드 추출 및 날짜 정리
filtered_data = []
for item in data:
    date_only = format_date(item.get("date", ""))
    filtered_data.append({
        "title": item.get("title"),
        "date": date_only,
        "url": item.get("url")
    })

# 기준 날짜: 마지막 항목의 날짜
if not filtered_data or not filtered_data[-1]["date"]:
    raise ValueError("마지막 게시글의 날짜를 읽을 수 없습니다.")

oldest_date_obj = datetime.strptime(filtered_data[-1]["date"], "%Y.%m.%d")
oldest_date_str = oldest_date_obj.strftime("%y%m%d")  # YYMMDD

# 오늘 날짜
today_str = datetime.today().strftime("%y%m%d")

# 저장 파일 이름
filename = f"{today_str}-{oldest_date_str}.json"
output_path = os.path.join(OUTPUT_DIR, filename)

# JSON 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"✅ 저장 완료: {output_path}")