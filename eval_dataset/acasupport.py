import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import re

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "eval_dataset_acasupport.csv"

rows = []

# acasupport JSON 파일 경로를 수정했습니다.
acasupport_json_path = DATA_DIR / "acasupport" / "250521-240708.json"

try:
    with open(acasupport_json_path, encoding="utf-8") as f:
        acasupport_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: {acasupport_json_path} not found. Please check the file path.")
    exit()
except json.JSONDecodeError as e:
    print(f"❌ Error decoding JSON from {acasupport_json_path}: {e}")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while reading {acasupport_json_path}: {e}")
    exit()

# 공지사항 데이터를 날짜 기준으로 정렬 (최신순)
# key=lambda x: datetime.strptime(x.get("date", "1900.01.01"), "%Y.%m.%d")는 날짜 형식이 'YYYY.MM.DD'일 때 유효합니다.
# sorted_acasupport_data[0]이 최신, sorted_acasupport_data[-1]이 가장 오래된 것이 되도록 reverse=False로 변경했습니다.
# (이전 코드에서 sorted_acasupport_data[0]이 최신, sorted_acasupport_data[-1]이 가장 오래된 것을 가정하고 reverse=True 였으나,
# '최신순' 정렬 시 sorted_acasupport_data[0]이 최신이 되려면 reverse=True가 맞습니다. 혼란을 드려 죄송합니다.
# 다시 원래대로 sorted_acasupport_data[0]이 최신, sorted_acasupport_data[-1]이 가장 오래된 것이 되도록 reverse=True를 유지합니다.)
sorted_acasupport_data = sorted(
    acasupport_data,
    key=lambda x: datetime.strptime(x.get("date", "1900.01.01"), "%Y.%m.%d"),
    reverse=True # 최신 공지가 리스트의 첫 번째에 오도록 (내림차순)
)

# 헬퍼 함수: 제목에서 불필요한 괄호 안의 영어 및 "[분류]" 제거
def clean_title(title):
    # "[분류]" 형태 제거 (제목 시작 부분)
    cleaned_title = re.sub(r'^\[[^\]]*\]\s*', '', title).strip()
    # (영어만 포함된 괄호) 패턴 제거 (단, "날짜:", "시간:", "링크:" 포함 제외)
    # 한글이 포함되지 않은 괄호만 제거하며, 숫자/특수문자만 있는 괄호도 제거될 수 있도록 수정
    # 예: (ABC), (123) 등. (날짜:...), (시간:...) 등은 유지
    cleaned_title = re.sub(r'\s*\((?![가-힣])(?!날짜:|시간:|링크:)[^)]*\)', '', cleaned_title).strip()
    return cleaned_title

# 헬퍼 함수: 단일 acasupport 항목을 포맷팅하여 답변 생성 (URL 강조)
def format_acasupport_item_for_url(item):
    title = clean_title(item.get("title", "제목 없음")) # 클리닝 함수 적용
    date = item.get("date", "날짜 없음")
    url = item.get("url", "#")
    if url and url != "#":
        return f"'{title}' 공지 (날짜: {date})의 링크는 [{url}] 입니다."
    else:
        return f"'{title}' 공지 (날짜: {date})의 링크를 찾을 수 없습니다."

# 헬퍼 함수: 여러 acasupport 항목을 포맷팅하여 답변 생성 (간단한 목록 후 URL 요청 유도)
def format_acasupport_list_for_overview(items):
    if not items:
        return "해당하는 공지사항이 없습니다."

    formatted_list = []
    for item in items:
        title = clean_title(item.get("title", "제목 없음")) # 클리닝 함수 적용
        date = item.get("date", "날짜 없음")
        formatted_list.append(f"- {title} ({date})")

    overview_text = "\n".join(formatted_list)
    if len(items) > 1:
        overview_text += "\n어떤 공지의 링크를 원하시나요? 제목을 말씀해주세요."
    elif len(items) == 1:
        overview_text += "\n이 공지의 링크를 원하시면 말씀해주세요."
    return overview_text

# 공통 키워드 매핑 (예시)
COMMON_KEYWORDS = {
    "등록금": ["등록금", "학비"],
    "장학금": ["장학", "장학금"],
    "수강신청": ["수강신청", "수강"],
    "계절학기": ["계절학기", "계절"],
    "전공": ["전공"],
    "학칙": ["학칙"],
    "학위": ["학위", "졸업"],
    "1학년": ["1학년", "신입생"],
    "모바일": ["모바일", "앱"],
    "평가": ["평가", "성적"]
}


# --- 모든 게시글에 대해 평가 데이터 생성 반복 ---
for i, item in enumerate(acasupport_data):
    original_title = item.get("title", f"제목 없음 {i}")
    title = clean_title(original_title) # 질문과 답변에 사용될 클리닝된 제목
    date = item.get("date", "날짜 없음")
    url = item.get("url", "#")

    if not (title and date and url and url != "#"): # 유효한 정보가 없으면 스킵
        print(f"⚠️ Warning: Skipping item {i} due to missing title, date or URL: {item}")
        continue

    # 1. 정확한 제목으로 URL 요청
    rows.append({
        "의도": "정확한 제목으로 URL 요청",
        "범위": "학사공지",
        "질문": f"'{title}' 공지 링크 알려줘.",
        "답변": format_acasupport_item_for_url(item),
        "답변 근거": f"{acasupport_json_path.name} (게시글 Index: {i}, 원본 제목: {original_title})"
    })
    rows.append({
        "의도": "정확한 제목으로 URL 요청 (다른 질문)",
        "범위": "학사공지",
        "질문": f"'{title}' 공지 URL 좀 알려줄래?",
        "답변": format_acasupport_item_for_url(item),
        "답변 근거": f"{acasupport_json_path.name} (게시글 Index: {i}, 원본 제목: {original_title})"
    })

    # 2. 제목 키워드 기반 URL 요청 (각 게시글의 제목에 포함된 키워드 활용)
    for category, keywords in COMMON_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in original_title.lower(): # 원본 제목으로 키워드 검색
                rows.append({
                    "의도": f"키워드 기반 URL 요청 - {category} (개별 게시글)",
                    "범위": "학사공지",
                    "질문": f"'{title}' 공지 (제목에 '{keyword}' 포함) 링크 알려줘.",
                    "답변": format_acasupport_item_for_url(item),
                    "답변 근거": f"{acasupport_json_path.name} (게시글 Index: {i}, 키워드: {keyword})"
                })

    # 3. 날짜와 제목을 조합하여 URL 요청 (각 게시글에 대해 생성)
    rows.append({
        "의도": "날짜+제목 조합 URL 요청",
        "범위": "학사공지",
        "질문": f"{date}에 올라온 '{title}' 공지 링크 줘.",
        "답변": format_acasupport_item_for_url(item),
        "답변 근거": f"{acasupport_json_path.name} (게시글 Index: {i}, 날짜: {date}, 원본 제목: {original_title})"
    })
    rows.append({
        "의도": "날짜+제목 조합 URL 요청 (다른 질문)",
        "범위": "학사공지",
        "질문": f"{date}자 '{title}' 공지 URL 알려줘.",
        "답변": format_acasupport_item_for_url(item),
        "답변 근거": f"{acasupport_json_path.name} (게시글 Index: {i}, 날짜: {date}, 원본 제목: {original_title})"
    })


# --- 특정 유형의 공지사항 (최신/가장 오래된)은 전체 데이터셋 기준으로 한 번만 생성 ---
if sorted_acasupport_data:
    # 가장 최신 공지사항
    latest_item = sorted_acasupport_data[0]
    rows.append({
        "의도": "최신 공지사항 URL",
        "범위": "학사공지",
        "질문": "가장 최근 공지사항 링크 뭐야?",
        "답변": format_acasupport_item_for_url(latest_item),
        "답변 근거": f"{acasupport_json_path.name} (최신 게시글)"
    })

    # 오늘 날짜로 올라온 공지 질문
    today_date_str = datetime.now().strftime("%Y.%m.%d")
    if latest_item.get("date") == today_date_str: # 최신 공지가 오늘 날짜인 경우
        rows.append({
            "의도": "오늘 공지사항 URL",
            "범위": "학사공지",
            "질문": "오늘 올라온 공지 링크 있어?",
            "답변": format_acasupport_item_for_url(latest_item),
            "답변 근거": f"{acasupport_json_path.name} (오늘 날짜 최신 게시글)"
        })
    else: # 최신 공지가 오늘 날짜가 아닌 경우 (오늘 공지가 없는 경우)
        rows.append({
            "의도": "오늘 공지사항 URL (없음)",
            "범위": "학사공지",
            "질문": "오늘 올라온 공지 링크 있어?",
            "답변": "오늘 날짜로 올라온 공지사항은 없습니다.",
            "답변 근거": f"{acasupport_json_path.name} (오늘 날짜 공지 없음)"
        })

    # 가장 오래된 공지사항
    oldest_item = sorted_acasupport_data[-1]
    rows.append({
        "의도": "가장 오래된 공지사항 URL",
        "범위": "학사공지",
        "질문": "가장 오래된 공지사항 링크 알려줘.",
        "답변": format_acasupport_item_for_url(oldest_item),
        "답변 근거": f"{acasupport_json_path.name} (가장 오래된 게시글)"
    })


# --- 키워드 기반 전체 목록 요청 ---
for category, keywords in COMMON_KEYWORDS.items():
    for keyword in keywords:
        all_related_items = [item for item in acasupport_data if keyword.lower() in item.get("title", "").lower()]
        if all_related_items:
            rows.append({
                "의도": f"키워드 기반 목록 요청 - {category}",
                "범위": "학사공지",
                "질문": f"{keyword} 관련 공지사항 목록을 보여줘.",
                "답변": format_acasupport_list_for_overview(all_related_items),
                "답변 근거": f"{acasupport_json_path.name} (키워드: {keyword}, 모든 목록)"
            })
        else:
            rows.append({
                "의도": f"키워드 기반 목록 요청 - {category} (없음)",
                "범위": "학사공지",
                "질문": f"{keyword} 관련 공지사항 목록을 보여줘.",
                "답변": f"'{keyword}' 관련 공지사항은 찾을 수 없습니다.",
                "답변 근거": f"{acasupport_json_path.name} (키워드: {keyword}, 목록 없음)"
            })


# CSV 저장
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ acasupport 평가 데이터셋 생성 완료: {OUTPUT_FILE}")