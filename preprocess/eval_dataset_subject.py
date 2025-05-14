import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "eval_dataset_subject.csv"

with open(DATA_DIR / "과목정보.json", encoding="utf-8") as f:
    subject_info = json.load(f)

rows = []

for college, college_data in subject_info["2025 서강대학교 요람"].items():
    for category, category_data in college_data.items():
        if not isinstance(category_data, dict):
            continue
        if "과목정보" not in category_data:
            continue

        subject_info_raw = category_data["과목정보"]

        if isinstance(subject_info_raw, dict):
            for group_name, subject_list in subject_info_raw.items():
                # dict 아닌 애들(비고 등) 무시하고 dict만 필터링
                subject_list = [sub for sub in subject_list if isinstance(sub, dict)]

                # 그룹 전체 과목 목록 질문 생성
                subject_list_for_group = [
                    f"{sub.get('과목코드')}({sub.get('과목명')})"
                    for sub in subject_list
                    if sub.get('과목코드') and sub.get('과목명')  # ⬅ None 걸러내기
                ]
                
                if subject_list_for_group:
                    subject_summary = ", ".join(subject_list_for_group)
                    rows.append({
                        "의도": "과목정보 - 트랙 전체 목록",
                        "범위": f"{college} {category}",
                        "질문": f"{category}의 {group_name} 과목들을 모두 알려줘.",
                        "답변": f"{category}의 {group_name} 과목 목록은 다음과 같습니다: {subject_summary}",
                        "답변 근거": f"과목정보.json > {college} > {category} > 과목정보 > {group_name}"
                    })

                # 개별 과목 질문 생성
                for subject in subject_list:
                    code = subject.get("과목코드")
                    name = subject.get("과목명")
                    credits = subject.get("학점")
                    time = subject.get("강의시간")
                    content = subject.get("내용")
                    
                    # ⬅ None 걸러내기
                    if not (code and name and credits and content):
                        continue

                    rows.append({
                        "의도": "과목정보 - 기본 정보",
                        "범위": f"{college} {category}",
                        "질문": f"{name}에 대해서 설명해줘.",
                        "답변": f"{name}({code})은 {credits} 과목으로 {group_name} 과목이며, 주요 내용은 다음과 같습니다. {content}",
                        "답변 근거": f"과목정보.json > {college} > {category} > 과목정보 > {group_name} > {code}"
                    })

                    rows.append({
                        "의도": "과목정보 - 학점",
                        "범위": f"{college} {category}",
                        "질문": f"{name}은 몇 학점 과목이야?",
                        "답변": f"{name}({code})은 {credits} 과목입니다.",
                        "답변 근거": f"과목정보.json > {college} > {category} > 과목정보 > {group_name} > {code}"
                    })

                    if (prerequisite := subject.get("선수과목")):
                        rows.append({
                            "의도": "과목정보 - 선수과목",
                            "범위": f"{college} {category}",
                            "질문": f"{name}의 선수과목은 무엇인가요?",
                            "답변": f"{name}의 선수과목은 {prerequisite} 입니다.",
                            "답변 근거": f"과목정보.json > {college} > {category} > 과목정보 > {group_name} > {code} > 선수과목"
                        })

        elif isinstance(subject_info_raw, list):
            group_name = "N/A"

            subject_list_for_group = [
                f"{sub.get('과목코드')}({sub.get('과목명')})"
                for sub in subject_info_raw if isinstance(sub, dict)
            ]

            if subject_list_for_group:
                subject_summary = ", ".join(subject_list_for_group)
                rows.append({
                    "의도": "과목정보 - 트랙 전체 목록",
                    "범위": f"{college} {category}",
                    "질문": f"{category}의 과목들을 모두 알려줘.",
                    "답변": f"{category} 과목 목록은 다음과 같습니다: {subject_summary}",
                    "답변 근거": f"과목정보.json > {college} > {category} > 과목정보"
                })

            for subject in subject_info_raw:
                if not isinstance(subject, dict):
                    continue

                code = subject.get("과목코드")
                name = subject.get("과목명")
                credits = subject.get("학점")
                time = subject.get("강의시간")
                content = subject.get("내용")
                
                # ⬅ None 걸러내기
                if not (code and name and credits and content):
                    continue

                rows.append({
                    "의도": "과목정보 - 기본 정보",
                    "범위": f"{college} {category}",
                    "질문": f"{name}에 대해서 설명해줘.",
                    "답변": f"{name}({code})은 {credits} 과목으로 주요 내용은 다음과 같습니다. {content}",
                    "답변 근거": f"과목정보.json > {college} > {category} > 과목정보 > {code}"
                })

                rows.append({
                    "의도": "과목정보 - 학점",
                    "범위": f"{college} {category}",
                    "질문": f"{name}은 몇 학점 과목이야?",
                    "답변": f"{name}({code})은 {credits} 과목입니다.",
                    "답변 근거": f"과목정보.json > {college} > {category} > 과목정보 > {code}"
                })

                if (prerequisite := subject.get("선수과목")):
                    rows.append({
                        "의도": "과목정보 - 선수과목",
                        "범위": f"{college} {category}",
                        "질문": f"{name}의 선수과목은 무엇인가요?",
                        "답변": f"{name}의 선수과목은 {prerequisite} 입니다.",
                        "답변 근거": f"과목정보.json > {college} > {category} > 과목정보 > {code} > 선수과목"
                    })

# CSV 저장
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 과목정보 평가 데이터셋 생성 완료: {OUTPUT_FILE}")
