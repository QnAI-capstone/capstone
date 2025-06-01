import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "eval_dataset_subject.csv"

rows = []

# subjectinfo 디렉토리 내의 모든 json 파일 순회
subject_info_dir = DATA_DIR / "subjectinfo"
for json_file_path in subject_info_dir.glob("*.json"):
    with open(json_file_path, encoding="utf-8") as f:
        subject_info = json.load(f)

    # 필드명 매핑 (한글 우선, 없으면 영어)
    FIELD_MAP = {
        "과목 코드": ["과목 코드", "Course Code"],
        "과목명": ["과목명", "Course Name"],
        "학점": ["학점", "Credits"],
        "강의 시간": ["강의 시간", "Lecture"],
        "내용": ["내용", "Description"],
        "권장학년": ["권장학년", "Recommended Year"],
        "수강신청 참조사항": ["수강신청 참조사항", "Application Notes"],
        "과목 설명": ["과목 설명", "Course Description", "Description"],
        "선수과목": ["선수과목", "Prerequisite"]
    }

    # "과목 정보" 키도 유연하게 처리 (과목 정보 또는 과목정보)
    subject_info_key_candidates = ["과목 정보", "과목정보"]

    # 필드 값을 가져오는 헬퍼 함수
    def get_field_value(subject_dict, field_name_candidates):
        for candidate in field_name_candidates:
            if candidate in subject_dict:
                return subject_dict[candidate]
        return None

    # 권장학년별 과목을 저장할 임시 딕셔너리
    # 구조: {college: {category: {year: [subject1_name(code), subject2_name(code)]}}}
    recommended_subjects_by_year = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for college, college_data in subject_info["2025 서강대학교 요람"].items():
        for category, category_data in college_data.items():
            if not isinstance(category_data, dict):
                continue
            
            subject_info_raw = None
            for key_candidate in subject_info_key_candidates:
                if key_candidate in category_data:
                    subject_info_raw = category_data[key_candidate]
                    break
            
            if subject_info_raw is None:
                continue

            if isinstance(subject_info_raw, dict):
                for group_name, subject_list in subject_info_raw.items():
                    subject_list = [sub for sub in subject_list if isinstance(sub, dict)]

                    # 그룹 전체 과목 목록 질문 생성 (기존 로직)
                    subject_list_for_group = []
                    for sub in subject_list:
                        code = get_field_value(sub, FIELD_MAP["과목 코드"])
                        name = get_field_value(sub, FIELD_MAP["과목명"])
                        if code and name:
                            subject_list_for_group.append(f"{code}({name})")
                            
                            # 권장학년별 과목 취합 로직 추가
                            recommended_year_raw = get_field_value(sub, FIELD_MAP["권장학년"])
                            if recommended_year_raw:
                                # 권장학년이 '[1, 2]' 형태일 수 있으므로 파싱
                                if isinstance(recommended_year_raw, list):
                                    years = recommended_year_raw
                                elif isinstance(recommended_year_raw, str):
                                    try:
                                        # "[1, 2]" 형태의 문자열을 리스트로 파싱
                                        years = json.loads(recommended_year_raw.replace("'", '"'))
                                    except json.JSONDecodeError:
                                        # 단일 숫자 문자열일 경우
                                        years = [int(recommended_year_raw)]
                                else:
                                    years = [recommended_year_raw] # 단일 숫자일 경우
                                
                                for year in years:
                                    if year in [1, 2, 3, 4]: # 유효 학년 범위 내에서만 처리
                                        recommended_subjects_by_year[college][category][year].append(f"{name}({code})")

                    if subject_list_for_group:
                        subject_summary = ", ".join(subject_list_for_group)
                        rows.append({
                            "의도": "과목 정보 - 트랙 전체 목록",
                            "범위": f"{college} {category}",
                            "질문": f"{college}의 {category}의 {group_name} 과목들을 모두 알려줘.",
                            "답변": f"{college}의 {category}의 {group_name} 과목 목록은 다음과 같습니다: {subject_summary}",
                            "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {group_name}"
                        })

                    # 개별 과목 질문 생성 (기존 로직)
                    for subject in subject_list:
                        code = get_field_value(subject, FIELD_MAP["과목 코드"])
                        name = get_field_value(subject, FIELD_MAP["과목명"])
                        credits = get_field_value(subject, FIELD_MAP["학점"])
                        lecture_time = get_field_value(subject, FIELD_MAP["강의 시간"])
                        content = get_field_value(subject, FIELD_MAP["내용"])
                        recommended_year = get_field_value(subject, FIELD_MAP["권장학년"])
                        application_notes = get_field_value(subject, FIELD_MAP["수강신청 참조사항"])
                        course_description = get_field_value(subject, FIELD_MAP["과목 설명"]) 
                        prerequisite = get_field_value(subject, FIELD_MAP["선수과목"])

                        if not (code and name and credits and content):
                            continue

                        base_answer_parts = [
                            f"{college}의 {category}의 {name}({code})은 {credits} 과목으로 {group_name} 과목이며, 주요 내용은 다음과 같습니다. {content}"
                        ]
                        if recommended_year:
                            base_answer_parts.append(f" 권장 학년은 {recommended_year}입니다.")
                        if lecture_time:
                            base_answer_parts.append(f" 강의 시간은 {lecture_time}입니다.")
                        if course_description and course_description != content: 
                            base_answer_parts.append(f" 과목 설명: {course_description}")
                        
                        base_answer = "".join(base_answer_parts)

                        rows.append({
                            "의도": "과목 정보 - 기본 정보",
                            "범위": f"{college} {category}",
                            "질문": f"{college}의 {category}의 {name}에 대해서 설명해줘.",
                            "답변": base_answer,
                            "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {group_name} > {code}"
                        })

                        rows.append({
                            "의도": "과목 정보 - 학점",
                            "범위": f"{college} {category}",
                            "질문": f"{college}의 {category}의 {name}은 몇 학점 과목이야?",
                            "답변": f"{college}의 {category}의 {name}({code})은 {credits} 과목입니다.",
                            "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {group_name} > {code}"
                        })

                        if prerequisite:
                            rows.append({
                                "의도": "과목 정보 - 선수과목",
                                "범위": f"{college} {category}",
                                "질문": f"{college}의 {category}의 {name}의 선수과목은 뭐야?",
                                "답변": f"{college}의 {category}의 {name}의 선수과목은 {prerequisite} 입니다.",
                                "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {group_name} > {code} > 선수과목"
                            })
                        
                        if recommended_year:
                            rows.append({
                                "의도": "과목 정보 - 권장학년",
                                "범위": f"{college} {category}",
                                "질문": f"{college}의 {category}의 {name}의 권장학년을 알려줘.",
                                "답변": f"{college}의 {category}의 {name}의 권장학년은 {recommended_year} 입니다.",
                                "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {group_name} > {code} > 권장학년"
                            })

                        if application_notes:
                            rows.append({
                                "의도": "과목 정보 - 수강신청 참조사항",
                                "범위": f"{college} {category}",
                                "질문": f"{college}의 {category}의 {name}의 수강신청 시 참조할 사항이 있어?",
                                "답변": f"{college}의 {category}의 {name}의 수강신청 참조사항은 {application_notes} 입니다.",
                                "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {group_name} > {code} > 수강신청 참조사항"
                            })

                        if course_description:
                            rows.append({
                                "의도": "과목 정보 - 과목 설명",
                                "범위": f"{college} {category}",
                                "질문": f"{college}의 {category}의 {name} 과목의 자세한 설명을 해줘.",
                                "답변": f"{college}의 {category}의 {name} 과목 설명은 다음과 같습니다: {course_description}",
                                "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {group_name} > {code} > 과목 설명"
                            })


            elif isinstance(subject_info_raw, list):
                group_name = "N/A" 

                subject_list_for_group = []
                for sub in subject_info_raw:
                    if isinstance(sub, dict):
                        code = get_field_value(sub, FIELD_MAP["과목 코드"])
                        name = get_field_value(sub, FIELD_MAP["과목명"])
                        if code and name:
                            subject_list_for_group.append(f"{code}({name})")

                            # 권장학년별 과목 취합 로직 추가 (리스트 형태)
                            recommended_year_raw = get_field_value(sub, FIELD_MAP["권장학년"])
                            if recommended_year_raw:
                                if isinstance(recommended_year_raw, list):
                                    years = recommended_year_raw
                                elif isinstance(recommended_year_raw, str):
                                    try:
                                        years = json.loads(recommended_year_raw.replace("'", '"'))
                                    except json.JSONDecodeError:
                                        years = [int(recommended_year_raw)]
                                else:
                                    years = [recommended_year_raw]
                                
                                for year in years:
                                    if year in [1, 2, 3, 4]:
                                        recommended_subjects_by_year[college][category][year].append(f"{name}({code})")

                if subject_list_for_group:
                    subject_summary = ", ".join(subject_list_for_group)
                    rows.append({
                        "의도": "과목 정보 - 트랙 전체 목록",
                        "범위": f"{college} {category}",
                        "질문": f"{college}의 {category}의 {category}의 과목들을 모두 알려줘.",
                        "답변": f"{college}의 {category}의 {category} 과목 목록은 다음과 같습니다: {subject_summary}",
                        "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보"
                    })

                for subject in subject_info_raw:
                    if not isinstance(subject, dict):
                        continue

                    code = get_field_value(subject, FIELD_MAP["과목 코드"])
                    name = get_field_value(subject, FIELD_MAP["과목명"])
                    credits = get_field_value(subject, FIELD_MAP["학점"])
                    lecture_time = get_field_value(subject, FIELD_MAP["강의 시간"])
                    content = get_field_value(subject, FIELD_MAP["내용"])
                    recommended_year = get_field_value(subject, FIELD_MAP["권장학년"])
                    application_notes = get_field_value(subject, FIELD_MAP["수강신청 참조사항"])
                    course_description = get_field_value(subject, FIELD_MAP["과목 설명"])
                    prerequisite = get_field_value(subject, FIELD_MAP["선수과목"])
                    
                    if not (code and name and credits and content):
                        continue

                    base_answer_parts = [
                        f"{college}의 {category}의 {name}({code})은 {credits} 과목으로 주요 내용은 다음과 같습니다. {content}"
                    ]
                    if recommended_year:
                        base_answer_parts.append(f" 권장 학년은 {recommended_year}입니다.")
                    if lecture_time:
                        base_answer_parts.append(f" 강의 시간은 {lecture_time}입니다.")
                    if course_description and course_description != content:
                        base_answer_parts.append(f" 과목 설명: {course_description}")
                    
                    base_answer = "".join(base_answer_parts)

                    rows.append({
                        "의도": "과목 정보 - 기본 정보",
                        "범위": f"{college} {category}",
                        "질문": f"{college}의 {category}의 {name}에 대해서 설명해줘.",
                        "답변": base_answer,
                        "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {code}"
                    })

                    rows.append({
                        "의도": "과목 정보 - 학점",
                        "범위": f"{college} {category}",
                        "질문": f"{college}의 {category}의 {name}은 몇 학점 과목이야?",
                        "답변": f"{college}의 {category}의 {name}({code})은 {credits} 과목입니다.",
                        "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {code}"
                    })

                    if prerequisite:
                        rows.append({
                            "의도": "과목 정보 - 선수과목",
                            "범위": f"{college} {category}",
                            "질문": f"{college}의 {category}의 {name}의 선수과목은 뭐야?",
                            "답변": f"{college}의 {category}의 {name}의 선수과목은 {prerequisite} 입니다.",
                            "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {code} > 선수과목"
                        })
                    
                    if recommended_year:
                        rows.append({
                            "의도": "과목 정보 - 권장학년",
                            "범위": f"{college} {category}",
                            "질문": f"{college}의 {category}의 {name}의 권장학년을 알려줘.",
                            "답변": f"{college}의 {category}의 {name}의 권장학년은 {recommended_year} 입니다.",
                            "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {code} > 권장학년"
                        })

                    if application_notes:
                        rows.append({
                            "의도": "과목 정보 - 수강신청 참조사항",
                            "범위": f"{college} {category}",
                            "질문": f"{college}의 {category}의 {name}의 수강신청 시 참조할 사항이 있어?",
                            "답변": f"{college}의 {category}의 {name}의 수강신청 참조사항은 {application_notes} 입니다.",
                            "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {code} > 수강신청 참조사항"
                        })
                    
                    if course_description:
                        rows.append({
                            "의도": "과목 정보 - 과목 설명",
                            "범위": f"{college} {category}",
                            "질문": f"{college}의 {category}의 {name} 과목의 자세한 설명을 해줘.",
                            "답변": f"{college}의 {category}의 {name} 과목 설명은 다음과 같습니다: {course_description}",
                            "답변 근거": f"{json_file_path.name} > {college} > {category} > 과목 정보 > {code} > 과목 설명"
                        })

    # 모든 JSON 파일 처리가 끝난 후, 권장학년별 질문-답변 쌍 생성
    for college, college_data in recommended_subjects_by_year.items():
        for category, year_data in college_data.items():
            for year, subjects_list in year_data.items():
                if subjects_list: # 권장 과목이 있는 경우에만 생성
                    subject_summary = ", ".join(sorted(list(set(subjects_list)))) # 중복 제거 및 정렬
                    rows.append({
                        "의도": "과목 정보 - 학년별 권장 과목",
                        "범위": f"{college} {category}",
                        "질문": f"{college}의 {category}의 {year}학년에게 권장되는 과목을 모두 알려줘.",
                        "답변": f"{college}의 {category}의 {year}학년에게 권장되는 과목은 다음과 같습니다: {subject_summary}",
                        "답변 근거": f"모든 JSON 파일에서 {college} > {category} > 과목 정보 > {year}학년" # 여러 파일에서 취합될 수 있으므로 일반화
                    })


# CSV 저장
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 과목 정보 평가 데이터셋 생성 완료: {OUTPUT_FILE}")