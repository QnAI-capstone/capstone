import os
import json
import pandas as pd
from glob import glob

# 경로 설정
course_info_path = "data/courseinfo/course_info_subject.csv"
subjectinfo_dir = "data/subject/"
output_dir = "data/subjectinfo/"

# CSV 불러오기
course_df = pd.read_csv(course_info_path)

# 병합용 딕셔너리 생성 (과목번호 → 필요한 정보만)
excluded_keys = {"과목번호", "학과"}
course_info_map = {
    row["과목번호"]: {
        col: row[col] for col in course_df.columns
        if pd.notna(row[col]) and col not in excluded_keys
    }
    for _, row in course_df.iterrows()
}

# 재귀 병합 함수
def merge_subject_info(node):
    if isinstance(node, list):
        for item in node:
            merge_subject_info(item)
    elif isinstance(node, dict):
        if "과목 코드" in node:
            code = node["과목 코드"]
            if code in course_info_map:
                node.update(course_info_map[code])  # 필요한 칼럼만 병합
        for v in node.values():
            merge_subject_info(v)

# output 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 각 JSON 파일 순회 병합
subject_files = glob(os.path.join(subjectinfo_dir, "subject_*.json"))
for file_path in subject_files:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    merge_subject_info(data)

    output_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
