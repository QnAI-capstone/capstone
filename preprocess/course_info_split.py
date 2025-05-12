import pandas as pd
from pathlib import Path

# 파일 경로 설정
data_dir = Path("./data/courseinfo")
input_file = data_dir / "course_info_all.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# 과목정보 칼럼 선택
subject_cols = [
    "학과", "과목번호", "과목명", "학점", "권장학년",
    "수강신청 참조사항", "과목 설명"
]

# 수업정보 칼럼 선택
class_cols = [
    "과목번호", "과목명", "학년도", "학기", "분반",
    "강의계획서", "요일", "시작시간", "종료시간", "강의실", "시간", "교수진",
    "수강생수", "영어강의", "중국어강의", "승인과목",
    "CU과목", "홀짝구분", "국제학생", "수강대상", "비고"
]

# ---------------------------
# 과목정보 처리 (중복 + 병합)
# ---------------------------

# 병합 대상 칼럼
merge_cols = ["과목명", "수강신청 참조사항", "과목 설명"]

# 기준 칼럼 (중복 제거 기준)
subject_key_cols = [col for col in subject_cols if col not in merge_cols]

# 안전하게 NaN, 공백 제거 후 문자열 병합하는 함수
def safe_join(values):
    clean_values = set(str(v).strip() for v in values if pd.notna(v) and str(v).strip() != "")
    return ', '.join(sorted(clean_values))

# 권장학년이 빈 값인 행 필터링
df = df[df['권장학년'].apply(lambda x: len(eval(x)) > 0)]  # 빈 리스트 제외

# ---------------------------
# 학점이나 권장학년 값이 다른 경우 최신 학기만 남기기
# ---------------------------

# 학점이나 권장학년 값이 다른 경우만 필터링
df_diff = df.groupby("과목번호").filter(lambda group: len(group["학점"].unique()) > 1 or len(group["권장학년"].unique()) > 1)

# 해당 과목번호가 중복된 학기에서 최신 학기만 남기기
df_diff = df_diff.sort_values(by=["학년도", "학기"], ascending=[False, False]).drop_duplicates(subset=["과목번호"], keep="first")

# ---------------------------
# 학점과 권장학년 값이 같은 경우는 모두 남기고, 최신 학기만 남기기
# ---------------------------

# 학점과 권장학년 값이 같은 과목번호만 필터링
df_same = df.groupby("과목번호").filter(lambda group: len(group["학점"].unique()) == 1 and len(group["권장학년"].unique()) == 1)

# 해당 과목번호에서 최신 학기만 남기기
df_same = df_same.sort_values(by=["학년도", "학기"], ascending=[False, False]).drop_duplicates(subset=["과목번호"], keep="first")

# ---------------------------
# 두 DataFrame 합치기
# ---------------------------
final_df = pd.concat([df_diff, df_same], ignore_index=True)

# ---------------------------
# 과목명 중복 처리 (최신 학기 과목명 남기고, 이전 학기 과목명은 과목 설명에 추가)
# ---------------------------
def update_subject_name_and_description(group):
    # 최신 학기의 과목명
    latest_subject_name = group.iloc[0]["과목명"]
    
    # 과목명에 중복이 있는 경우 이전 학기의 과목명 추가
    previous_subject_names = group["과목명"].iloc[1:].tolist()
    if previous_subject_names:
        previous_subject_names_str = ', '.join(previous_subject_names)
        current_description = group.iloc[0]["과목 설명"]
        # 과목 설명에 '구) 과목명' 추가
        if pd.isna(current_description):
            new_description = f"구) {previous_subject_names_str}"
        else:
            new_description = f"{current_description}, 구) {previous_subject_names_str}"
        
        # 최신 학기의 과목명은 그대로 두고, 이전 학기의 과목명은 설명에 추가
        group.iloc[0]["과목 설명"] = new_description

    return group.iloc[0]  # 가장 최신 학기만 반환

# 과목번호 기준으로 그룹화하고, 중복된 과목명을 처리한 후 과목 정보 병합
subject_df = final_df.groupby("과목번호").apply(update_subject_name_and_description).reset_index(drop=True)

# 필요한 칼럼만 선택
required_columns = [
    "학과", "과목번호", "과목명", "학점", 
    "권장학년", "수강신청 참조사항", "과목 설명"
]

# 필터링 후 파일 저장
subject_df[required_columns].to_csv(data_dir / "course_info_subject.csv", index=False, encoding='utf-8-sig')
print(f"Saved subject info to {data_dir / 'course_info_subject.csv'}")

# ---------------------------
# 수업정보는 그대로 저장 (중복 제거 후)
# ---------------------------
class_df = final_df[class_cols]
class_df.to_csv(data_dir / "course_info_class.csv", index=False, encoding='utf-8-sig')
print(f"Saved class info to {data_dir / 'course_info_class.csv'}")
