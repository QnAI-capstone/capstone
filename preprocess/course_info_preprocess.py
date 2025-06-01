import pandas as pd
import re
from pathlib import Path

def load_course_info(file_path):
    df = pd.read_csv(file_path, dtype=str).fillna("")
    print(f"Loaded {file_path} with shape {df.shape}")
    return df

def parse_class_time_place(value):
    pattern_date = r'(\d{2}\.\d{2})\((.)\)\s+(\d{2}:\d{2})~(\d{2}:\d{2})\s+\[(.+?)\]'
    match_date = re.match(pattern_date, value)
    if match_date:
        date, weekday, start_time, end_time, room = match_date.groups()
        return pd.Series([date, [weekday], start_time, end_time, room])

    pattern_multi = r'([가-힣,]+)\s+(\d{2}:\d{2})~(\d{2}:\d{2})\s+\[(.+?)\]'
    match_multi = re.match(pattern_multi, value)
    if match_multi:
        weekdays_str, start_time, end_time, room = match_multi.groups()
        weekdays = weekdays_str.split(',')
        return pd.Series(["", weekdays, start_time, end_time, room])

    pattern_single = r'([가-힣])\s+(\d{2}:\d{2})~(\d{2}:\d{2})\s+\[(.+?)\]'
    match_single = re.match(pattern_single, value)
    if match_single:
        weekday, start_time, end_time, room = match_single.groups()
        return pd.Series(["", [weekday], start_time, end_time, room])

    return pd.Series(["", [], "", "", ""])

def normalize_grade_target(value):
    if not value or pd.isna(value):
        return []

    value = value.strip()
    if "전학년" in value:
        return [1, 2, 3, 4]

    value = value.replace("학년", "").replace("~", "-")

    if "," in value:
        try:
            return sorted(list({int(v.strip()) for v in value.split(",") if v.strip().isdigit()}))
        except:
            return []

    range_match = re.match(r'^(\d)-(\d)$', value)
    if range_match:
        start, end = map(int, range_match.groups())
        return list(range(start, end + 1))

    single_match = re.match(r'^(\d+)$', value)
    if single_match:
        return [int(single_match.group(1))]

    return []

def preprocess_course_info(df):
    # 수업시간/강의실 분리
    time_place_cols = df['수업시간/강의실'].apply(parse_class_time_place)
    time_place_cols.columns = ['수업일자', '요일', '시작시간', '종료시간', '강의실']
    df = pd.concat([df, time_place_cols], axis=1)

    # 원본 '수업시간/강의실' 칼럼 삭제
    df = df.drop(columns=['수업시간/강의실'])

    # 권장학년, 수강대상 정규화
    df['권장학년'] = df['권장학년'].apply(normalize_grade_target)
    df['수강대상'] = df['수강대상'].apply(normalize_grade_target)

    return df

def merge_course_files(folder_path):
    csv_files = Path(folder_path).glob("*.csv")
    merged_df = pd.DataFrame()

    for file in csv_files:
        df = load_course_info(file)
        df = preprocess_course_info(df)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    print(f"Merged dataframe shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    data_dir = Path("./data/courseinfo")
    data_dir.mkdir(parents=True, exist_ok=True)

    merged_df = merge_course_files(data_dir)
    merged_df.to_csv(data_dir / "course_info_all.csv", index=False, encoding='utf-8-sig')
    print("Saved merged course info to ./data/courseinfo/course_info_all.csv")
