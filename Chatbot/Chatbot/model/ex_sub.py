from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "all_subject_names_unique.txt")

# 과목명 추출 함수 using rapidfuzz
def extract_subject_by_rapidfuzz(query,top_n=5, score_cutoff=65):
    # 텍스트 파일에서 과목명 로드
    with open(FILE_PATH, encoding="utf-8-sig") as f:
        subject_names = [line.strip() for line in f if line.strip()]
    # 일단 score_cutoff 없이 top_n 추출
    matches = process.extract(
        query,
        subject_names,
        scorer=fuzz.partial_ratio,
        limit=top_n
    )
    
    # 결과 출력 (점수 포함)
    for i, (match, score, _) in enumerate(matches):
        print(f"[{i}] {match} | 유사도 점수: {score}")

    # 실제 반환은 과목명만, score_cutoff 이상만 필터링
    return [match for match, score, _ in matches if score >= score_cutoff]

# ✅ 메인 실행 루프
if __name__ == "__main__":
    query = input("입력: ")
    extract_subject_by_rapidfuzz(query)