import json
import os

def flatten_json(data, parent_key='', sep='.'):
    """
    딕셔너리와 리스트가 중첩된 JSON을 평탄화
    """
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep).items())
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten_json(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, data))
    return dict(items)

def flatten_json_to_text(data, sep='.'):
    """
    평탄화한 JSON을 key: value 문자열로 변환 (벡터 DB 입력용)
    """
    flat = flatten_json(data, sep=sep)
    lines = [f"{k}: {v}" for k, v in flat.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    # 예: 학사요람 JSON 파일이 위치한 디렉토리 경로
    json_dir = 'subjectinfo/'  # JSON 파일들이 들어 있는 디렉토리 경로 (필요 시 변경)

    # 디렉토리 내 모든 JSON 파일을 처리
    for file in os.listdir(json_dir):
        if file.endswith(".json"):  # JSON 파일만 필터링
            file_path = os.path.join(json_dir, file)

            # 파일 열기
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # 전공별로 데이터를 평탄화하여 텍스트로 변환 후 출력
            for major_name, major_data in raw.items():
                text = flatten_json_to_text(major_data)
                #major_name = flatten_json(major_data).parent_key
                print(f"\n🔹 전공: {major_name}")
                print(text[:500])  # 일부만 출력