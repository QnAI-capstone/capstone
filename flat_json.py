import json

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
    # 예: 학사요람 JSON 로드
    with open("./data/json/마이크로전공.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 전공이 여러 개일 경우 반복
    for major_name, major_data in raw.items():
        text = flatten_json_to_text(major_data)
        print(f"\n🔹 전공: {major_name}")
        print(text[:500])  # 일부만 출력