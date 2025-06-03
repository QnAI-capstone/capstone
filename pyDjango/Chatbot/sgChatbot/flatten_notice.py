import json
import os

def flat_notice(data, parent_key='', sep='.'):
    """
    딕셔너리와 리스트가 중첩된 JSON을 평탄화
    """
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flat_notice(v, new_key, sep=sep).items())
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flat_notice(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, data))
    return dict(items)

def merge_by_index(flat_dict, sep='.'):
    """
    같은 인덱스(예: 0, 1, 2)를 기준으로 title, date, url을 한 줄로 합침
    """
    from collections import defaultdict

    grouped = defaultdict(dict)
    for k, v in flat_dict.items():
        if sep in k:
            idx, field = k.split(sep, 1)
            grouped[idx][field] = v

    lines = []
    for idx in sorted(grouped, key=int):
        item = grouped[idx]
        line = f"{idx}: {item.get('title', '')} | {item.get('date', '')} | {item.get('url', '')}"
        lines.append(line)

    return "\n".join(lines)

if __name__ == "__main__":
    # 예: 학사요람 JSON 파일이 위치한 디렉토리 경로
    json_dir = 'notice/'  # JSON 파일들이 들어 있는 디렉토리 경로 (필요 시 변경)

    # 디렉토리 내 모든 JSON 파일을 처리
    for file in os.listdir(json_dir):
        if file.endswith(".json"):  # JSON 파일만 필터링
            file_path = os.path.join(json_dir, file)

            # 파일 열기
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            text = merge_by_index(flat_notice(raw))
            print(text[:500])