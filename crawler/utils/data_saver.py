import json
import pandas as pd
import os

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장 완료: {path}")

def save_csv(data, path, columns=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장 완료: {path}")
