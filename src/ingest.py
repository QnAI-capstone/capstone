import json
from src.embedder import get_embedding
from src.vectorstore import save_documents

def ingest_json_to_chroma(json_path: str, source_name: str, verbose: bool = True):
    with open(json_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    docs = []
    current_page = None
    if verbose:
        print(f"\n📄 JSON 로드 완료: {json_path} (총 {len(parsed)} 문서)\n")

    for item in parsed:
        content = item["content"]
        # 테이블(dict or list) → 문자열로 변환
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        embedding = get_embedding(text)

        if embedding is None:
            continue

        docs.append({
            "content": text,
            "page": item.get("page", -1),
            "embedding": embedding,
            "source": source_name
        })

        # 콘솔 출력 (요약)
        if verbose:
            page = item.get("page", -1)
            if current_page != page:
                current_page = page
                print("─" * 60)
                print(f"📘 Page {page}")

            if item.get("type") == "paragraph":
                print(f"\n📑 {item['id']}:\n{text[:200]}...\n")
            elif item.get("type") == "table":
                print(f"\n📊 {item['id']} (표 요약):")
                for row in content[:2]:  # 앞 2행만 요약
                    print("  -", row)
                print()

    save_documents(docs)
    print(f"\n✅ 저장 완료: {len(docs)}개의 문서가 Chroma DB에 추가되었습니다.")
