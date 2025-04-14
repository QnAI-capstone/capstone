import json
from src.vectorstore import collection

def show_docs(limit=10):
    data = collection.get()
    documents = data["documents"]
    metadatas = data["metadatas"]

    print(f"\n📦 현재 Chroma DB에 저장된 문서 수: {len(documents)}\n")

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        if i >= limit:
            print(f"... (생략됨: 총 {len(documents) - limit}개 문서 더 있음)")
            break

        print("─" * 60)
        print(f"📄 Page {meta['page']} | Source: {meta['source']}")

        try:
            parsed = json.loads(doc)
            if isinstance(parsed, list):  # table JSON
                print(f"📊 Table (요약):")
                for row in parsed[:2]:  # 앞부분 미리보기
                    print("  -", row)
            else:
                print(f"📑 Paragraph (요약):\n{parsed[:200]}...")
        except json.JSONDecodeError:
            print(f"📑 Paragraph (요약):\n{doc[:200]}...")

    print()

if __name__ == "__main__":
    show_docs()
