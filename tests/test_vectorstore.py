from src.vectorstore import save_documents, query_documents

def test_vectorstore_storage_and_query():
    dummy_doc = {
        "content": "캡스톤디자인은 4학년 학생들이 수행하는 프로젝트 수업입니다.",
        "page": 1,
        "embedding": [0.1] * 768,  # Gemini 임베딩과 같은 형태
        "source": "dummy.pdf"
    }

    # 저장
    save_documents([dummy_doc])

    # 검색
    results = query_documents([0.1] * 768, top_k=1)
    assert len(results["documents"][0]) > 0
    print(f"✅ Chroma DB 검색 결과: {results['documents'][0][0][:50]}...")
