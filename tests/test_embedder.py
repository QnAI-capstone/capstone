from src.embedder import get_embedding

def test_get_embedding():
    text = "이것은 테스트 문장입니다."
    embedding = get_embedding(text)

    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    print(f"✅ 임베딩 길이: {len(embedding)}")
