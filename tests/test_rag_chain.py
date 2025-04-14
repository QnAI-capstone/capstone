from src.rag_chain import answer_query

def test_rag_chain():
    question = "캡스톤디자인의 목표는 무엇인가요?"
    response = answer_query(question)

    assert isinstance(response, str)
    assert len(response.strip()) > 0
    print("✅ LLM 응답:", response[:100], "...")
