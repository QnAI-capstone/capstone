import openai
from src.config import OPENAI_API_KEY
from src.embedder import get_embedding
from src.vectorstore import query_documents

openai.api_key = OPENAI_API_KEY

def answer_query(query: str, top_k: int = 3) -> str:
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "❌ 쿼리 임베딩 실패"

    search_result = query_documents(query_embedding, top_k=top_k)
    contexts = search_result.get("documents", [[]])[0]
    metadatas = search_result.get("metadatas", [[]])[0]

    # 컨텍스트 + 출처 정보
    context_text = ""
    for i, (text, meta) in enumerate(zip(contexts, metadatas), start=1):
        page = meta.get("page", "?")
        src = meta.get("source", "?")
        context_text += f"\n---\n📄 Doc {i} (Page {page}, Source: {src})\n{text}\n"

    # GPT에 전달할 프롬프트
    prompt = f"""
당신은 PDF 문서 기반의 지식 챗봇입니다.
아래 문서를 참고하여 사용자 질문에 답하세요.
문서에 없는 내용은 모른다고 하세요.

질문: {query}

문서 내용:
{context_text}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4" 사용 가능
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GPT 응답 실패: {e}"
