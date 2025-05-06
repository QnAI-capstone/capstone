import os
from config import OPENAI_API_KEY
import openai
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

# ✅ API 키 로딩
openai.api_key = OPENAI_API_KEY

# ✅ KoSimCSE 임베딩 함수 (ChromaDB에 저장한 것과 동일하게 유지해야 함)
class KoSimCSEEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("jhgan/ko-sbert-sts")
    def __call__(self, input):
        return self.model.encode(input).tolist()

# ✅ ChromaDB 불러오기
client = PersistentClient(path="./chroma_store")
embedding_fn = KoSimCSEEmbedding()

collection = client.get_or_create_collection(
    name="micro_collection",
    embedding_function=embedding_fn
)

# ✅ 질문 → 검색 → GPT 응답
def ask(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    context = "\n\n".join(results["documents"][0])
    prompt = f"""다음 내용을 참고하여 질문에 답변하세요:\n\n{context}\n\n질문: {query}\n답변:"""

    completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

# ✅ 실행 루프
if __name__ == "__main__":
    print("💬 학사요람 기반 RAG 시스템 시작됨. 질문을 입력하세요 (종료: 'exit')")

    while True:
        query = input("\n❓ 질문: ")
        if query.lower() == "exit":
            break
        answer = ask(query)
        print(f"\n🧠 GPT 응답:\n{answer}")