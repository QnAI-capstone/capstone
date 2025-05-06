from config import OPENAI_API_KEY
import openai
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction


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

# ✅ 문서 전부 불러오기 (한 번만 수행)
print("📚 문서 로딩 중...")
all_docs = collection.get(include=["documents", "metadatas"])
corpus = all_docs["documents"]
ids = all_docs["ids"]

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ✅ DPR 임베딩 (코사인 유사도용 벡터화)
dense_model = SentenceTransformer("jhgan/ko-sbert-sts")
dense_embeddings = dense_model.encode(corpus)

# ✅ 질문 → 하이브리드 검색 → GPT 응답
def ask(query, top_k_bm25=10, top_k_dpr=3):
    # 1. BM25 후보 추출
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]

    bm25_candidates = [corpus[i] for i in bm25_top_indices]
    bm25_embeddings = dense_embeddings[bm25_top_indices]

    # 2. DPR 랭킹 (semantic ranking)
    query_embedding = dense_model.encode([query])
    similarities = cosine_similarity(query_embedding, bm25_embeddings)[0]
    final_top_indices = np.argsort(similarities)[::-1][:top_k_dpr]

    final_docs = [bm25_candidates[i] for i in final_top_indices]

    # 3. GPT 프롬프트 구성
    context = "\n\n".join(final_docs)
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