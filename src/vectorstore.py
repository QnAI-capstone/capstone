import chromadb
from chromadb.config import Settings

# 🧠 Chroma client 설정 (로컬 저장소)
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb"  # 변경 가능
))

# ✅ 기존 컬렉션 초기화 후 새로 생성
def reset_collection(name="pdf_docs"):
    if name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name)
    return chroma_client.create_collection(name)

# 기본 컬렉션 객체
collection = reset_collection()

# 📦 문서 저장
def save_documents(docs):
    for doc in docs:
        collection.add(
            documents=[doc["content"]],
            embeddings=[doc["embedding"]],
            metadatas=[{
                "page": doc["page"],
                "source": doc["source"]
            }],
            ids=[f'{doc["source"]}_p{doc["page"]}']
        )

# 🔍 쿼리 벡터로 검색
def query_documents(query_embedding, top_k=3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
