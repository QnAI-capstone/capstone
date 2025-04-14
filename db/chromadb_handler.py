from chromadb import Client
import numpy as np

# Chroma 클라이언트 생성
client = Client()

# 벡터 데이터베이스에 저장
def save_to_chroma(pdf_path):
    embedding = embed_pdf_with_openai(pdf_path)
    # Chroma 컬렉션 생성 (하나의 문서로 저장)
    collection = client.create_collection("pdf_collection")
    collection.add(
        documents=[pdf_path],  # 문서 이름이나 페이지 정보
        metadatas=[{"pdf_path": pdf_path}],
        embeddings=[embedding]  # 임베딩 벡터
    )

# 사용자가 입력한 질문에 대해 가장 관련성 높은 벡터 검색
def query_chroma(query_text):
    query_embedding = embed_pdf_with_openai(query_text)
    results = client.query(
        collection_name="pdf_collection",
        query_embeddings=[query_embedding],
        n_results=1  # 가장 관련성 높은 1개의 결과만 검색
    )
    return results
