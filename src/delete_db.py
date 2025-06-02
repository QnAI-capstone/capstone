from chromadb import PersistentClient
import openai
import tiktoken
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
import re # 정규표현식 사용을 위해 추가
from rapidfuzz import process
from dictionary import ABBREVIATION_GROUPS
from collections import defaultdict
from flat_json import flatten_json_to_text
import json

def delete_collection_by_name(collection_name: str, db_path="./chroma_store"):
    client = PersistentClient(path=db_path)

    # 존재하는 컬렉션 이름 목록 확인
    all_collections = [col.name for col in client.list_collections()]
    
    if collection_name not in all_collections:
        print(f"❌ 컬렉션 '{collection_name}'이 존재하지 않습니다.")
        return

    # 컬렉션 삭제
    client.delete_collection(name=collection_name)
    print(f"🗑️ 컬렉션 '{collection_name}' 및 해당 문서들을 삭제했습니다.")

class KoSimCSEEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("jhgan/ko-sbert-sts")
    def __call__(self, input):
        return self.model.encode(input).tolist()

def load_corpus_by_collection():
    """
    ChromaDB의 모든 컬렉션을 순회하여,
    각 컬렉션 이름을 key로 하는 딕셔너리 형태로 데이터를 분리해 반환합니다.

    Returns:
        dict[str, dict]: {
            collection_name: {
                "documents": [...],
                "metadatas": [...],
                "majors": [...]
            },
            ...
        }
    """
    client = PersistentClient(path="./chroma_store")
    embedding_fn = KoSimCSEEmbedding()
    collections_info = client.list_collections()

    print(f"총 {len(collections_info)}개의 컬렉션을 불러옵니다.")

    result = {}

    for col_info in collections_info:
        collection_name = col_info.name
        collection = client.get_collection(name=collection_name, embedding_function=embedding_fn)
        data = collection.get(include=["documents", "metadatas"])

        documents = data["documents"]
        metadatas = data["metadatas"]
        majors = list({meta["major"] for meta in metadatas if meta and "major" in meta})

        result[collection_name] = {
            "documents": documents,
            "metadatas": metadatas,
            "majors": majors
        }

        print(f"✅ '{collection_name}' 컬렉션 불러오기 완료. 문서 수: {len(documents)}")

    return result

def print_all_majors_in_collection(collection_name: str, db_path="./chroma_store"):
    client = PersistentClient(path=db_path)
    embedding_fn = KoSimCSEEmbedding()

    collection = client.get_collection(name=collection_name, embedding_function=embedding_fn)
    data = collection.get(include=["metadatas"])

    all_majors = set()

    for meta in data["metadatas"]:
        if meta and "major" in meta:
            all_majors.add(meta["major"])

    print(f"📘 컬렉션 '{collection_name}'에 포함된 고유 major 목록 ({len(all_majors)}개):\n")
    for major in sorted(all_majors):
        print(f" - {major}")


# ✅ 메인 실행 루프
if __name__ == "__main__":
    
    delete_collection_by_name("collection_subjectinfo")
    
    # 현재 디비에 있는 컬렉션 출력
    client = PersistentClient(path="./chroma_store")

    collections = client.list_collections()

    print(f"📦 현재 저장된 컬렉션 수: {len(collections)}")
    for i, col in enumerate(collections):
        print(f"  [{i}] {col.name}")


    delete_collection_by_name("collection_course")
    
    # 현재 디비에 있는 컬렉션 출력
    client = PersistentClient(path="./chroma_store")

    collections = client.list_collections()

    print(f"📦 현재 저장된 컬렉션 수: {len(collections)}")
    for i, col in enumerate(collections):
        print(f"  [{i}] {col.name}")