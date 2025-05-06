import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from flat_json import flatten_json_to_text
import os

# ✅ 한국어 임베딩 함수 (KoSimCSE)
class KoSimCSEEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("jhgan/ko-sbert-sts")
    def __call__(self, input):
        return self.model.encode(input).tolist()
    

# ✅ ChromaDB 클라이언트 초기화
client = PersistentClient(path="./chroma_store")
embedding_fn = KoSimCSEEmbedding()

collection = client.get_or_create_collection(
    name="micro_collection",
    embedding_function=embedding_fn
)

# ✅ JSON 파일 경로
file_path = "./data/json/마이크로전공.json"
file_name = os.path.splitext(os.path.basename(file_path))[0]

with open(file_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

for idx, (major, data) in enumerate(raw.items()):
    doc_id = f"{file_name}_{major}_{idx}"  # ← 파일명 포함하여 고유 ID 생성

    print(f"📌 {major} 저장 중.")
    text = flatten_json_to_text(data)
    collection.add(
        documents=[text],
        metadatas=[{"major": major, "source_file": file_name}],
        ids=[doc_id]
    )

print("✅ 저장 완료.")
