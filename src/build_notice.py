import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from flatten_notice import flat_notice,merge_by_index
import os

# collection 이름: collection_subjectinfo, collection_course

# ✅ 한국어 임베딩 함수 (KoSimCSE)
class KoSimCSEEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    def __call__(self, input):
        return self.model.encode(input).tolist()

# ✅ ChromaDB 클라이언트 초기화
client = PersistentClient(path="./chroma_store")
embedding_fn = KoSimCSEEmbedding()

# ✅ JSON 파일이 들어있는 디렉토리 경로 설정
json_dir = "notice"  # JSON 파일들이 있는 디렉토리 경로 (필요 시 변경)
file_list = [f for f in os.listdir(json_dir) if f.endswith(".json")]


# 주제마다 컬렉션 생성 -> 지금은 subjectinfo로 db 빌드함
collection = client.get_or_create_collection(
    name=f"collection_notice",  # 폴더명을 컬렉션 이름으로 사용
    embedding_function=embedding_fn
)

# ✅ 모든 JSON 파일 처리
for file_name in os.listdir(json_dir):
    if not file_name.endswith(".json"):
        continue

    file_path = os.path.join(json_dir, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # ✅ flatten 후 인덱스 기준 한 줄로 merge
    flat = flat_notice(raw_data)
    text = merge_by_index(flat)

    # ✅ metadata 구성
    date_part = file_name.rsplit("_", 1)[-1].replace(".json", "")
    metadata = {
        "source_file": file_name,
        "date": date_part
    }

    # ✅ Chroma DB에 추가
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[file_name.replace(".json", "")]
    )

print("✅ 전체 파일 저장 완료.")


# 컬렉션 목록 가져오기
collections = client.list_collections()

# 생성된 컬렉션 및 metadata 출력력
print("현재 생성된 컬렉션들 및 메타데이터:")

for col_info in collections:
    print(f"\n컬렉션 이름: {col_info.name}")
    collection = client.get_collection(name=col_info.name)
    data = collection.get(include=["metadatas"], limit=5)  # limit은 출력할 문서 수 제한
    metadatas = data.get("metadatas", [])

    if not metadatas:
        print(" - 메타데이터가 없습니다.")
        continue

    print(" - 일부 메타데이터:")
    for meta in metadatas:
        print(f"    {meta}")