import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from flat_json import flatten_json_to_text
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
json_dir = "subjectinfo"  # JSON 파일들이 있는 디렉토리 경로 (필요 시 변경)
file_list = [f for f in os.listdir(json_dir) if f.endswith(".json")]


# 주제마다 컬렉션 생성 -> 지금은 subjectinfo로 db 빌드함
collection = client.get_or_create_collection(
    name=f"collection_subjectinfo",  # 폴더명을 컬렉션 이름으로 사용
    embedding_function=embedding_fn
)

# ✅ 디렉토리 내 모든 JSON 파일을 처리
for file_name in file_list:
    file_path = os.path.join(json_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # "2025 서강대학교 요람" 최상위 key로 접근
    base_key = "2025 서강대학교 요람"
    if base_key not in raw:
        print(f"⚠️ '{base_key}' 키가 파일 {file_name}에 없습니다.")
        continue

    base_data = raw[base_key]

    # 대학명별 순회
    for university_name, majors_data in base_data.items():
        safe_university = university_name.replace(" ", "_").replace("(", "").replace(")", "")

        # 학과별 순회
        for major, data in majors_data.items():
            safe_major = major.replace(" ", "_").replace("(", "").replace(")", "")
            doc_id = f"{file_name}_{safe_university}_{safe_major}"

            print(f"📌 {safe_university} - {safe_major} 저장 중.")
            text = flatten_json_to_text(data)

            metadata = {
                "university": safe_university,
                "major": safe_major,
                "source_file": file_name,
            }

            collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            print(f"📌 {safe_university} - {safe_major} 데이터 저장 완료. 현재 컬렉션 데이터 수: {collection.count()}")
            print(f"    저장된 메타데이터: {metadata}")


print("✅ 전체 파일 저장 완료.")


# 컬렉션 목록 가져오기
collections = client.list_collections()

# 생성된 컬렉션 및 metadata 출력
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