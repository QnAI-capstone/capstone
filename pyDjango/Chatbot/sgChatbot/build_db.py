import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from sgChatbot.flat_json import flatten_json_to_text
from sgChatbot.chunk_split import group_by_course_blocks
from sgChatbot.flatten_notice import flat_notice, merge_by_index
import os

# collection 이름: collection_course, collection_subjectinfo, collection_notice

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
json_dir_list = ["course", "subjectinfo", "notice"] # JSON 파일들이 있는 디렉토리 경로 (필요 시 변경)

# ✅ 디렉토리별로 컬렉션 생성 및 파일 처리
for json_dir in json_dir_list:
    # 디렉토리 이름을 컬렉션 이름으로 사용
    collection_name = f"collection_{json_dir}"
    try:
        collection = client.get_collection(collection_name)
        print(f"⚠️ 컬렉션 '{collection_name}'은 이미 존재합니다. 컬렉션을 생성하지 않습니다.")
    except Exception as e:
        print(f"컬렉션 '{collection_name}'을 새로 생성합니다.")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
        file_list = [f for f in os.listdir(json_dir) if f.endswith(".json")]

        # ✅ 디렉토리 내 모든 JSON 파일을 처리
        for file_name in file_list:
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if collection_name == "collection_notice":
                # ✅ flatten 후 인덱스 기준 한 줄로 merge
                flat = flat_notice(raw)
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
                continue

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

                    if collection_name == "collection_course":
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

                    elif collection_name == "collection_subjectinfo":
                        chunks = group_by_course_blocks(flatten_json_to_text(data))

                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_id = f"{file_name}_{safe_university}_{safe_major}_{chunk_idx}"

                            metadata = {
                                "university": safe_university,
                                "major": safe_major,
                                "source_file": file_name,
                                "section": chunk_idx,  # or add section name if available
                                "id":chunk_id
                            }

                            collection.add(
                                documents=[chunk],
                                metadatas=[metadata],
                                ids=[chunk_id]
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