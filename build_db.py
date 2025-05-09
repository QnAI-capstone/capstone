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

# ✅ JSON 파일이 들어있는 디렉토리 경로 설정
json_dir = "./data/json/"  # JSON 파일들이 있는 디렉토리 경로
file_list = [f for f in os.listdir(json_dir) if f.endswith(".json")]
#print(file_list)

# ✅ 디렉토리 내 모든 JSON 파일을 처리
for file_name in file_list:
        file_path = os.path.join(json_dir, file_name)
        
        # 파일명만 추출하여 고유 ID 생성에 사용
        #file_name = os.path.splitext(file)[0]

        # JSON 파일 열기
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 각 학과 데이터를 벡터화하여 ChromaDB에 저장
        for idx, (major, data) in enumerate(raw.items()):
            majors = raw["2025 서강대학교 요람"]["인문대학"].keys()
            major = list(majors)[0]
            safe_major = major.replace(" ", "_").replace("(", "").replace(")", "")
            doc_id = f"{file_name}_{safe_major}_{idx}"  # ← 파일명 포함하여 고유 ID 생성

            # 파일마다 컬렉션 생성
            collection = client.get_or_create_collection(
            name=f"collection_{file_name}",  # 파일명을 컬렉션 이름으로 사용
            embedding_function=embedding_fn
            )

            print(f"📌 {safe_major} 저장 중.")
            text = flatten_json_to_text(data)  # 데이터를 텍스트로 평탄화

             # 디버깅용 출력: 텍스트와 메타데이터 확인
            #print(f"Text: {text[:200]}")  # 첫 200자 출력 (디버깅용)
            print(f"Metadata: {{'major': {safe_major}, 'source_file': {file_name}}}")  # 메타데이터 확인

            
            collection.add(
                documents=[text],
                metadatas=[{"major": safe_major, "source_file": file_name}],
                ids=[doc_id]
            )
            # 컬렉션 추가 후 결과 출력
            collection_size = collection.count()
            print(f"📌 {major} 데이터 저장 완료. 현재 컬렉션 데이터 수: {collection_size}")
            #except Exception as e:
                #print(f"❌ 에러 발생: {e}")


        print(f"✅ {file_name} 저장 완료.")

print("✅ 전체 파일 저장 완료.")

# 컬렉션 목록 가져오기
collections = client.list_collections()

# 생성된 컬렉션 출력
print("현재 생성된 컬렉션들:")
for collection in collections:
    print(collection)
