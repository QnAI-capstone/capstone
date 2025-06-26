from chromadb import PersistentClient
import os

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.abspath(os.path.join(__file__, "../../.."))
CHROMA_STORE_PATH = os.path.join(BASE_DIR, 'chroma_store')

client = PersistentClient(path=CHROMA_STORE_PATH)

collections = client.list_collections()

if not collections:
    print("현재 저장된 컬렉션이 없습니다.")
else:
    print(f"\n현재 저장된 컬렉션 수: {len(collections)}")
    for col in collections:
        name = col.name
        print(f"\n📁 컬렉션 이름: {name}")

        # 컬렉션 가져오기
        collection = client.get_collection(name=name)

        # 컬렉션 내 모든 문서 가져오기
        try:
            results = collection.get(include=["documents", "metadatas"])
        except Exception as e:
            print(f"⚠️ 컬렉션 '{name}'에서 데이터를 가져오는 중 오류 발생: {e}")
            continue

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        # 문서 수 출력
        print(f"   - 문서 수: {len(docs)}")

        # 제목 및 문서 내용 일부 출력
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            title = meta.get("title", "(제목 없음)")
            preview = doc[:100].replace("\n", " ") + ("..." if len(doc) > 100 else "")
            print(f"     [{i+1}] 제목: {title}")
            print(f"          내용 미리보기: {preview}")
        
        while True:
            choice = input(f"컬렉션 '{name}'을(를) 삭제하시겠습니까? (y/n): ").strip().lower()
            if choice == "y":
                client.delete_collection(name=name)
                print(f">> 컬렉션 '{name}' 삭제 완료.")
                break
            elif choice == "n":
                print(f">> 컬렉션 '{name}' 유지됨.")
                break
            else:
                print(">> 유효하지 않은 입력입니다. 'y' 또는 'n'을 입력하세요.")
        
print("✅ 컬렉션 확인 완료")
