from chromadb import PersistentClient
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_STORE_PATH = os.path.join(BASE_DIR, 'chroma_store')

client = PersistentClient(path=CHROMA_STORE_PATH)

collections = client.list_collections()

if not collections:
    print("현재 저장된 컬렉션이 없습니다.")
else:
    print(f"\n현재 저장된 컬렉션 수: {len(collections)}")
    for col in collections:
        name = col.name
        print(f"\n컬렉션 이름: {name}")
        
        # 3. 사용자 입력으로 삭제 여부 결정
        while True:
            choice = input(f"이 컬렉션 '{name}'을(를) 삭제하시겠습니까? (y/n): ").strip().lower()
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