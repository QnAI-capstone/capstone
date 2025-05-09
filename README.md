# capstone
build_db.py - line 37: raw["2025 서강대학교 요람"]["인문대학"].keys() 에서 "인문대학" 부분을 대학별로 변경해서 build하면 학과별 collection 뽑아낼 수 있음

collection_name: json 파일명 영어로 변경 -> collection_{file_name}으로 사용
-> ex) 학과별 요람정보(국어국문학과): Major_Korean.json

flat_json.py / hybrid_rag.py - 수정된 코드 구조에 맞춰 flatten 후 hybrid 검색 가능한 형식으로 수정

# 추가 고려사항
1. collection 구조
- 요람 정보 나누기 / 개설교과목 정보와 합치기 작업 진행중인데 collection 구조를 어떻게 설계해서 documents 넣을지 정해야함
2. collection 이름
- 각 학과별 / 정보별로 collection명 어떤식으로 하면 좋을지 정해야함
3. rag 검색 구조
- gpt-4o 토큰 수 제한 이슈로 질문을 받으면 context를 걸러서 넘겨야 될 듯 (개선 필요)
- gpt api에 전달하는 context 분량 or 디비에 저장할 데이터의 양을 줄일 수 있는 방법이 있으면 참 좋을 것 같습니당 (같이 고민해봐요)
