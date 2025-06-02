from config import OPENAI_API_KEY
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
from dictionary import ABBREVIATION_GROUPS,DATE_GROUPS
import math
from ex_sub import extract_subject_by_rapidfuzz

# ✅ API 키
openai.api_key = OPENAI_API_KEY

# ✅ 한국어 임베딩 함수
class KoSimCSEEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    def __call__(self, input):
        return self.model.encode(input).tolist()

#notice -> 수정
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

        #공지 데이터 처리
        if collection_name == "collection_notice":
            dates = list({meta.get("date") for meta in metadatas if meta and "date" in meta})
            result[collection_name] = {
                "documents": documents,
                "metadatas": metadatas,
                "dates": dates
            }
            print(f"📌 'notice_collection' 불러오기 완료. 문서 수: {len(documents)}")

        
        #과목 이수, 과목 설명 데이터 처리
        else:
            majors = list({meta["major"] for meta in metadatas if meta and "major" in meta})

            result[collection_name] = {
                "documents": documents,
                "metadatas": metadatas,
                "majors": majors
            }

            print(f"✅ '{collection_name}' 컬렉션 불러오기 완료. 문서 수: {len(documents)}")

    return result

def count_total_tokens(messages, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for message in messages:
        total_tokens += 4
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
    total_tokens += 2
    return total_tokens

# ✅ 사용자 질의에서 학과 키워드 추출 함수
def extract_major_keyword(query, majors_list,threshold=70):
    """
    사용자 질의에서 언급된 학과 키워드를 majors_list (DB에 저장된 실제 학과명) 기준으로 유사 문자열 매칭하여 추출합니다.
    띄어쓰기, 오타, 축약어 차이 등으로 정확히 일치하지 않아도 가장 유사한 학과명을 찾아 반환합니다.
    """
    # 사용자 질의 정규화: 공백 제거, 소문자 변환
    normalized_query = query.replace(" ", "").lower()

    # majors_list의 학과명도 정규화 (언더스코어 제거, 소문자 변환)
    candidates = [m.replace("_", "").lower() for m in majors_list]

    # rapidfuzz의 extractOne으로 가장 유사한 학과명과 유사도 반환
    best_match = process.extractOne(normalized_query, candidates)
    
    # ✅ 유사한 항목 여러 개 추출
    matches = process.extract(normalized_query, candidates, limit=2)
    # ✅ 유사도 기준 통과한 학과만 반환
    result = []
    for match_str, score, _ in matches:
        idx = candidates.index(match_str)
        print(majors_list[idx])
        if score >= threshold:
            idx = candidates.index(match_str)
            matched_major = majors_list[idx]
            result.append(matched_major)
            print(f"✅ 유사도 {score} → 매칭된 학과: {matched_major}")
        else:
            print(f"❌ 유사도 {score} → 무시됨")
        
    return result  # 최대 top_k개의 학과명 반환



# ✅ 하이브리드 검색기 초기화
class HybridRetriever:
    def __init__(self, corpus_all, metadatas_all,collection_name):
        # 초기화 시점에는 전체 데이터를 보관
        self.corpus_all = corpus_all
        self.metadatas_all = metadatas_all
        self.encoder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        self.collection_name = collection_name

    def retrieve(self, query, top_k_bm25=10, top_k_dpr=3, filter_major=None,alpha=0.5,cat=1):
        # 실제 검색 대상이 될 코퍼스와 메타데이터
        current_corpus = self.corpus_all
        current_metadatas = self.metadatas_all

        # 필터링할 학과가 지정된 경우
        if filter_major:
            print(f"🔍 '{filter_major}' 학과 관련 문서로 필터링 중...")
            filtered_indices = [
                i for i, meta in enumerate(self.metadatas_all)
                if meta and meta.get('major') in filter_major #filter_major를 리스트로 바꿈
            ]
            if not filtered_indices:
                print(f"⚠️ '{filter_major}' 학과 관련 문서를 찾을 수 없습니다. 전체 문서에서 검색합니다.")
            else:
                current_corpus = [self.corpus_all[i] for i in filtered_indices]
                current_metadatas = [self.metadatas_all[i] for i in filtered_indices]
                print(f"🔎 필터링 결과: 총 {len(current_corpus)}개의 문서로 제한됨.")


        if not current_corpus: # 필터링 후 문서가 없을 경우
             print("⚠️ 검색할 문서가 없습니다.")
             return []

        # BM25 계산 (필터링된 코퍼스 또는 전체 코퍼스 대상)
        tokenized_corpus = [doc.split() for doc in current_corpus]
        
        bm25 = BM25Okapi(tokenized_corpus,b=0.25)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # BM25 결과가 top_k_bm25보다 적을 경우를 대비
        num_bm25_candidates = min(top_k_bm25, len(current_corpus))
        bm25_indices_in_current = np.argsort(bm25_scores)[::-1][:num_bm25_candidates]
        bm25_candidates_docs = [current_corpus[i] for i in bm25_indices_in_current]
        bm25_candidates_meta = [current_metadatas[i] for i in bm25_indices_in_current]

        # -------------------------------
        # 2️⃣ ChromaDB 벡터 불러와 의미 유사도 계산
        # -------------------------------
        
        embedding_fn = KoSimCSEEmbedding()
        client = PersistentClient(path="./chroma_store")
        collection = client.get_collection(name=self.collection_name, embedding_function=embedding_fn)

        # BM25 후보에 해당하는 문서 ID 생성
        bm25_candidate_ids = []
        if cat == 1:
            for meta in bm25_candidates_meta:
                file = meta.get("source_file", "")
                university = meta.get("university", "")
                major = meta.get("major", "")
                bm25_candidate_ids.append(f"{file}_{university}_{major}")
        else:
            bm25_candidate_ids = [meta["id"] for meta in bm25_candidates_meta if "id" in meta]

        # 해당 ID들의 임베딩 불러오기
        retrieved = collection.get(ids=bm25_candidate_ids, include=["embeddings"])
        candidate_embeddings = retrieved["embeddings"]

        # 질의 임베딩 생성
        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Top-K DPR 유사도 정렬
        num_dpr_candidates = min(top_k_dpr, len(bm25_candidates_docs))
        top_indices_in_bm25 = np.argsort(similarities)[::-1][:num_dpr_candidates]

        # ✅ 하이브리드 점수 계산
        bm25_top_scores = np.array([bm25_scores[idx] for idx in bm25_indices_in_current])
        bm25_norm = (bm25_top_scores - np.min(bm25_top_scores)) / (np.max(bm25_top_scores) - np.min(bm25_top_scores) + 1e-8)
        dpr_norm = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities) + 1e-8)

        hybrid_scores = alpha * bm25_norm + (1 - alpha) * dpr_norm
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k_dpr]

        sorted_indices = np.argsort(hybrid_scores)[::-1]  # 내림차순 정렬 인덱스

        for rank, i in enumerate(sorted_indices):
            raw_bm25_score = bm25_top_scores[i]
            raw_dpr_score = similarities[i]
    
            print(f"  [{rank}] 통합점수: {hybrid_scores[i]:.4f} | "
                f"BM25(norm): {bm25_norm[i]:.4f} | BM25(raw): {raw_bm25_score:.4f} | "
                f"DPR(norm): {dpr_norm[i]:.4f} | DPR(raw): {raw_dpr_score:.4f} | "
                f"메타: {bm25_candidates_meta[i]}")


        final_results = [
            (bm25_candidates_docs[i], bm25_candidates_meta[i]) for i in sorted_indices[:top_k_dpr]
        ]


        return final_results

# ✅ GPT 응답 생성기
def generate_answer(query, context_docs,cat):
    
    if not context_docs: # 참고 문서가 없는 경우
        return "관련 정보를 찾지 못했습니다. 질문을 조금 더 구체적으로 해주시거나 다른 키워드로 시도해주세요."

    if cat == 1:
        
        context = "\n\n".join([
            f"[{meta.get('university', 'Unknown University')} - {meta.get('major', 'Unknown Major')} - {meta.get('source_file', 'Unknown Source')}]\n{doc}"
            for doc, meta in context_docs
        ])

        prompt = (
            "당신은 서강대학교의 학사 요람에 기반하여 정확하고 간결하게 답변해야 합니다.\n"
            "질문이 모호하더라도 관련 학과 또는 규정 문서를 참고하여 정확하게 답변하세요.\n"
            "사용자는 아래의 세 가지 전공 유형 중 하나에 해당할 수 있습니다. 이 구분은 모든 학과에 동일하게 적용되며, 어떤 전공이 주 전공인지에 따라 학과별 졸업 요건 및 이수 기준이 달라질 수 있습니다:\n"
            
            "1. **단일전공**: 사용자는 특정 학과(예: 컴퓨터공학과)만 전공합니다.\n"
            "2. **다전공(자신의 학과)**: 사용자는 해당 학과를 제1전공으로 하고, 다른 학과를 복수전공합니다.\n"
            "3. **다전공(타 학과)**: 사용자는 다른 학과를 제1전공으로 하고, 해당 학과를 복수전공합니다.\n"

            "예시) 질문이 컴퓨터공학과에 대한 것일 경우:\n"
            "- \"단일전공\" 사용자는 컴퓨터공학과만 전공\n"
            "- \"다전공(컴공)\" 사용자는 컴퓨터공학과가 제1전공 + 다른 학과 복수전공\n"
            "- \"다전공(타전공)\" 사용자는 다른 학과가 제1전공 + 컴퓨터공학과 복수전공\n"

            "질문이 어느 전공 유형에 해당하는지 명확하지 않더라도, 각 경우에 따라 달라지는 내용을 **모두 분리된 문단**으로 나눠 설명하세요.\n"
            "제공된 context에서 찾을 수 없다면 찾을 수 없다고 메시지를 출력해주세요.\n"
            
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"context:\n{context}\n\n질문: {query}\n답변:"}
        ]
        model_name = "gpt-4o"
    
    elif cat == 2:
        context = "\n\n".join([
            f"[{meta.get('university', 'Unknown University')} - {meta.get('major', 'Unknown Major')} - {meta.get('source_file', 'Unknown Source')}]\n{doc}"
            for doc, meta in context_docs
        ])

        prompt = (
            "당신은 서강대학교의 학사 요람 정보를 기반으로, 사용자 질문에 대해 정확하고 간결하게 답변해야 합니다.\n"
            "- 질문이 모호하더라도, 관련 학과 또는 규정 문서를 모두 참고하여 가능한 모든 정보를 포함하세요.\n"
            "- 같은 과목에 대한 설명이 여러 학과 또는 전공에서 반복될 경우, **모든 관련 문서에서 나온 설명을 빠짐없이 포함**하세요.\n"
            "- 각각의 설명은 **출처 학과명 기준으로 문단을 분리하여 출력**하고, 중복된 내용이 있더라도 **학과 문맥 내에서는 생략하지 말고 모두 출력**하세요.\n"
            "- 요약하지 마세요. **모든 학과별 설명을 전부 나열**하는 것이 중요합니다.\n"
            "- 제공된 context에서 답변을 찾을 수 없을 경우, \"제공된 정보에서 답변을 찾을 수 없습니다.\"라고 출력하세요.\n"

            
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"context:\n{context}\n\n질문: {query}\n답변:"}
        ]
        #model_name = "ft:gpt-3.5-turbo-0125:capston::Bdtr05OS"
        model_name="gpt-4o"

    else:
        
        context = context_docs
        prompt = (
            "당신은 서강대학교의 공지사항 데이터를 기반으로 질문에 정확하고 간결하게 답변하는 어시스턴트입니다.\n"
            #"다음 예시를 참고하여 답변하되, 링크는 한 번만 출력하세요.\n"
            "질문이 모호하더라도, 제공된 공지 context를 바탕으로 규정과 사실에 근거해 답변해야 합니다.\n"
            "가능한 한 질문과 키워드가 정확히 일치하는 공지를 찾아서 제시하세요.\n"
            "여러 개의 공지가 관련 있다면, 날짜(date)가 가장 최신인 순서로 정렬하여 출력하세요.\n"
            "제공된 context에 관련 정보가 없다면, '관련 공지를 찾을 수 없습니다.'라고 답변하세요.\n"
            "링크는 반드시 한 번만 출력하고, 마크다운 문법을 사용하지 말고 순수한 URL만 출력하세요.\n\n"
        )


        messages = [
            {"role": "system", "content": prompt},
            # 🟡 One-shot 예시
            {"role": "user", "content": "context:\n[졸업] 2023학년도 후기(2024년 8월) 졸업_학위증 배부 및 학위가운 대여 안내|2024.07.30|https://sogang.ac.kr/ko/detail/\n\n질문: 학위 가운은 어디서 대여할 수 있어?\n답변:"},
            {"role": "assistant", "content": "학위 가운 대여와 관련하여 다음 공지를 참조하세요.\n제목:[졸업] 2023학년도 후기(2024년 8월) 졸업_학위증 배부 및 학위가운 대여 안내\n업로드일자: 2024.07.30\n링크:https://sogang.ac.kr/ko/detail/\n"},
            {"role": "user", "content": f"context:\n{context}\n\n질문: {query}\n답변:"}
        ]
        model_name = "gpt-4o"

    total_tokens = count_total_tokens(messages, model="gpt-4o")
    max_tokens_model = 128000 # 모델의 최대 토큰 (gpt-4o 기준)
    max_response_tokens = 4096 # 답변으로 받고자 하는 최대 토큰 수

    # 프롬프트가 너무 길 경우, 모델의 최대 토큰 수를 넘지 않도록 자르거나,
    # 답변 생성 토큰 수를 고려하여 입력 토큰을 조절해야 함.
    # 여기서는 단순 경고만 출력
    if total_tokens > max_tokens_model - max_response_tokens : # 모델 한계 - 응답 토큰 = 프롬프트 최대
        print(f"⚠️ 전체 프롬프트 토큰 수가 {total_tokens}개로 모델 제한({max_tokens_model})을 초과할 수 있습니다. 일부 컨텍스트가 잘릴 수 있습니다.")
        # 실제로는 context를 줄이는 로직이 필요합니다.
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=4096, # 답변 토큰 수 제한
            temperature = 0.3,
            top_p = 0.9
        )
        return response.choices[0].message['content'] # 수정: .message.content -> .message['content']
    except openai.error.InvalidRequestError as e:
        print(f"❌ 오류 발생: {e}")
        # 토큰 초과 에러의 경우, 여기서 context를 줄여서 재시도하는 로직을 넣을 수 있습니다.
        if "maximum context length" in str(e):
            return "질문과 참고 문서의 내용이 너무 길어 답변을 생성할 수 없습니다. 더 짧게 질문해주시거나, 필터링을 통해 문서 범위를 줄여주세요."
        return "에러가 발생했습니다. 다시 시도해주세요."
    except Exception as e: # 다른 예외 처리
        print(f"❌ 예상치 못한 오류 발생: {e}")
        return "예상치 못한 오류로 답변을 생성할 수 없습니다."


#query preprocess module
KOREAN_PARTICLE_PATTERN = r'(은|는|이|가|을|를|에|에서|으로|로|도|만|까지|부터|조차|인데|고|와|과|마저|처럼|께서|밖에|이며|이고|이나|라도|라고|라는|든지|만큼|야)?'

def preprocess_query(query):
    used_majors = []
    replaced_ranges = []

    for full_name, variants in ABBREVIATION_GROUPS.items():
        for variant in sorted(variants, key=lambda x: -len(x)):
            # 변형어가 조사와 함께 붙은 형태로 끝날 때도 매칭
            pattern = re.compile(rf'(?<!\w)({re.escape(variant)}){KOREAN_PARTICLE_PATTERN}')

            def replacer(m):
                start, end = m.start(1), m.end(1)  # variant만큼의 범위로 비교
                for r_start, r_end in replaced_ranges:
                    if not (end <= r_start or start >= r_end):
                        return m.group(0)  # 이미 겹친 경우 치환하지 않음

                replaced_ranges.append((start, start + len(full_name)))
                if full_name not in used_majors:
                    used_majors.append(full_name)
                particle = m.group(2) or ""
                return full_name + particle

            query = pattern.sub(replacer, query)

    return query

    
def extract_date_key_from_query(query: str) -> str | None:
    """
    query에 DATE_GROUPS의 value 중 하나라도 포함되면 해당 key를 반환.
    없으면 None 반환.
    """
    for key, phrases in DATE_GROUPS.items():
        for phrase in phrases:
            if phrase in query:
                return key
    return None

# ✅ 메인 실행 루프
if __name__ == "__main__":
    print("💬 학사요람 기반 RAG 시스템 시작됨.")

    collection_data = load_corpus_by_collection()
    if not collection_data:
        print("⚠️ 로드된 문서가 없습니다. DB를 먼저 생성하거나 경로를 확인해주세요.")
        exit()

    # ✅ 컬렉션별 retriever 초기화
    retrievers = {}
    for col_name, content in collection_data.items():
        retrievers[col_name] = HybridRetriever(
            content["documents"],
            content["metadatas"],
            collection_name=col_name
        )

    # ✅ major 목록도 함께 저장
    majors_by_collection = {
        col_name: content["majors"]
        for col_name, content in collection_data.items()
        if "majors" in content
    }   

    # ✅ 카테고리 → 컬렉션 이름 매핑
    category_to_collection = {
        "1": "collection_course",
        "2": "collection_subjectinfo_chunk",
        "3": "collection_notice"
    }

    while True:
        #category 초기화
        print("\n어떤 카테고리의 질문을 할지 골라주세요.")
        cat = input("\n1. 과목/전공 이수 요건 2. 과목 정보 3. 학사 공지\n-> ")

        if cat not in category_to_collection:
            print("⚠️ 잘못된 입력입니다. 1, 2 또는 3을 입력하세요.")
            continue
        else:
            break
    
    selected_collection = category_to_collection[cat]

    if selected_collection not in retrievers:
        print(f"❌ 선택한 컬렉션 '{selected_collection}'이 로드되지 않았습니다.")
        exit()

    retriever = retrievers[selected_collection]

    if selected_collection != "collection_notice":
        unique_majors = majors_by_collection[selected_collection]


    while True:
        query = input("\n❓ 질문을 입력하세요 (종료를 원하면 exit을, category 변경을 원하면 cat을 입력해주세요.): ")
        if query.lower().strip() == "exit":
            print("🚫챗봇을 종료합니다.\n")
            break
        elif query.lower().strip() == "cat":
            print("\n어떤 카테고리의 질문을 할지 골라주세요.")
            cat = input("\n1. 과목/전공 이수 요건 2. 과목 정보 3. 학사 공지\n-> ")

            if cat not in category_to_collection:
                print("⚠️ 잘못된 입력입니다. 1, 2 또는 3을 입력하세요.")
                continue
    
            selected_collection = category_to_collection[cat]

            if selected_collection not in retrievers:
                print(f"❌ 선택한 컬렉션 '{selected_collection}'이 로드되지 않았습니다.")
                break

            retriever = retrievers[selected_collection]
            if selected_collection != "collection_notice":
                unique_majors = majors_by_collection[selected_collection]

            query = input("\n❓ 질문을 입력하세요 (종료를 원하면 exit을, category 변경을 원하면 cat을을 입력해주세요.): ")

        if selected_collection == "collection_notice":
            #시기 추출
            date_keyword = extract_date_key_from_query(query)
            

             # 기본값 설정
            if date_keyword is None:
                print("ℹ️ 특정 시기 키워드가 감지되지 않았습니다. '2025-1' 문서만 검색합니다.")
                date_keyword = "2025-1"
            else:
                print(f"✨ '{date_keyword}' 관련 정보로 필터링하여 검색합니다.")

            query = query.strip()+" "+date_keyword+"와 관련된 공지를 찾아줘."
            #print(query)

            # ✅ 해당 시기의 문서만 필터링
            all_docs = collection_data[selected_collection]["documents"]
            all_metas = collection_data[selected_collection]["metadatas"]

            answer=generate_answer(query, all_docs, cat=3)

        elif selected_collection == "collection_subjectinfo_chunk":
            # 1) 축약어 그룹 치환 적용
            query = preprocess_query(query)

            # 2) 변환된 질의로 학과 키워드 추출
            major_filter_keyword = extract_major_keyword(query, unique_majors,threshold = 80)

            if major_filter_keyword:
                print(f"✨ '{major_filter_keyword}' 관련 정보로 필터링하여 검색합니다.")
            else:
                print("ℹ️ 특정 학과 키워드가 감지되지 않았습니다. 전체 문서에서 검색합니다.")
                sub = extract_subject_by_rapidfuzz(query)
                query = query.strip()+" "+sub[0]+" 과목"

            print(f"query: {query}")
        

            # 3) 필터링 키워드를 retriever에 전달
            top_docs_with_meta = retriever.retrieve(query, top_k_bm25=10, top_k_dpr=3, filter_major=major_filter_keyword,alpha=0.8,cat= 2)
            print(cat)

            if not top_docs_with_meta:
                print("\n🧠 chatbot 응답:\n관련된 문서를 찾지 못했습니다. 다른 질문을 해주시거나 키워드를 확인해주세요.")
                continue # 다음 질문으로

            answer = generate_answer(query, top_docs_with_meta, cat=2)

            print("\n📎 참고한 문서 메타데이터:")
            for doc_content, meta in top_docs_with_meta: # 문서 내용도 함께 출력 (디버깅용)
                # print(f" - (내용 일부: {doc_content[:50]}...) 메타데이터: {meta}")
                print(f" - 메타데이터: {meta}")
        else:
            # 1) 축약어 그룹 치환 적용
            query = preprocess_query(query)

            print(f"query: {query}")

            # 2) 변환된 질의로 학과 키워드 추출
            major_filter_keyword = extract_major_keyword(query, unique_majors,threshold = 60)

            if major_filter_keyword:
                print(f"✨ '{major_filter_keyword}' 관련 정보로 필터링하여 검색합니다.")
            else:
                print("ℹ️ 특정 학과 키워드가 감지되지 않았습니다. 전체 문서에서 검색합니다.")
        

            # 3) 필터링 키워드를 retriever에 전달
            top_docs_with_meta = retriever.retrieve(query, top_k_bm25=10, top_k_dpr=3, filter_major=major_filter_keyword,cat=1)

            if not top_docs_with_meta:
                print("\n🧠 chatbot 응답:\n관련된 문서를 찾지 못했습니다. 다른 질문을 해주시거나 키워드를 확인해주세요.")
                continue # 다음 질문으로

            answer = generate_answer(query, top_docs_with_meta, cat=1)

            print("\n📎 참고한 문서 메타데이터:")
            for doc_content, meta in top_docs_with_meta: 
                print(f" - 메타데이터: {meta}")

        print(f"\n🧠 chatbot 응답:\n{answer}")