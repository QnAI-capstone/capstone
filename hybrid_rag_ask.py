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
from dictionary import ABBREVIATION_GROUPS

# ✅ API 키
openai.api_key = OPENAI_API_KEY

# ✅ 한국어 임베딩 함수
class KoSimCSEEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("jhgan/ko-sbert-sts")
    def __call__(self, input):
        return self.model.encode(input).tolist()

# ✅ ChromaDB 초기화 및 문서+메타데이터 불러오기
def load_corpus():
    client = PersistentClient(path="./chroma_store")
    embedding_fn = KoSimCSEEmbedding()
    collections_info = client.list_collections() # get_collection 대신 list_collections 사용
    print(f"총 {len(collections_info)}개의 컬렉션을 불러옵니다.")

    corpus, metadatas_list = [], []
    unique_majors = set() # 고유 학과명 저장을 위한 set

    for col_info in collections_info:
        collection = client.get_collection(name=col_info.name, embedding_function=embedding_fn)
        data = collection.get(include=["documents", "metadatas"])
        corpus.extend(data["documents"])
        metadatas_list.extend(data["metadatas"])
        for meta in data["metadatas"]:
            if meta and 'major' in meta: # 메타데이터 및 'major' 키 존재 확인
                unique_majors.add(meta['major'])

    return corpus, metadatas_list, list(unique_majors) # 고유 학과명 리스트 반환

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
def extract_major_keyword(query, majors_list):
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

    # 유사도 기준 설정 (예: 80 이상일 때만 매칭 인정)
    if best_match and best_match[1] > 80:
        idx = candidates.index(best_match[0])
        return majors_list[idx]


    # "공지", "안내", "일정" 등 일반적인 키워드도 여기서 처리할 수 있습니다.
    # 예를 들어 '공지'라는 단어가 있으면, 문서 유형을 '공지'로 필터링하도록 설정
    # if "공지" in query:
    # return {"type": "notice"} # 이런 식으로 다른 필터링 기준도 추가 가능

    # 매칭되는 학과명이 없으면 None 반환
    return None


# ✅ 하이브리드 검색기 초기화
class HybridRetriever:
    def __init__(self, corpus_all, metadatas_all):
        # 초기화 시점에는 전체 데이터를 보관
        self.corpus_all = corpus_all
        self.metadatas_all = metadatas_all
        self.encoder = SentenceTransformer("jhgan/ko-sbert-sts")
        # 전체 문서에 대한 임베딩을 미리 계산해둘 수 있으나, 필터링 시 메모리 사용량 고려
        # self.dense_embeddings_all = self.encoder.encode(self.corpus_all) # 필요 시 활성화

    def retrieve(self, query, top_k_bm25=10, top_k_dpr=3, filter_major=None):
        # 실제 검색 대상이 될 코퍼스와 메타데이터
        current_corpus = self.corpus_all
        current_metadatas = self.metadatas_all

        # 필터링할 학과가 지정된 경우
        if filter_major:
            print(f"🔍 '{filter_major}' 학과 관련 문서로 필터링 중...")
            filtered_indices = [
                i for i, meta in enumerate(self.metadatas_all)
                if meta and meta.get('major') == filter_major
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
        if not tokenized_corpus: # 토큰화된 코퍼스가 비어있는 경우 처리
            print("⚠️ 토큰화된 문서가 없어 BM25 검색을 수행할 수 없습니다.")
            return []
        
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # BM25 결과가 top_k_bm25보다 적을 경우를 대비
        num_bm25_candidates = min(top_k_bm25, len(current_corpus))
        bm25_indices_in_current = np.argsort(bm25_scores)[::-1][:num_bm25_candidates]

        # 현재 코퍼스(필터링되었을 수 있음) 내에서의 인덱스이므로,
        # 원래 코퍼스에서의 인덱스로 변환할 필요는 없음.
        # 바로 현재 코퍼스에서 해당 문서와 메타데이터를 가져옴.
        bm25_candidates_docs = [current_corpus[i] for i in bm25_indices_in_current]
        bm25_candidates_meta = [current_metadatas[i] for i in bm25_indices_in_current]

        if not bm25_candidates_docs:
            print("⚠️ BM25 검색 결과가 없습니다.")
            return []

        # Dense retrieval (의미론적 검색)
        # 필터링된 문서들의 임베딩을 사용하거나, 필요시 즉석에서 계산
        # 전체 문서 임베딩을 미리 계산해두고 필터링된 인덱스로 가져오는 방법도 있음
        # self.dense_embeddings_all 사용 시:
        # if filter_major and filtered_indices:
        #     candidate_embeddings = self.dense_embeddings_all[filtered_indices_for_dense] # 주의: 인덱스 매칭 필요
        # else:
        #     candidate_embeddings = self.encoder.encode(bm25_candidates_docs) # 현재 방식: BM25 결과에 대해서만 인코딩
        
        candidate_embeddings = self.encoder.encode(bm25_candidates_docs)
        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # DPR 결과가 top_k_dpr보다 적을 경우를 대비
        num_dpr_candidates = min(top_k_dpr, len(bm25_candidates_docs))
        top_indices_in_bm25 = np.argsort(similarities)[::-1][:num_dpr_candidates]

        final_results = [
            (bm25_candidates_docs[i], bm25_candidates_meta[i]) for i in top_indices_in_bm25
        ]
        return final_results

# ✅ GPT 응답 생성기
def generate_answer(query, context_docs):
    if not context_docs: # 참고 문서가 없는 경우
        return "관련 정보를 찾지 못했습니다. 질문을 조금 더 구체적으로 해주시거나 다른 키워드로 시도해주세요."

    context = "\n\n".join([doc for doc, meta in context_docs])

    messages = [
        {"role": "system", "content": "다음 내용을 참고하여 질문에 답변하세요. 다음 텍스트는 2025년도 서강대학교 학사요람에서 추출한 각 학과 및 과목별 정보입니다. 만약 내용이 충분하지 않다면, 아는 선에서 최대한 답변하거나 추가 정보가 필요하다고 언급하세요."},
        {"role": "user", "content": f"{context}\n\n질문: {query}\n답변:"}
    ]

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
            model="gpt-4o",
            messages=messages,
            max_tokens=max_response_tokens # 답변 토큰 수 제한
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


def preprocess_query(query):
    for full_name, abbr_list in ABBREVIATION_GROUPS.items():
        for abbr in abbr_list:
            if abbr in query:
                query = query.replace(abbr, full_name)
    return query

# ✅ 메인 실행 루프
if __name__ == "__main__":
    print("💬 학사요람 기반 RAG 시스템 시작됨. 질문을 입력하세요 (종료: 'exit')")

    corpus, metadatas, unique_majors = load_corpus()
    if not corpus:
        print("⚠️ 로드된 문서가 없습니다. DB를 먼저 생성하거나 경로를 확인해주세요.")
        exit()
        
    print(f"📚 총 {len(unique_majors)}개의 학과 정보 로드됨: {unique_majors[:10]} 등") # 처음 10개 학과만 출력
    retriever = HybridRetriever(corpus, metadatas)

    while True:
        query = input("\n❓ 질문: ")
        if query.lower().strip() == "exit":
            break

        # 1) 축약어 그룹 치환 적용
        query = preprocess_query(query)

        # 2) 변환된 질의로 학과 키워드 추출
        major_filter_keyword = extract_major_keyword(query, unique_majors)

        if major_filter_keyword:
            print(f"✨ '{major_filter_keyword}' 관련 정보로 필터링하여 검색합니다.")
        else:
            print("ℹ️ 특정 학과 키워드가 감지되지 않았습니다. 전체 문서에서 검색합니다.")

        # 3) 필터링 키워드를 retriever에 전달
        top_docs_with_meta = retriever.retrieve(query, top_k_bm25=10, top_k_dpr=3, filter_major=major_filter_keyword)

        if not top_docs_with_meta:
            print("\n🧠 GPT 응답:\n관련된 문서를 찾지 못했습니다. 다른 질문을 해주시거나 키워드를 확인해주세요.")
            continue # 다음 질문으로

        answer = generate_answer(query, top_docs_with_meta)
        print(f"\n🧠 GPT 응답:\n{answer}")

        print("\n📎 참고한 문서 메타데이터:")
        for doc_content, meta in top_docs_with_meta: # 문서 내용도 함께 출력 (디버깅용)
            # print(f" - (내용 일부: {doc_content[:50]}...) 메타데이터: {meta}")
            print(f" - 메타데이터: {meta}")