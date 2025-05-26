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
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import defaultdict

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
    
    # ✅ 유사한 항목 여러 개 추출
    '''matches = process.extract(normalized_query, candidates, limit=3)
    # ✅ 유사도 기준 통과한 학과만 반환
    result = []
    for match_str, score, _ in matches:
        idx = candidates.index(match_str)
        print(majors_list[idx])
        if score >= 80:
            idx = candidates.index(match_str)
            
            result.append(majors_list[idx])
            print(f"threshold보다 큰 major: {majors_list[idx]}")
        
        return result  # 최대 top_k개의 학과명 반환'''


    # "공지", "안내", "일정" 등 일반적인 키워드도 여기서 처리할 수 있습니다.
    # 예를 들어 '공지'라는 단어가 있으면, 문서 유형을 '공지'로 필터링하도록 설정
    # if "공지" in query:
    # return {"type": "notice"} # 이런 식으로 다른 필터링 기준도 추가 가능
    return None

# ✅ 하이브리드 검색기 초기화
class HybridRetriever:
    def __init__(self, corpus_all, metadatas_all, collection_name):
        # 초기화 시점에는 전체 데이터를 보관
        self.corpus_all = corpus_all
        self.metadatas_all = metadatas_all
        self.encoder = SentenceTransformer("jhgan/ko-sbert-sts")
        self.collection_name = collection_name
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
        
        bm25 = BM25Okapi(tokenized_corpus)
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
        for meta in bm25_candidates_meta:
            file = meta.get("source_file", "")
            university = meta.get("university", "")
            major = meta.get("major", "")
            bm25_candidate_ids.append(f"{file}_{university}_{major}")

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
        alpha = 0.5
        bm25_top_scores = np.array([bm25_scores[idx] for idx in bm25_indices_in_current])
        bm25_norm = (bm25_top_scores - np.min(bm25_top_scores)) / (np.max(bm25_top_scores) - np.min(bm25_top_scores) + 1e-8)
        dpr_norm = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities) + 1e-8)

        hybrid_scores = alpha * bm25_norm + (1 - alpha) * dpr_norm
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k_dpr]

        sorted_indices = np.argsort(hybrid_scores)[::-1]  # 내림차순 정렬 인덱스

        for rank, i in enumerate(sorted_indices):
            print(f"  [{rank}] 통합점수: {hybrid_scores[i]:.4f} | BM25: {bm25_norm[i]:.4f} | DPR: {dpr_norm[i]:.4f} | 메타: {bm25_candidates_meta[i]}")


        final_results = [
            (bm25_candidates_docs[i], bm25_candidates_meta[i], hybrid_scores[i]) for i in top_indices
        ]


        return final_results


def group_by_section(text):
    """
    'overview.0: ...' 형식의 텍스트를 section 단위로 묶어 반환
    """
    section_map = defaultdict(list)
    for line in text.strip().split('\n'):
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        section = key.split('.')[0]
        section_map[section].append(value.strip())
    return [" ".join(lines) for lines in section_map.values()]

# ✅ Cross-Encoder 로드 (한번만 실행)
cross_encoder_tokenizer = AutoTokenizer.from_pretrained("bm-k/KoSimCSE-roberta-multitask")
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained("bm-k/KoSimCSE-roberta-multitask")

# ✅ Cross-Encoder 기반 재랭킹 함수 (섹션 단위 분해 + Hybrid 점수 조합)
def rerank_with_crossencoder(query, docs_with_meta, hybrid_scores=None, top_k=3, alpha=0.7):
    expanded_pairs = []
    chunk_map = []

    for i, (doc, meta) in enumerate(docs_with_meta):
        chunks = group_by_section(doc) # 섹션 단위 처리
        for chunk in chunks:
            expanded_pairs.append((query, chunk))
            chunk_map.append((i, chunk))

    inputs = cross_encoder_tokenizer.batch_encode_plus(
        expanded_pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = cross_encoder_model(**inputs)
        logits = outputs.logits.squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        ce_scores = [s[0] if isinstance(s, list) else s for s in torch.sigmoid(logits).tolist()]

    # 문장 score들을 문서 단위로 다시 합산 → 문서별 최고 점수만 사용
    doc_scores = {}  # doc_idx -> 최고 점수
    doc_best_chunk = {}  # doc_idx -> 최고 점수 chunk (섹션 텍스트)

    for (doc_idx, chunk), score in zip(chunk_map, ce_scores):
        if doc_idx not in doc_scores or score > doc_scores[doc_idx]:
            doc_scores[doc_idx] = score
            doc_best_chunk[doc_idx] = chunk  # ✅ 최고 점수 받은 섹션 저장


    final_scores = []
    for i in range(len(docs_with_meta)):
        ce_score = doc_scores.get(i, 0)
        '''
        if hybrid_scores:
            score = (1 - alpha) * ce_score + alpha * hybrid_scores[i]  # ✅ Hybrid + CE 점수 혼합
        else:
            score = ce_score
        '''
        score = ce_score
        final_scores.append((docs_with_meta[i], score))

    reranked = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_k]

    print("\n🔁 절 단위 Cross-Encoder + Hybrid 점수 재랭킹 결과:")
    for i, ((doc, meta), score) in enumerate(reranked):
        print(f"  🔝 최고 점수 받은 섹션 내용 (문서 {i}):\n{doc_best_chunk.get(i, '[섹션 없음]')}\n")
        print(f"  [{i}] 최종 점수: {score:.4f} | 메타: {meta}\n")

    return [doc for doc, _ in reranked]

def compute_hybrid_scores(bm25_scores, dpr_scores, alpha=0.5):
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    dpr_norm = (dpr_scores - np.min(dpr_scores)) / (np.max(dpr_scores) - np.min(dpr_scores) + 1e-8)
    return alpha * bm25_norm + (1 - alpha) * dpr_norm



# ✅ GPT 응답 생성기
def generate_answer(query, context_docs):
    if not context_docs: # 참고 문서가 없는 경우
        return "관련 정보를 찾지 못했습니다. 질문을 조금 더 구체적으로 해주시거나 다른 키워드로 시도해주세요."

    context = "\n\n".join([
        f"[{meta.get('university', 'Unknown University')} - {meta.get('major', 'Unknown Major')} - {meta.get('source_file', 'Unknown Source')}]\n{doc}"
        for doc, meta in context_docs
    ])

    prompt = (
        "당신은 서강대학교의 학사 요람과 공지사항에 기반하여 정확하고 간결하게 답변해야 합니다.\n"
        "질문이 모호하더라도 관련 학과 또는 규정 문서를 참고하여 정확하게 답변하세요.\n"
        "같은 과목에 대한 설명이 여러 학과의 문서에 나뉘어 있을 경우, 해당 학과에서의 설명을 모두 넣어 주세요.\n"
        "중복된 내용은 각각의 문맥에서 필요한 경우 반복해도 됩니다.\n"
        "각 학과별의 설명은 분리된 문단으로 출력해주세요.\n"
    )



    messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": f"context:\n{context}\n\n질문: {query}\n답변:"}
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
            max_tokens=600, # 답변 토큰 수 제한
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


def preprocess_query(query):
    for full_name, abbr_list in ABBREVIATION_GROUPS.items():
        for abbr in abbr_list:
            if abbr in query:
                query = query.replace(abbr, full_name)
    return query

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
            collection_name = col_name
        )

    # ✅ major 목록도 함께 저장
    majors_by_collection = {
        col_name: content["majors"] for col_name, content in collection_data.items()
    }

    # ✅ 카테고리 → 컬렉션 이름 매핑
    category_to_collection = {
        "1": "collection_course",
        "2": "collection_subjectinfo"
    }

    #category 초기화
    print("어떤 카테고리의 질문을 할지 골라주세요.")
    cat = input("\n1. 과목/전공 이수 요건 2. 과목 정보\n-> ")

    if cat not in category_to_collection:
        print("⚠️ 잘못된 입력입니다. 1 또는 2를 입력하세요.")
    
    selected_collection = category_to_collection[cat]

    if selected_collection not in retrievers:
        print(f"❌ 선택한 컬렉션 '{selected_collection}'이 로드되지 않았습니다.")

    retriever = retrievers[selected_collection]
    unique_majors = majors_by_collection[selected_collection]


    while True:
        query = input("\n❓ 질문을 입력하세요 (종료를 원하면 exit을, category 변경을 원하면 cat을 입력해주세요.): ")
        if query.lower().strip() == "exit":
            print("🚫챗봇을 종료합니다.\n")
            break
        elif query.lower().strip() == "cat":
            print("어떤 카테고리의 질문을 할지 골라주세요.")
            cat = input("\n1. 과목/전공 이수 요건 2. 과목 정보\n-> ")

            if cat not in category_to_collection:
                print("⚠️ 잘못된 입력입니다. 1 또는 2를 입력하세요.")
                continue
    
            selected_collection = category_to_collection[cat]

            if selected_collection not in retrievers:
                print(f"❌ 선택한 컬렉션 '{selected_collection}'이 로드되지 않았습니다.")

            retriever = retrievers[selected_collection]
            unique_majors = majors_by_collection[selected_collection]

            query = input("\n❓ 질문을 입력하세요 (종료를 원하면 exit을, category 변경을 원하면 cat을을 입력해주세요.): ")

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
            print("\n🧠 chatbot 응답:\n관련된 문서를 찾지 못했습니다. 다른 질문을 해주시거나 키워드를 확인해주세요.")
            continue # 다음 질문으로

        # ➕ 문서/메타데이터/하이브리드 점수 분리
        docs_with_meta = [r[:2] for r in top_docs_with_meta]
        hybrid_scores = [r[2] for r in top_docs_with_meta]

        # ✅ Cross-Encoder rerank 적용
        top_docs_with_meta = rerank_with_crossencoder(query, docs_with_meta, hybrid_scores, top_k=3)

        #answer = generate_answer(query, top_docs_with_meta)
        #print(f"\n🧠 chatbot 응답:\n{answer}")

        print("\n📎 참고한 문서 메타데이터:")
        for doc_content, meta in top_docs_with_meta: # 문서 내용도 함께 출력 (디버깅용)
            # print(f" - (내용 일부: {doc_content[:50]}...) 메타데이터: {meta}")
            print(f" - 메타데이터: {meta}")