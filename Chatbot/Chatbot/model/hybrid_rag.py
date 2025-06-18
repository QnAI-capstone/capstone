import os
import sys
import django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Chatbot.settings')
django.setup()
from django.conf import settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_STORE_PATH = os.path.join(BASE_DIR, 'chroma_store')

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
from scipy.stats import rankdata
from Chatbot.model.dictionary import ABBREVIATION_GROUPS,DATE_GROUPS
from Chatbot.model.ex_sub import extract_subject_by_rapidfuzz
from django.contrib.auth.models import User

# ✅ API 키
openai.api_key = settings.OPENAI_API_KEY

# ✅ 한국어 임베딩 함수
class KoSimCSEEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    def __call__(self, input):
        return self.model.encode(input).tolist()

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
            print(f"✅ 'collection_notice' 불러오기 완료. 문서 수: {len(documents)}")

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
def extract_major_keyword(query, majors_list, threshold=70):
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
    def __init__(self, corpus_all, metadatas_all, collection_name):
        # 초기화 시점에는 전체 데이터를 보관
        self.corpus_all = corpus_all
        self.metadatas_all = metadatas_all
        self.encoder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        self.collection_name = collection_name
        # 전체 문서에 대한 임베딩을 미리 계산해둘 수 있으나, 필터링 시 메모리 사용량 고려
        # self.dense_embeddings_all = self.encoder.encode(self.corpus_all) # 필요 시 활성화

    def retrieve(self, query, top_k_bm25=10, top_k_dpr=3, filter_major=None, alpha=0.5, cat=1):
        print("🟢 문서를 retrieve 함수를 활용하여 검색합니다.")
        # 실제 검색 대상이 될 코퍼스와 메타데이터
        current_corpus = self.corpus_all
        current_metadatas = self.metadatas_all
        final_results = []
        query_bm25 = query

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

        ###추가
        if cat == 2:
            sub = extract_subject_by_rapidfuzz(query)
            if sub:
                query_bm25 = query.strip()+" "+sub[0]+" 과목"
        
        print(f"query bm25 : {query_bm25}")
        ###

        # BM25 계산 (필터링된 코퍼스 또는 전체 코퍼스 대상)
        tokenized_corpus = [doc.split() for doc in current_corpus]
        
        bm25 = BM25Okapi(tokenized_corpus,b=0.25)
        tokenized_query = query_bm25.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # BM25 결과가 top_k_bm25보다 적을 경우를 대비
        num_bm25_candidates = min(top_k_bm25, len(current_corpus))
        bm25_indices_in_current = np.argsort(bm25_scores)[::-1][:num_bm25_candidates]
        bm25_candidates_docs = [current_corpus[i] for i in bm25_indices_in_current]
        bm25_candidates_meta = [current_metadatas[i] for i in bm25_indices_in_current]

        # -------------------------------
        # 2️⃣ cat = 2인 경우에만 ChromaDB 벡터 불러와 의미 유사도 계산
        # -------------------------------
        
        if cat == 2:
            embedding_fn = KoSimCSEEmbedding()
            client = PersistentClient(path="./chroma_store")
            collection = client.get_collection(name=self.collection_name, embedding_function=embedding_fn)

            retrieved = collection.get(include=["embeddings", "metadatas", "documents"])
            all_embeddings = retrieved["embeddings"]
            all_metadatas = retrieved["metadatas"]
            all_documents = retrieved["documents"]

            # 질의 임베딩
            query_embedding = self.encoder.encode([query])  # (1, dim)

            # 코사인 유사도 계산 (전체 문서 대상)
            similarities = cosine_similarity(query_embedding, all_embeddings)[0]  # (num_docs,)

            # 4️⃣ 상위 top_k_dpr 개수만큼 추출
            dpr_candidate = min(top_k_dpr, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:dpr_candidate]

            # 중복 제거된 DPR 결과
            dpr_indices_unique = [i for i in top_indices if all_documents[i] not in bm25_candidates_docs]

            # 최대 3개씩 BM25/DPR 교차 배치
            for i in range(3):
                if i < len(bm25_indices_in_current):
                    doc = bm25_candidates_docs[i]
                    meta = bm25_candidates_meta[i]
                    final_results.append((doc, meta))
                if i < len(dpr_indices_unique):
                    dpr_doc = all_documents[dpr_indices_unique[i]]
                    dpr_meta = all_metadatas[dpr_indices_unique[i]]
                    final_results.append((dpr_doc, dpr_meta))
            
        else:
            final_results = [
                (bm25_candidates_docs[i], bm25_candidates_meta[i]) for i in bm25_indices_in_current[:num_bm25_candidates]
            ]

        return final_results

# ✅ GPT 응답 생성기
def generate_answer(query, context_docs, log, cat):
    if not context_docs: # 참고 문서가 없는 경우
        return "관련 정보를 찾지 못했습니다. 질문을 조금 더 구체적으로 해주시거나 다른 키워드로 시도해주세요."

    if cat == 1: # collection_course
        context = "\n\n".join([
            f"[{meta.get('university', 'Unknown University')} - {meta.get('major', 'Unknown Major')} - {meta.get('source_file', 'Unknown Source')}]\n{doc}"
            for doc, meta in context_docs
        ])

        prompt = (
            "당신은 서강대학교의 학사 요람에 기반하여 정확하고 간결하게 답변해야 합니다.\n"
            "질문이 모호하더라도 관련 학과 또는 규정 문서를 참고하여 정확하게 답변하세요.\n"
            "이전 대화 내용이 제공된 경우 해당 내용을 참고하여 **대화의 맥락을 유지**하세요.\n"
            "사용자가 이전 응답을 이어서 질문할 경우(예: '그중에서', '그러면', '이전에 말한 것 중'), 직전의 질문과 모델의 응답 내용을 함께 참고하여 일관된 맥락 속에서 답변하세요. 이전 질문/응답은 시스템이 메시지 히스토리로 제공합니다.\n"
            "사용자는 아래의 세 가지 전공 유형 중 하나에 해당할 수 있습니다. 이 구분은 모든 학과에 동일하게 적용되며, 어떤 전공이 주 전공인지에 따라 학과별 졸업 요건 및 이수 기준이 달라질 수 있습니다:\n"
            
            "1. **단일전공**: 사용자는 특정 학과(예: 컴퓨터공학과)만 전공합니다.\n"
            "2. **다전공(자신의 학과)**: 사용자는 해당 학과를 제1전공으로 하고, 다른 학과를 복수전공합니다.\n"
            "3. **다전공(타 학과)**: 사용자는 다른 학과를 제1전공으로 하고, 해당 학과를 복수전공합니다.\n"

            "예시) 질문이 컴퓨터공학과에 대한 것일 경우:\n"
            "- \"단일전공\" 사용자는 컴퓨터공학과만 전공\n"
            "- \"다전공(컴공)\" 사용자는 컴퓨터공학과가 제1전공 + 다른 학과 복수전공\n"
            "- \"다전공(타전공)\" 사용자는 다른 학과가 제1전공 + 컴퓨터공학과 복수전공\n"

            "질문이 어느 전공 유형에 해당하는지 명확하지 않더라도, 각 경우에 따라 달라지는 내용을 **모두 분리된 문단**으로 나눠 설명하세요.\n"
            "- 제목, 소제목, 리스트 등을 적절히 활용하여 알아보기 좋게 정리하세요.\n"
            "제공된 context에서 찾을 수 없다면 찾을 수 없다고 메시지를 출력해주세요.\n"
        )

        messages = [{"role": "system", "content": prompt}]
        check_log(log, messages)
        messages.append({"role": "user", "content": f"context:\n{context}\n\n질문: {query}\n답변:"})
        model_name = "gpt-4o"
    
    elif cat == 2: # collection_subjectinfo
        context = "\n\n".join([
            f"[{meta.get('university', 'Unknown University')} - {meta.get('major', 'Unknown Major')} - {meta.get('source_file', 'Unknown Source')}]\n{doc}"
            for doc, meta in context_docs
        ])

        prompt = (
            "당신은 서강대학교의 학사 요람 정보를 기반으로, 사용자 질문에 대해 정확하고 간결하게 답변해야 합니다.\n"
            "질문이 모호하더라도, 관련 학과 또는 규정 문서를 모두 참고하여 가능한 모든 정보를 포함하세요.\n"
            "이전 대화 내용이 제공된 경우 해당 내용을 참고하여 **대화의 맥락을 유지**하세요.\n"
            "사용자가 이전 응답을 이어서 질문할 경우(예: '그중에서', '그러면', '이전에 말한 것 중'), 직전의 질문과 모델의 응답 내용을 함께 참고하여 일관된 맥락 속에서 답변하세요. 이전 질문/응답은 시스템이 메시지 히스토리로 제공합니다.\n"
            "같은 과목에 대한 설명이 여러 학과 또는 전공에서 반복될 경우, **모든 관련 문서에서 나온 설명을 빠짐없이 포함**하세요.\n"
            "각각의 설명은 **출처 학과명 기준으로 문단을 분리하여 출력**하고, 중복된 내용이 있더라도 **학과 문맥 내에서는 생략하지 말고 모두 출력**하세요.\n"
            "요약하지 마세요. **모든 학과별 설명을 전부 나열**하는 것이 중요합니다.\n"
            "제목, 소제목, 리스트 등을 적절히 활용하여 알아보기 좋게 정리하세요.\n"
            "제공된 context에서 답변을 찾을 수 없을 경우, \"제공된 정보에서 답변을 찾을 수 없습니다.\"라고 출력하세요.\n"
        )

        messages = [{"role": "system", "content": prompt}]
        check_log(log, messages)
        messages.append({"role": "user", "content": f"context:\n{context}\n\n질문: {query}\n답변:"})
        model_name="gpt-4o"

    else: # collection_notice        
        context = context_docs
        prompt = (
            "당신은 서강대학교의 공지사항 데이터를 기반으로 질문에 정확하고 간결하게 답변하는 어시스턴트입니다.\n"
            "이전 대화 내용이 제공된 경우 해당 내용을 참고하여 **대화의 맥락을 유지**하세요.\n"
            "사용자가 이전 응답을 이어서 질문할 경우(예: '그중에서', '그러면', '이전에 말한 것 중'), 직전의 질문과 모델의 응답 내용을 함께 참고하여 일관된 맥락 속에서 답변하세요. 이전 질문/응답은 시스템이 메시지 히스토리로 제공합니다.\n"
            "질문이 모호하더라도, 제공된 공지 context를 바탕으로 규정과 사실에 근거해 답변해야 합니다.\n"
            "가능한 한 질문과 키워드가 정확히 일치하는 공지를 찾아서 제시하세요.\n"
            "여러 개의 공지가 관련 있다면, 날짜(date)가 가장 최신인 순서로 정렬하여 출력하세요.\n"
            "제공된 context에 관련 정보가 없다면, '관련 공지를 찾을 수 없습니다.'라고 답변하세요.\n"
            "링크는 반드시 한 번만 출력하고, 마크다운 문법을 사용하지 말고 순수한 URL만 출력하세요.\n\n"
        )

        messages = [{"role": "system", "content": prompt}]
        check_log(log, messages)
        messages.extend([
            # 🟡 One-shot 예시
            {"role": "user", "content": "context:\n[졸업] 2023학년도 후기(2024년 8월) 졸업_학위증 배부 및 학위가운 대여 안내|2024.07.30|https://sogang.ac.kr/ko/detail/\n\n질문: 학위 가운은 어디서 대여할 수 있어?\n답변:"},
            {"role": "assistant", "content": "학위 가운 대여와 관련하여 다음 공지를 참조하세요.\n제목:[졸업] 2023학년도 후기(2024년 8월) 졸업_학위증 배부 및 학위가운 대여 안내\n업로드일자: 2024.07.30\n링크:https://sogang.ac.kr/ko/detail/\n"},
            {"role": "user", "content": f"context:\n{context}\n\n질문: {query}\n답변:"}
        ])
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
            model = model_name,
            messages = messages,
            max_tokens = 4096, # 답변 토큰 수 제한
            temperature = 0.3,
            top_p = 0.9
        )
        return response.choices[0].message['content']
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
KOREAN_PARTICLE_PATTERN = r'(은|는|이|가|을|를|에|에서|으로|로|도|만|까지|부터|조차|인데|고|와|과|마저|처럼|께서|밖에|이며|이고|이나|라도|라고|라는|든지|만큼)?'

def preprocess_query(query):
    used_majors = []
    replaced_ranges = []  # 이미 치환된 텍스트의 위치를 저장

    for full_name, variants in ABBREVIATION_GROUPS.items():
        for variant in sorted(variants, key=lambda x: -len(x)):
            # 변형어가 조사와 함께 붙은 형태로 끝날 때도 매칭
            pattern = re.compile(rf'(?<!\w)({re.escape(variant)}){KOREAN_PARTICLE_PATTERN}')

            def replacer(m):
                start, end = m.start(1), m.end(1)  # variant만큼의 범위로 비교
                for r_start, r_end in replaced_ranges:
                    if not (end <= r_start or start >= r_end):
                        return m.group(0)  # 기존 텍스트 그대로 반환

                replaced_ranges.append((start, start + len(full_name)))
                if full_name not in used_majors:
                    used_majors.append(full_name)
                return full_name + (m.group(2) or "")
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

def extract_relate_query(query, context_docs):
    prompt = (
    "당신은 서강대학교의 학사 요람 및 공지사항 데이터를 기반으로 작동하는 AI 챗봇입니다.\n"
    "- 아래에 주어진 context는 과거 공지사항 및 학사 요람의 일부입니다.\n"
    "- 사용자로부터의 query는 그 다음에 주어집니다.\n"
    "- 이 query와 관련하여 사용자가 후속으로 궁금해할 수 있는 자연스러운 질문 3가지를 한국어로 제안하세요.\n"
    "- 주어진 context의 범위를 벗어나는 질문은 절대 포함하지 마세요.\n"
    "- 질문은 간결하고 명확하게 작성하며, 리스트 형식으로 출력하세요.\n\n"
    )

    messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"query: {query}\n context:{context_docs}\n 추천 질문:"}
        ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4096, # 답변 토큰 수 제한
            temperature = 0.3,
            top_p = 0.9
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"❌ 추천 질문 생성 중 오류: {e}")
        return "추천 질문을 생성할 수 없습니다."

def auto_linkify(text):
    if not isinstance(text, str):
        return text
    url_pattern = re.compile(r'(https?://[^\s]+)')
    return url_pattern.sub(r'<a href="\1" target="_blank">링크</a>', text)

def check_log(log, messages):
    if not log:
        return
    recent_conversation = log[-6:]
    print("현재 log 내용:")
    for i, m in enumerate(log):
        print(f"{i}: role={m['role']}, content={m['content']}")
    for i, msg in enumerate(recent_conversation):
        messages.append(msg)

# ✅ 카테고리 → 컬렉션 이름 매핑
category_to_collection = {
    "1": "collection_course",
    "2": "collection_subjectinfo",
    "3": "collection_notice"
}

def get_categories():
    return {
        "1": "과목/전공 이수 요건",
        "2": "과목 정보",
        "3": "학사 공지"
}

retrievers = {}
majors_by_collection = {}
collection_data = load_corpus_by_collection()
top_docs_with_meta = None


def initialize_rag():
    print("💬 학사요람 기반 RAG 시스템 시작됨.")
    global retrievers, majors_by_collection, collection_data
    if not collection_data:
        print("⚠️ 로드된 문서가 없습니다. DB를 먼저 생성하거나 경로를 확인해주세요.")
        exit()
        
    # ✅ 컬렉션별 retriever 초기화
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

def initialize_cat():
    global top_docs_with_meta

    top_docs_with_meta = None


# 검색기에서 데이터를 추출하는 함수
def get_response_from_retriever(query: str, selected_collection: str, chat_log: list):
    global top_docs_with_meta
    
    if selected_collection not in retrievers:
        return {
            "answer": f"❌ 선택한 컬렉션 '{selected_collection}'이 로드되지 않았습니다.",
            "questions": []
        }
        exit()

    retriever = retrievers[selected_collection]
    context_text = ""
    answer = ""

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

        context_text = "\n\n".join(all_docs)
        answer=generate_answer(query, all_docs, chat_log, cat=3)

    elif selected_collection == "collection_subjectinfo":
        unique_majors = majors_by_collection[selected_collection]
        # 1) 축약어 그룹 치환 적용
        query = preprocess_query(query)

        # 2) 변환된 질의로 학과 키워드 추출
        major_filter_keyword = extract_major_keyword(query, unique_majors,threshold = 80)

        if major_filter_keyword:
            print(f"✨ '{major_filter_keyword}' 관련 정보로 필터링하여 검색합니다.")
            # 3) 필터링 키워드를 retriever에 전달
            top_docs_with_meta = retriever.retrieve(query, top_k_bm25=3, top_k_dpr=3, filter_major=major_filter_keyword,alpha=0.5,cat= 2)
        else:
            if not top_docs_with_meta:
                print("ℹ️ 특정 학과 키워드가 감지되지 않았습니다. 전체 문서에서 검색합니다.")
                # 3) 필터링 키워드를 retriever에 전달
                top_docs_with_meta = retriever.retrieve(query, top_k_bm25=3, top_k_dpr=3, filter_major=major_filter_keyword,alpha=0.5,cat= 2)
            elif extract_subject_by_rapidfuzz(query):
                print("ℹ️ 특정 과목 키워드가 감지되었습니다. context를 새로 검색합니다.")
                top_docs_with_meta = retriever.retrieve(query, top_k_bm25=3, top_k_dpr=3, filter_major=major_filter_keyword,alpha=0.5,cat= 2)
        
        print(f"query: {query}")

        if not top_docs_with_meta:
            print("\n🧠 chatbot 응답:\n관련된 문서를 찾지 못했습니다. 다른 질문을 하거나 키워드를 확인해주세요.")

        context_text = "\n\n".join([doc for doc, _ in top_docs_with_meta])
        answer = generate_answer(query, top_docs_with_meta, chat_log, cat=2)

        print("\n📎 참고한 문서 메타데이터:")
        for doc_content, meta in top_docs_with_meta: # 문서 내용도 함께 출력 (디버깅용)
            # print(f" - (내용 일부: {doc_content[:50]}...) 메타데이터: {meta}")
            print(f" - 메타데이터: {meta}")

    else:
        unique_majors = majors_by_collection[selected_collection]
        # 1) 축약어 그룹 치환 적용
        query = preprocess_query(query)

        print(f"query: {query}")

        # 2) 변환된 질의로 학과 키워드 추출
        major_filter_keyword = extract_major_keyword(query, unique_majors,threshold = 60)

        #수정!!!!!!
        if major_filter_keyword:
            print(f"✨ '{major_filter_keyword}' 관련 정보로 필터링하여 검색합니다.")
            # 3) 필터링 키워드를 retriever에 전달
            top_docs_with_meta = retriever.retrieve(query, top_k_bm25=10, top_k_dpr=3, filter_major=major_filter_keyword,cat=1)
        else:
            if not top_docs_with_meta:
                print("ℹ️ 특정 학과 키워드가 감지되지 않았습니다. 전체 문서에서 검색합니다.")
            else:
                print("ℹ️ 같은 문서를 context로 제공합니다.")
        #여기까지 수정

        if not top_docs_with_meta:
            print("\n🧠 chatbot 응답:\n관련된 문서를 찾지 못했습니다. 다른 질문을 하거나 키워드를 확인해주세요.")

        context_text = "\n\n".join([doc for doc, _ in top_docs_with_meta])
        answer = generate_answer(query, top_docs_with_meta, chat_log, cat=1)

        print("\n📎 참고한 문서 메타데이터:")
        if top_docs_with_meta is None or not top_docs_with_meta:
            print("No documents retrieved.")
        else:
            for doc_content, meta in top_docs_with_meta: 
                print(f" - 메타데이터: {meta}")

    suggested_questions_text = extract_relate_query(query, context_text)
    # 리스트로 분리 (숫자 bullet 제거 + strip)
    suggested_question_list = [
        line.lstrip("1234567890.●•- ").strip()
        for line in suggested_questions_text.strip().splitlines()
        if line.strip()
    ]
    re.sub(r'^\d+\.\s*', '', answer, flags=re.MULTILINE)

    return {
        "answer": answer,
        "questions": suggested_question_list
    }