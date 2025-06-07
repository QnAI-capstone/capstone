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
import time
from konlpy.tag import Okt
import pandas as pd
from hybrid_rag import load_corpus_by_collection, extract_major_keyword, preprocess_query, KoSimCSEEmbedding,count_total_tokens
from openpyxl import Workbook

# ✅ API 키
openai.api_key = OPENAI_API_KEY
# ✅ GPT 응답 생성기
def generate_answer(query, context_docs):
    
    if not context_docs: # 참고 문서가 없는 경우
        return "관련 정보를 찾지 못했습니다. 질문을 조금 더 구체적으로 해주시거나 다른 키워드로 시도해주세요."
    
    
    context = context_docs

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
    model_name="gpt-4o"


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

# ✅ 하이브리드 검색기 초기화
class HybridRetriever:
    def __init__(self, corpus_all, metadatas_all,collection_name):
        # 초기화 시점에는 전체 데이터를 보관
        self.corpus_all = corpus_all
        self.metadatas_all = metadatas_all
        self.encoder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        self.collection_name = collection_name

    def retrieve(self, query, top_k = 10, filter_major=None,alpha=0.5,cat=1):
        bm25_rows, dpr_rows = [], []
        hybrid_rows = []
        flag = 0
        query_bm25 = query
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
            flag = 1
            if not filtered_indices:
                print(f"⚠️ '{filter_major}' 학과 관련 문서를 찾을 수 없습니다. 전체 문서에서 검색합니다.")
            else:
                current_corpus = [self.corpus_all[i] for i in filtered_indices]
                current_metadatas = [self.metadatas_all[i] for i in filtered_indices]
                print(f"🔎 필터링 결과: 총 {len(current_corpus)}개의 문서로 제한됨.")


        if not current_corpus: # 필터링 후 문서가 없을 경우
             print("⚠️ 검색할 문서가 없습니다.")
             return []
        
        if flag == 0:
            sub = extract_subject_by_rapidfuzz(query)
            query_bm25 = query.strip()+" "+sub[0]+" 과목"

        # BM25 계산 (필터링된 코퍼스 또는 전체 코퍼스 대상)
        tokenized_corpus = [doc.split() for doc in current_corpus]
        
        bm25 = BM25Okapi(tokenized_corpus,b=0.25)
        tokenized_query = query_bm25.split()
        print(query_bm25)
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)

        # BM25 결과가 top_k_bm25보다 적을 경우를 대비
        num_bm25_candidates = min(6, len(current_corpus))
        bm25_indices_in_current = np.argsort(bm25_scores)[::-1][:num_bm25_candidates]
        bm25_candidates_docs = [current_corpus[i] for i in bm25_indices_in_current]
        bm25_candidates_meta = [current_metadatas[i] for i in bm25_indices_in_current]

        doc_gpt_blocks = []  # 각 문서 블록을 담을 리스트

        for rank, i in enumerate(bm25_indices_in_current):
            bm25_rows.append({
                "rank": rank + 1,
                "score": bm25_norm[i],
                "meta": current_metadatas[i],
                "doc_full": current_corpus[i].replace("\n", " ")
            })

            # 문서 블록 포맷 생성 및 누적
            meta = current_metadatas[i]
            doc = current_corpus[i]
            print(doc[:50])
            block = f"[{meta.get('university', 'Unknown University')} - {meta.get('major', 'Unknown Major')} - {meta.get('source_file', 'Unknown Source')}]\n{doc}"
            doc_gpt_blocks.append(block)

        # 최종 GPT input으로 사용할 텍스트 구성
        doc_gpt = "\n\n".join(doc_gpt_blocks)


        answer = generate_answer(query, doc_gpt)
        print(f"bm25 answer: {answer}")

        bm25_rows.append(
            {
                "answer": answer
            }
        )

        # -------------------------------
        # 2️⃣ ChromaDB 벡터 불러와 의미 유사도 계산
        # -------------------------------
        
        embedding_fn = KoSimCSEEmbedding()
        client = PersistentClient(path="./chroma_store")
        collection = client.get_collection(name=self.collection_name, embedding_function=embedding_fn)

        retrieved = collection.get(include=["embeddings", "metadatas", "documents"])
        all_embeddings = retrieved["embeddings"]
        all_metadatas = retrieved["metadatas"]
        all_documents = retrieved["documents"]

        # 2️⃣ 질의 임베딩
        query_embedding = self.encoder.encode([query])  # (1, dim)

        # 3️⃣ 코사인 유사도 계산 (전체 문서 대상)
        similarities = cosine_similarity(query_embedding, all_embeddings)[0]  # (num_docs,)

        dpr_candidate = min(6, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:dpr_candidate]
        dpr_candidates_docs = [all_documents[i] for i in top_indices]
        dpr_candidates_meta = [all_metadatas[i] for i in top_indices]

        # 중복 제거된 DPR 결과
        dpr_indices_unique = [i for i in top_indices if all_documents[i] not in bm25_candidates_docs]

        doc_gpt_blocks = []  # 각 문서 블록을 담을 리스트

        for rank, i in enumerate(top_indices):
            dpr_rows.append({
                "rank": rank + 1,
                "score": similarities[i],
                "meta": all_metadatas[i],
                "doc_full": all_documents[i].replace("\n", " ")
            })

            # 문서 블록 포맷 생성 및 누적
            meta = all_metadatas[i]
            doc = all_documents[i]
            block = f"[{meta.get('university', 'Unknown University')} - {meta.get('major', 'Unknown Major')} - {meta.get('source_file', 'Unknown Source')}]\n{doc}"
            doc_gpt_blocks.append(block)

        # 최종 GPT input으로 사용할 텍스트 구성
        doc_gpt = "\n\n".join(doc_gpt_blocks)

        answer = generate_answer(query, doc_gpt)
        print(f"answer: {answer}")

        dpr_rows.append(
            {
                "answer": answer
            }
        )

        '''doc_gpt_blocks = []

        for i in range(3):
            if i < len(bm25_indices_in_current):
                doc = bm25_candidates_docs[i]
                meta = bm25_candidates_meta[i]
                hybrid_rows.append({"rank": len(hybrid_rows)+1, "doc": doc, "meta": meta})
                doc_gpt_blocks.append(f"[{meta.get('university', '')} - {meta.get('major', '')} - {meta.get('source_file', '')}]\n{doc}")

            if i < len(dpr_indices_unique):
                dpr_doc = all_documents[dpr_indices_unique[i]]
                dpr_meta = all_metadatas[dpr_indices_unique[i]]
                hybrid_rows.append({"rank": len(hybrid_rows)+1, "doc": dpr_doc, "meta": dpr_meta})
                doc_gpt_blocks.append(f"[{dpr_meta.get('university', '')} - {dpr_meta.get('major', '')} - {dpr_meta.get('source_file', '')}]\n{dpr_doc}")

        doc_gpt = "\n\n".join(doc_gpt_blocks)
        answer = generate_answer(query, doc_gpt)
        print(f"answer: {answer}")

        hybrid_rows.append({"answer": answer})'''


        return bm25_rows,dpr_rows

print("💬 학사요람 기반 RAG 시스템 시작됨.")

collection_data = load_corpus_by_collection()

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
    "3": "collection_notice_md"
}

def overall_logic(query, cat):
    selected_collection = category_to_collection[cat]

    retriever = retrievers[selected_collection]

    if selected_collection != "collection_notice_md":
        unique_majors = majors_by_collection[selected_collection]

    if selected_collection == "collection_notice_md":
        #시기 추출
        date_keyword = extract_date_key_from_query(query)
            

        # 기본값 설정
        if date_keyword is None:
            print("ℹ️ 특정 시기 키워드가 감지되지 않았습니다. '2025-1' 문서만 검색합니다.")
            date_keyword = "2025-1"
        else:
            print(f"✨ '{date_keyword}' 관련 정보로 필터링하여 검색합니다.")

        query = query.strip()+" "+date_keyword+"와 관련된 공지를 찾아줘."

        # ✅ 해당 시기의 문서만 필터링
        all_docs = collection_data[selected_collection]["documents"]
        all_metas = collection_data[selected_collection]["metadatas"]

        return None

    elif selected_collection == "collection_subjectinfo_chunk":
        # 1) 축약어 그룹 치환 적용
        query = preprocess_query(query)

        # 2) 변환된 질의로 학과 키워드 추출
        major_filter_keyword = extract_major_keyword(query, unique_majors,threshold = 70)

        if major_filter_keyword:
            print(f"✨ '{major_filter_keyword}' 관련 정보로 필터링하여 검색합니다.")
        else:
            print("ℹ️ 특정 학과 키워드가 감지되지 않았습니다. 전체 문서에서 검색합니다.")

        # 3) 필터링 키워드를 retriever에 전달
        bm25_rows,dpr_rows = retriever.retrieve(query, top_k = 6, filter_major=major_filter_keyword,alpha=0.5,cat= 2)
        print("ok")
        return bm25_rows,dpr_rows

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
        bm25_rows, dpr_rows, hybrid_rows = retriever.retrieve(query, top_k=4, filter_major=major_filter_keyword,cat=1)

        return bm25_rows, dpr_rows, hybrid_rows

if __name__ == "__main__":
    
    # 엑셀 파일 경로
    file_path = "data/testdata.xlsx"  # 실제 파일 경로로 바꿔주세요

    # 1. 엑셀 파일 읽기
    df = pd.read_excel(file_path)

    all_bm25, all_dpr, all_hybrid = [], [], []

    # 2. query와 cat 컬럼 순회하면서 함수 호출
    for idx, row in df.iterrows():
        query = str(row["query"])
        #cat = str(row["cat"])  # int로 변환해서 전달

        print(f"\n🟢 [{idx+1}] Query: {query}")
        
        try:
            bm25_rows,dpr_rows = overall_logic(query, cat = "2")
            #bm25_rows = overall_logic(query, cat = "2")

            # 각 결과에 원래 query 정보도 함께 저장
            for r in bm25_rows:
                r["query"] = query
                r["method"] = "bm25"
            for r in dpr_rows:
                r["query"] = query
                r["method"] = "DPR"

            all_bm25.extend(bm25_rows)
            all_dpr.extend(dpr_rows)

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            continue

    # 4. DataFrame으로 변환
    df_bm25 = pd.DataFrame(all_bm25)
    df_dpr = pd.DataFrame(all_dpr)

    # 5. Excel로 저장 (시트별 분리)
    with pd.ExcelWriter("retrieval_results13.xlsx", engine="openpyxl") as writer:
        df_bm25.to_excel(writer, sheet_name="hybrid", index=False)
        df_dpr.to_excel(writer, sheet_name="DPR", index=False)
