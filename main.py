from src.rag_chain import answer_query

if __name__ == "__main__":
    print("📚 PDF 기반 챗봇입니다. 질문을 입력하세요. 종료하려면 'q' 입력.")
    while True:
        q = input("\n❓ 질문: ")
        if q.lower() == "q":
            break
        print("\n📌 답변:")
        print(answer_query(q))
        print("\n" + "-"*60)
