import openai
import json
import os
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
model = "gpt-4-1106-preview"
flag = 1 #flag가 1이면 parse_all로 실행, flag가 0이면 개별로 실행

def parse_pdf_with_openai(txt_path: str, output_path: str = None):
    # PDF 텍스트 파일 load
    with open(txt_path, "r", encoding="utf-8") as f:
        pdf_text = f.read()

    # GPT에 요청할 텍스트 생성 (요구 구조에 맞게)
    prompt = (
    "당신은 문서를 구조화된 JSON으로 정리하는 데이터 분석 도우미입니다.\n\n"
    "다음은 PDF 문서에서 추출된 전체 텍스트입니다. 이 내용을 빠짐없이 분석해서 JSON 구조로 요약해줘:\n\n"
    "- 가능한 모든 내용을 포함해줘. 표와 설명 모두 빠짐없이 요약해야 해.\n"
    "- 과목의 내용을 적을 때에는 요약하지 말고 전체 내용을 모두 적어줘.\n"
    "- 항목 내부에 \\\"비고\\\"라는 필드는 해당 그룹 전체에 공통되는 주석일 수 있으므로 필드가 항목 안에 존재하는 경우, 그 내용을 항목에서 제거하고, 그룹의 마지막에 \\\"비고\\\" 필드로 한 번만 나타내줘..\n"
    "- 나머지 구조는 바꾸지 말고, 항목 순서와 키 이름은 그대로 유지해줘.\n"
    "- 과목 코드와 교과목명은 반드시 별도의 key로 저장해주세요\n"
    "- 출력은 반드시 JSON 형식만 포함해야 하며, 마크다운 코드블록(```json 등)은 절대 쓰지 마.\n"
    "- 모든 문자열과 key값은 반드시 큰따옴표(\\\")로 감싸줘.\n\n"
    "아래는 PDF에서 추출한 텍스트입니다:\n\n"
    f"{pdf_text}"
    )

    # OpenAI GPT 요청
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4096, # context 제한 고려
    )

    json_text = response["choices"][0]["message"]["content"].strip()

    # 코드 블럭 제거
    if json_text.startswith("```json"):
        json_text = json_text[7:-3].strip()
    elif json_text.startswith("```"):
        json_text = json_text[3:-3].strip()

    # JSON 파싱
    try:
        parsed = json.loads(json_text)
    except Exception as e:
        print(f"❌ JSON 파싱 실패:", e)
        return

    # 출력 경로 지정
    if not output_path:
        if flag == 1:
            os.makedirs("data/json", exist_ok=True)
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            output_path = f"data/json/{base_name}.json"
        else:
            os.makedirs("data/json/ind", exist_ok=True)
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            output_path = f"data/json/ind/{base_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"✅ GPT 기반 PDF 파싱 완료 → {output_path}")


# ✅ 메인 실행
if __name__ == "__main__":
    flag = 0
    parse_pdf_with_openai("data/txt/국문1.txt")