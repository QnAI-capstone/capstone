import google.generativeai as genai
import json
import os
from src.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

def parse_pdf_with_gemini(pdf_path: str, output_path: str = None):
    # PDF 파일을 바이너리 형식으로 읽기
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Gemini에 요청할 텍스트 생성 (주석에서 요구한 구조에 맞게)
    prompt = (
        "이 PDF를 표와 텍스트를 구조화된 JSON 형식으로 요약해줘. "
        "모든 출력은 JSON이어야 하고, 각 문자열은 반드시 따옴표로 감싸줘. "
        "``` 같은 마크다운 코드 블록도 사용하지 마. 오직 JSON만."
        "PDF의 모든 내용을 빠짐없이 요약해줘."
    )
    
    # Gemini에 요청
    response = model.generate_content([
        {"mime_type": "application/pdf", "data": pdf_bytes},
        prompt
    ], stream=False)
    
    print("📥 Gemini 응답 원문:")
    print(response.text)  # 응답 원문 출력

    json_text = response.text.strip()
    
    # 코드 블록 제거
    if json_text.startswith("```json"):
        json_text = json_text.strip()[7:-3].strip()  # ```json\n ... \n``` 제거
    elif json_text.startswith("```"):
        json_text = json_text.strip()[3:-3].strip()  # ``` ... ``` 제거

    # JSON 파싱 시도
    try:
        parsed = json.loads(json_text)
        print(f"✅ Gemini 기반 PDF 파싱 완료 → {output_path}")
    except Exception as e:
        print(f"❌ JSON 파싱 실패:", e)
        return

    # JSON 파일로 저장
    if not output_path:
        os.makedirs("data/json", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"data/json/{base_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"✅ Gemini 기반 PDF 파싱 완료 → {output_path}")


# ✅ 메인 실행 부분 추가
if __name__ == "__main__":
    # 테스트용 기본 파일 경로
    parse_pdf_with_gemini("data/2022-kll.pdf")
