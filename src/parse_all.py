import os
from pdf_extract import save_pdf_text_to_txt
from gpt_parse import parse_pdf_with_openai

def batch_parse_all_pdfs(pdf_dir="data/pdf/단과대별"):
    # PDF 폴더 내 모든 .pdf 파일 순회
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"📄 처리 중: {pdf_path}")

            # 1. PDF → TXT 저장
            txt_path = save_pdf_text_to_txt(pdf_path, output_dir="data/txt")

            # 2. TXT → GPT 파싱 후 JSON 저장
            parse_pdf_with_openai(txt_path)

if __name__ == "__main__":
    batch_parse_all_pdfs()