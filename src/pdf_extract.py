import fitz
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()

def save_pdf_text_to_txt(pdf_path: str, output_dir="data/txt") -> str:
    text = extract_text_from_pdf(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ 텍스트 저장 완료 → {output_path}")
    return output_path  # 저장된 파일 경로 반환

if __name__ == "__main__":
    pdf_path = "data/pdf/글경_과목이수.pdf"
    txt_path = save_pdf_text_to_txt(pdf_path)