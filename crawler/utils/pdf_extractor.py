import requests
from io import BytesIO
import PyPDF2

def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url, verify=False)
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"⚠️ PDF 텍스트 추출 실패: {e}")
        return ""
