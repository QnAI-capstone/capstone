from src.pdf_parser import parse_pdf

def test_parse_pdf():
    file_path = "data/syllabus-ex.pdf"
    parsed = parse_pdf(file_path)

    assert isinstance(parsed, list)
    assert len(parsed) > 0
    assert "page" in parsed[0]
    assert "content" in parsed[0]
    print(f"✅ 총 {len(parsed)}페이지 파싱 완료")
