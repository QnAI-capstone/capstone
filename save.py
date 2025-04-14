from src.ingest import ingest_json_to_chroma

if __name__ == "__main__":
    ingest_json_to_chroma("data/json/syllabus-ex.json", "syllabus-ex.pdf")
