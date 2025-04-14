import openai
from src.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_embedding(text: str) -> list:
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",  # 또는 "text-embedding-ada-002"
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print("❌ OpenAI 임베딩 실패:", e)
        return None
