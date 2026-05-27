import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent

load_dotenv(PROJECT_DIR / ".env")


def ask_ai(question, context):
    key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not key:
        return "Chưa cấu hình GROQ_API_KEY trong file .env."

    client = Groq(api_key=key)

    prompt = f"""
Bạn là trợ lý AI tư vấn an toàn thực phẩm và giải thích Grad-CAM.
Trả lời tiếng Việt, ngắn gọn, rõ ràng, dễ hiểu.
Không khẳng định thay xét nghiệm phòng lab.
Nếu có nguy cơ thịt hỏng, khuyên người dùng không nên sử dụng.

Kết quả mô hình:
{context}

Câu hỏi:
{question}
"""

    res = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Bạn là trợ lý AI giải thích kết quả phân loại thịt tươi/thịt hỏng và Grad-CAM."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=500
    )

    return res.choices[0].message.content