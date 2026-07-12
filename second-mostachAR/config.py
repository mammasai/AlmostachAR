"""
 تحميل متغيرات البيئة وإعداد الاتصال بـ LLM
"""

import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY غير موجود في ملف .env. "
        "تأكدي من إضافة السطر: GROQ_API_KEY=gsk_..."
    )

def get_llm(model_name: str = "groq/llama-3.3-70b-versatile", temperature: float = 0.3):
    return LLM(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")