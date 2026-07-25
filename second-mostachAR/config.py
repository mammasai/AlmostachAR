# إعدادات المشروع المركزية وتحميل متغيرات البيئة

import os
from dotenv import load_dotenv

import litellm

_original_completion = litellm.completion
_original_acompletion = litellm.acompletion


def _strip_cache_control(messages):
    if not messages:
        return messages
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop("cache_control", None)
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part.pop("cache_control", None)
    return messages


def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_control(kwargs["messages"])
    return _original_completion(*args, **kwargs)


async def _patched_acompletion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_control(kwargs["messages"])
    return await _original_acompletion(*args, **kwargs)


litellm.completion = _patched_completion
litellm.acompletion = _patched_acompletion

from crewai import LLM

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY غير موجود في ملف .env. "
        "تأكدي من إضافة السطر: GROQ_API_KEY=gsk_..."
    )


def get_llm(model_name: str = "groq/llama-3.3-70b-versatile", temperature: float = 0.3):
    """
    يرجع نسخة مهيأة من crewai.LLM جاهزة للاستخدام في أي وكيل.
    """
    return LLM(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")