"""
 تنظيف ومعالجة النصوص العربية باستخدام CAMeL Tools
"""

import os
import sys
import re
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm

from crewai import Agent, Task, Crew
from crewai.tools import tool

from camel_tools.utils.normalize import (
    normalize_unicode,
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)
from camel_tools.utils.dediac import dediac_ar


def _read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        return pd.DataFrame({"text": lines})
    else:
        raise ValueError("صيغة الملف غير مدعومة. استخدمي CSV, Excel, أو TXT.")


def _write_file(df: pd.DataFrame, output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif output_path.endswith((".xlsx", ".xls")):
        df.to_excel(output_path, index=False)
    elif output_path.endswith(".txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(df.iloc[:, 0].astype(str).tolist()))
    else:
        raise ValueError("صيغة الملف غير مدعومة للحفظ.")


def _detect_text_column(df: pd.DataFrame, text_column: str = None) -> str:
    if text_column and text_column in df.columns:
        return text_column
    text_columns = df.select_dtypes(include="object").columns
    if len(text_columns) == 0:
        raise ValueError("لم يتم العثور على عمود نصي في الملف.")
    return text_columns[0]


@tool("Repeated Characters Remover")
def remove_repeated_chars(file_path: str, text_column: str = "", output_path: str = "") -> str:
    try:
        df = _read_file(file_path)
        col = _detect_text_column(df, text_column)

        def clean_repeats(text):
            if not isinstance(text, str):
                return text
            return re.sub(r"(.)\1{2,}", r"\1", text)

        df[col] = df[col].apply(clean_repeats)

        if not output_path:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_cleaned{ext}"

        _write_file(df, output_path)
        return (
            f"تم إزالة تكرار الحروف بنجاح. عدد الصفوف المعالجة: {len(df)}. "
            f"الملف المُعالج محفوظ في: {output_path}"
        )

    except Exception as e:
        return f"خطأ أثناء إزالة تكرار الحروف: {str(e)}"


@tool("Arabic Text Normalizer")
def normalize_arabic_text(file_path: str, text_column: str = "", output_path: str = "") -> str:
    try:
        df = _read_file(file_path)
        col = _detect_text_column(df, text_column)

        def normalize_text(text):
            if not isinstance(text, str):
                return text
            text = normalize_unicode(text)
            text = normalize_alef_ar(text)
            text = normalize_alef_maksura_ar(text)
            text = normalize_teh_marbuta_ar(text)
            return text

        df[col] = df[col].apply(normalize_text)

        if not output_path:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_normalized{ext}"

        _write_file(df, output_path)
        return (
            f"تم تطبيع النصوص بنجاح. عدد الصفوف المعالجة: {len(df)}. "
            f"الملف المُعالج محفوظ في: {output_path}"
        )

    except Exception as e:
        return f"خطأ أثناء التطبيع: {str(e)}"


@tool("Arabic Diacritics Remover")
def remove_diacritics(file_path: str, text_column: str = "", output_path: str = "") -> str:
    try:
        df = _read_file(file_path)
        col = _detect_text_column(df, text_column)

        df[col] = df[col].apply(lambda t: dediac_ar(t) if isinstance(t, str) else t)

        if not output_path:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_dediac{ext}"

        _write_file(df, output_path)
        return (
            f"تم إزالة التشكيل بنجاح. عدد الصفوف المعالجة: {len(df)}. "
            f"الملف المُعالج محفوظ في: {output_path}"
        )

    except Exception as e:
        return f"خطأ أثناء إزالة التشكيل: {str(e)}"


@tool("Full Arabic Text Cleaner")
def full_clean_pipeline(file_path: str, text_column: str = "", keep_diacritics: bool = False, output_path: str = "") -> str:
    try:
        df = _read_file(file_path)
        col = _detect_text_column(df, text_column)

        def clean_pipeline(text):
            if not isinstance(text, str):
                return text
            text = re.sub(r"(.)\1{2,}", r"\1", text)
            text = normalize_unicode(text)
            text = normalize_alef_ar(text)
            text = normalize_alef_maksura_ar(text)
            text = normalize_teh_marbuta_ar(text)
            if not keep_diacritics:
                text = dediac_ar(text)
            return text.strip()

        df[col] = df[col].apply(clean_pipeline)

        if not output_path:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_processed{ext}"

        _write_file(df, output_path)

        diacritics_note = "تم الإبقاء على التشكيل" if keep_diacritics else "تم إزالة التشكيل"
        return (
            f"تم تنظيف الملف بنجاح عبر خط الأنابيب الكامل "
            f"(إزالة تكرار الحروف + التطبيع + {diacritics_note}).\n"
            f"عدد الصفوف المعالجة: {len(df)}\n"
            f"الملف النهائي محفوظ في: {output_path}"
        )

    except Exception as e:
        return f"خطأ أثناء التنظيف الشامل: {str(e)}"


def create_preprocessor_agent():
    llm = get_llm(temperature=0.2)

    agent = Agent(
        role="أخصائية معالجة النصوص العربية",
        goal=(
            "تنظيف وتجهيز النصوص العربية الخام لتصبح جاهزة للاستخدام في نماذج "
            "معالجة اللغة الطبيعية، عبر التطبيع، إزالة التكرار غير الطبيعي، "
            "والتعامل الذكي مع التشكيل حسب طبيعة المهمة."
        ),
        backstory=(
            "أنتِ متخصصة في معالجة اللغة العربية الطبيعية (Arabic NLP)، خبيرة "
            "في استخدام مكتبة CAMeL Tools. تفهمين الفرق بين المهام التي تحتاج "
            "تشكيلاً (مثل تحويل النص إلى كلام) والمهام التي لا تحتاجه (مثل التصنيف). "
            "تعملين بدقة ولا تتلفين البيانات الأصلية أبداً، بل تنشئين نسخاً معالجة جديدة. "
            "دائماً تستشهدين بالأرقام الفعلية اللي رجعتها الأدوات، ولا تخترعين أرقاماً "
            "غير موجودة في نتائج الأدوات. قاعدة صارمة: إذا رجعت أي أداة نتيجة تبدأ برمز الخطأ، "
            "يُمنع منعاً باتاً أن تكتبي في إجابتك النهائية أن العملية نجحت "
            "أو تخترعي مساراً لملف غير موجود فعلياً. في هذه الحالة، اذكري الخطأ بصراحة "
            "واقترحي حلاً (مثل استخدام مسار افتراضي بدل تحديد مسار مخصص)."
        ),
        tools=[
            remove_repeated_chars,
            normalize_arabic_text,
            remove_diacritics,
            full_clean_pipeline,
        ],
        llm=llm,
        verbose=True,
    )
    return agent


def create_preprocessing_task(agent, file_path: str, task_type: str = "classification"):
    keep_diacritics = task_type.lower() in ["tts", "text_to_speech"]

    task = Task(
        description=(
            f"نظّفي الملف الموجود في المسار التالي: {file_path}\n"
            f"نوع المهمة المستهدفة: {task_type}\n"
            "استخدمي أداة 'Full Arabic Text Cleaner' لتطبيق خط الأنابيب الكامل. "
            "لا تحددي output_path بنفسك — مرّريها كنص فارغ '' (وليس null) ليتم استخدام المسار "
            "الافتراضي الآمن تلقائياً بجانب الملف الأصلي.\n"
            f"(keep_diacritics={keep_diacritics} بناءً على نوع المهمة).\n"
            "بعد التنظيف، اكتبي ملخصاً قصيراً بالعربية يوضح: عدد الصفوف المعالجة "
            "(استخدمي الرقم الفعلي الراجع من الأداة فقط، لا تخترعي رقماً)، "
            "أهم التغييرات اللي طبّقتيها، ومسار الملف النهائي الفعلي كما رجعته الأداة. "
            "إذا رجعت الأداة خطأ، اذكري ذلك بصراحة في الملخص ولا تدّعي النجاح."
        ),
        expected_output=(
            "ملخص بالعربية يوضح نجاح عملية التنظيف، عدد الصفوف الفعلي المعالج، "
            "الخطوات المطبقة، ومسار الملف المُعالج النهائي."
        ),
        agent=agent,
    )
    return task


if __name__ == "__main__":
    test_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "sample.csv"
    )

    if not os.path.exists(test_file_path):
        print(f"الملف التجريبي غير موجود في: {test_file_path}")
    else:
        preprocessor = create_preprocessor_agent()
        task = create_preprocessing_task(preprocessor, test_file_path, task_type="classification")

        crew = Crew(
            agents=[preprocessor],
            tasks=[task],
            verbose=True,
        )

        result = crew.kickoff()
        print("\n" + "=" * 50)
        print("النتيجة النهائية:")
        print("=" * 50)
        print(result)