"""
 فحص وتشخيص جودة الداتاسيت العربي قبل أي معالجة
"""

import os
import sys
import chardet
import pandas as pd
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm

from crewai import Agent, Task, Crew
from crewai.tools import tool


@tool("Encoding Checker")
def check_encoding(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(100000)
            result = chardet.detect(raw_data)

        encoding = result["encoding"]
        confidence = result["confidence"]

        if encoding and encoding.lower().replace("-", "") == "utf8":
            return f"الترميز سليم: UTF-8 (نسبة الثقة: {confidence:.2%})"
        else:
            return (
                f"تحذير: الترميز المكتشف هو {encoding} "
                f"(نسبة الثقة: {confidence:.2%})، وليس UTF-8. "
                f"يُنصح بإعادة حفظ الملف بترميز UTF-8 قبل المتابعة."
            )
    except Exception as e:
        return f"خطأ أثناء فحص الترميز: {str(e)}"


@tool("Arabic Word Distribution Analyzer")
def analyze_word_distribution(file_path: str, text_column: str = None) -> str:
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_data = f.read()
            df = pd.DataFrame({"text": text_data.splitlines()})
        else:
            return "صيغة الملف غير مدعومة. استخدمي CSV, Excel, أو TXT."

        if text_column is None:
            text_columns = df.select_dtypes(include="object").columns
            if len(text_columns) == 0:
                return "لم يتم العثور على عمود نصي في الملف."
            text_column = text_columns[0]

        if text_column not in df.columns:
            return f"العمود '{text_column}' غير موجود. الأعمدة المتاحة: {list(df.columns)}"

        all_text = " ".join(df[text_column].astype(str).tolist())
        words = all_text.split()

        total_words = len(words)
        arabic_words = [w for w in words if any("\u0600" <= c <= "\u06FF" for c in w)]
        arabic_ratio = len(arabic_words) / total_words if total_words > 0 else 0

        word_counts = Counter(arabic_words)
        top_10 = word_counts.most_common(10)

        report = f"""
تقرير توزيع الكلمات:
- إجمالي عدد الصفوف: {len(df)}
- إجمالي عدد الكلمات: {total_words}
- عدد الكلمات العربية: {len(arabic_words)}
- نسبة الكلمات العربية: {arabic_ratio:.2%}

أكثر 10 كلمات تكراراً:
"""
        for word, count in top_10:
            report += f"  - {word}: {count} مرة\n"

        return report

    except Exception as e:
        return f"خطأ أثناء تحليل توزيع الكلمات: {str(e)}"


@tool("Vocabulary Density Calculator")
def calculate_vocabulary_density(file_path: str, text_column: str = None) -> str:
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_data = f.read()
            df = pd.DataFrame({"text": text_data.splitlines()})
        else:
            return "صيغة الملف غير مدعومة."

        if text_column is None:
            text_columns = df.select_dtypes(include="object").columns
            if len(text_columns) == 0:
                return "لم يتم العثور على عمود نصي."
            text_column = text_columns[0]

        all_text = " ".join(df[text_column].astype(str).tolist())
        words = all_text.split()

        total_words = len(words)
        unique_words = len(set(words))
        density = unique_words / total_words if total_words > 0 else 0

        if density > 0.5:
            interpretation = "تنوع لغوي مرتفع جداً — قد يشير لبيانات غير متجانسة أو ضوضاء."
        elif density > 0.3:
            interpretation = "تنوع لغوي جيد ومتوازن."
        else:
            interpretation = "تنوع لغوي منخفض — البيانات متكررة، قد تحتاجي مصادر إضافية."

        return f"""
كثافة المفردات (Vocabulary Density):
- إجمالي الكلمات: {total_words}
- الكلمات الفريدة: {unique_words}
- نسبة الكثافة: {density:.2%}
- التفسير: {interpretation}
"""
    except Exception as e:
        return f"خطأ أثناء حساب كثافة المفردات: {str(e)}"


def create_data_inspector_agent():
    llm = get_llm(temperature=0.2)

    agent = Agent(
        role="محلل جودة البيانات العربية",
        goal=(
            "فحص أي داتاسيت عربي يُرفع من المستخدم بدقة، والتحقق من الترميز، "
            "وتحليل توزيع الكلمات، وحساب كثافة المفردات، ثم تقديم تقرير "
            "تشخيصي واضح بالعربية يوجّه المستخدم للخطوة التالية."
        ),
        backstory=(
            "أنتِ خبير في معالجة اللغة العربية الطبيعية (Arabic NLP)، "
            "عملت لسنوات في تجهيز داتاسيتات عربية لمشاريع تعلم آلي. "
            "تعرف تماماً المشاكل الشائعة في البيانات العربية: مشاكل الترميز، "
            "اختلاط اللهجات مع الفصحى، والتكرار المفرط. أسلوبك مباشر وعملي."
        ),
        tools=[check_encoding, analyze_word_distribution, calculate_vocabulary_density],
        llm=llm,
        verbose=True,
    )
    return agent


def create_inspection_task(agent, file_path: str):
    task = Task(
        description=(
            f"افحص الملف الموجود في المسار التالي: {file_path}\n"
            "1. تحقق من ترميز الملف (يجب أن يكون UTF-8).\n"
            "2. حلل توزيع الكلمات العربية ونسبتها.\n"
            "3. احسب كثافة المفردات وفسريها.\n"
            "4. اكتب تقريراً استشارياً نهائياً بالعربية يلخص كل النتائج "
            "ويقترح خطوات عملية للمستخدم (مثلاً: هل البيانات جاهزة للمعالجة، "
            "أم تحتاج تنظيف إضافي)."
        ),
        expected_output=(
            "تقرير منظم بالعربية يحتوي على: حالة الترميز، إحصائيات الكلمات، "
            "كثافة المفردات، وتوصيات عملية واضحة للخطوة التالية."
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
        print("ضع ملف CSV أو TXT في مجلد data/raw/ جرب مرة أخرى.")
    else:
        inspector = create_data_inspector_agent()
        task = create_inspection_task(inspector, test_file_path)

        crew = Crew(
            agents=[inspector],
            tasks=[task],
            verbose=True,
        )

        result = crew.kickoff()
        print("\n" + "=" * 50)
        print("النتيجة النهائية:")
        print("=" * 50)
        print(result)