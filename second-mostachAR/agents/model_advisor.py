"""
 اقتراح أفضل نموذج لغوي عربي حسب طبيعة المهمة وخصائص البيانات
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm

from crewai import Agent, Task, Crew
from crewai.tools import tool


MODELS_KNOWLEDGE_BASE = {
    "arabert": {
        "full_name": "AraBERT",
        "hf_path": "aubmindlab/bert-base-arabertv2",
        "best_for": ["تصنيف نصوص عامة", "استخراج الكيانات (NER)", "الإجابة على الأسئلة", "الفصحى الحديثة (MSA)"],
        "notes": "الأنسب للنصوص الرسمية والفصحى الحديثة. أداء ممتاز في المهام العامة.",
    },
    "camelbert_msa": {
        "full_name": "CAMeLBERT-MSA",
        "hf_path": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
        "best_for": ["الفصحى الحديثة (MSA)", "المهام الصرفية والنحوية", "تصنيف نصوص رسمية"],
        "notes": "مدرّب خصيصاً على نصوص فصحى، دقيق جداً في المهام الصرفية.",
    },
    "camelbert_dialect": {
        "full_name": "CAMeLBERT-DA (Dialectal Arabic)",
        "hf_path": "CAMeL-Lab/bert-base-arabic-camelbert-da",
        "best_for": ["اللهجات العربية", "تحليل نصوص عامية", "المهام اللي فيها خليط لهجات"],
        "notes": "الأنسب إذا كانت البيانات فيها لهجات مصرية، خليجية، شامية، مغربية.",
    },
    "camelbert_mix": {
        "full_name": "CAMeLBERT-Mix (MSA + Dialect)",
        "hf_path": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        "best_for": ["بيانات مختلطة (فصحى + لهجات)", "مهام عامة بدون معرفة مسبقة بنوع النص"],
        "notes": "خيار آمن ومتوازن إذا كانت البيانات غير متجانسة أو غير معروفة الطبيعة.",
    },
    "marbert": {
        "full_name": "MarBERT",
        "hf_path": "UBC-NLP/MARBERT",
        "best_for": ["نصوص تويتر/X", "تحليل المشاعر", "اللهجات العامية على وسائل التواصل"],
        "notes": "مدرّب على تغريدات عربية، الأفضل لتحليل المشاعر والنصوص القصيرة العامية من السوشيال ميديا.",
    },
}


DIALECT_MARKERS = {
    "EGY": ["مش", "ازيك", "عايز", "عاوز", "دلوقتي", "كده", "ايه", "علشان", "خالص", "بتاع"],
    "GLF": ["شلون", "وايد", "شنو", "يبا", "زين", "ابغى", "احين", "مب", "ماكو"],
    "LEV": ["شو", "هيك", "منيح", "كتير", "هلق", "ليش", "بدي", "تبعي"],
    "alg": ["واش", "بزاف", "زعما", "دابا", "غادي", "نتا", "شحال", "بصح"],
}


def _read_any_file(file_path):
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


@tool("Dialect Identifier")
def identify_dialect(file_path: str, text_column: str = None, sample_size: int = 50) -> str:
    try:
        df = _read_any_file(file_path)

        if text_column is None or text_column not in df.columns:
            text_columns = df.select_dtypes(include="object").columns
            if len(text_columns) == 0:
                return "لم يتم العثور على عمود نصي."
            text_column = text_columns[0]

        sample_texts = df[text_column].dropna().astype(str).head(sample_size).tolist()
        if not sample_texts:
            return "لا توجد نصوص كافية للتحليل."

        dialect_hits = {k: 0 for k in DIALECT_MARKERS}
        msa_count = 0

        for text in sample_texts:
            found_dialect = False
            for dialect, markers in DIALECT_MARKERS.items():
                if any(marker in text for marker in markers):
                    dialect_hits[dialect] += 1
                    found_dialect = True
            if not found_dialect:
                msa_count += 1

        total = len(sample_texts)
        report = "تقرير تقدير اللهجة (بمنهج المؤشرات اللغوية):\n"
        report += f"- حجم العينة المفحوصة: {total} نص\n\n"
        report += "توزيع المؤشرات المكتشفة:\n"
        report += f"  - فصحى/غير محدد (MSA): {msa_count} ({msa_count/total*100:.1f}%)\n"
        for dialect, count in sorted(dialect_hits.items(), key=lambda x: x[1], reverse=True):
            report += f"  - {dialect}: {count} ({count/total*100:.1f}%)\n"

        all_counts = {"MSA": msa_count, **dialect_hits}
        dominant = max(all_counts, key=all_counts.get)
        report += f"\nالفئة المهيمنة (تقديرية): {dominant}"
        report += "\nملاحظة: هذا تقدير مبني على مؤشرات لغوية شائعة، وليس نموذج DID متقدم (غير متاح على Windows)."

        return report

    except Exception as e:
        return f"خطأ أثناء تقدير اللهجة: {str(e)}"


@tool("Arabic Models Knowledge Base")
def query_models_knowledge_base(query_type: str = "all") -> str:
    if query_type.lower() == "all":
        report = "قاعدة معرفية بالنماذج العربية المتاحة:\n\n"
        for key, info in MODELS_KNOWLEDGE_BASE.items():
            report += f"{info['full_name']} ({info['hf_path']})\n"
            report += f"   الأنسب لـ: {', '.join(info['best_for'])}\n"
            report += f"   ملاحظة: {info['notes']}\n\n"
        return report

    key = query_type.lower().replace("-", "_").replace(" ", "_")
    if key in MODELS_KNOWLEDGE_BASE:
        info = MODELS_KNOWLEDGE_BASE[key]
        return (
            f"{info['full_name']} ({info['hf_path']})\n"
            f"الأنسب لـ: {', '.join(info['best_for'])}\n"
            f"ملاحظة: {info['notes']}"
        )
    else:
        available = ", ".join(MODELS_KNOWLEDGE_BASE.keys())
        return f"النموذج '{query_type}' غير موجود في القاعدة. النماذج المتاحة: {available}"


@tool("Model Recommendation Engine")
def recommend_model(task_type: str, dominant_dialect: str = "MSA") -> str:
    task_type = task_type.lower()
    is_msa = dominant_dialect.upper() == "MSA"

    if task_type in ["sentiment", "social_media", "twitter"]:
        recommendation = "marbert"
        reason = "المهمة متعلقة بتحليل مشاعر أو محتوى سوشيال ميديا، وMarBERT مدرّب خصيصاً على هالنوع من النصوص."
    elif task_type in ["ner", "morphology", "syntax"] and is_msa:
        recommendation = "camelbert_msa"
        reason = "المهمة صرفية/نحوية والبيانات فصحى، فـ CAMeLBERT-MSA الأدق في هالحالة."
    elif not is_msa:
        recommendation = "camelbert_dialect"
        reason = f"البيانات فيها لهجة مهيمنة ({dominant_dialect})، وCAMeLBERT-DA مصمم خصيصاً للهجات."
    elif task_type in ["classification", "qa", "general"] and is_msa:
        recommendation = "arabert"
        reason = "المهمة تصنيف عام أو أسئلة/أجوبة، والبيانات فصحى — AraBERT خيار قوي وموثوق في هالحالات."
    else:
        recommendation = "camelbert_mix"
        reason = "الحالة غير محددة بوضوح، فـ CAMeLBERT-Mix خيار آمن ومتوازن."

    info = MODELS_KNOWLEDGE_BASE[recommendation]
    return (
        f"التوصية النهائية: {info['full_name']}\n"
        f"المسار على Hugging Face: {info['hf_path']}\n"
        f"سبب الاختيار: {reason}\n"
        f"استخدامات أخرى للنموذج: {', '.join(info['best_for'])}"
    )


def create_model_advisor_agent():
    llm = get_llm(temperature=0.3)

    agent = Agent(
        role="مستشارة اختيار النماذج اللغوية العربية",
        goal=(
            "تحليل خصائص البيانات (اللهجة، نوع المهمة) واقتراح أنسب نموذج لغوي "
            "عربي مُدرّب مسبقاً (AraBERT, CAMeLBERT, MarBERT) مع تبرير واضح للاختيار."
        ),
        backstory=(
            "أنتِ خبيرة في نماذج المحولات (Transformers) العربية المتاحة على Hugging Face. "
            "تعرفين نقاط القوة والضعف لكل نموذج، وتفهمين إن اختيار النموذج الخاطئ "
            "يضيع وقت وموارد التدريب."
        ),
        tools=[identify_dialect, query_models_knowledge_base, recommend_model],
        llm=llm,
        verbose=True,
    )
    return agent


def create_advisory_task(agent, file_path, task_type="classification"):
    task = Task(
        description=(
            f"حلّلي الملف الموجود في المسار التالي: {file_path}\n"
            f"نوع المهمة المستهدفة من المستخدم: {task_type}\n\n"
            "الخطوات المطلوبة:\n"
            "1. استخدمي أداة 'Dialect Identifier' لتقدير اللهجة أو الفصحى المهيمنة في البيانات.\n"
            "2. استخدمي أداة 'Model Recommendation Engine' بناءً على نوع المهمة واللهجة المكتشفة.\n"
            "3. اكتبي تقريراً استشارياً نهائياً بالعربية."
        ),
        expected_output=(
            "تقرير استشاري بالعربية يوضح: نتيجة تقدير اللهجة، النموذج الموصى به، "
            "ومسار Hugging Face الخاص فيه، مع تبرير منطقي للاختيار."
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
        advisor = create_model_advisor_agent()
        task = create_advisory_task(advisor, test_file_path, task_type="classification")

        crew = Crew(
            agents=[advisor],
            tasks=[task],
            verbose=True,
        )

        result = crew.kickoff()
        print("\n" + "=" * 50)
        print("النتيجة النهائية:")
        print("=" * 50)
        print(result)