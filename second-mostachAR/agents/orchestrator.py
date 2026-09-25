# وكيل المدير الذي يوجّه المحادثة للوكلاء المتخصصين

import os
import sys
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm

from crewai import Agent, Task, Crew
from crewai.tools import tool

from agents.data_inspector import create_data_inspector_agent, create_inspection_task
from agents.preprocessor import create_preprocessor_agent, create_preprocessing_task
from agents.model_advisor import create_model_advisor_agent, create_advisory_task
from agents.trainer import create_trainer_agent, create_training_prep_task
from agents.evaluator import create_evaluator_agent, create_evaluation_task
from agents.dialect_agent import run_dialect_turn, AVAILABLE_DIALECTS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# نموذج Groq اللي بيدعم الرؤية البصرية (Vision) بالإضافة لاستخدام الأدوات.
# ملاحظة: llama-4-scout-17b-16e-instruct تم إيقافه (deprecated) من Groq بتاريخ
# 17 يونيو 2026 كمان. البديل الرسمي الموصى به لمهام الرؤية هو qwen/qwen3.6-27b.
VISION_MODEL_NAME = "groq/qwen/qwen3.6-27b"


@tool("Delegate to Data Inspector")
def delegate_to_inspector(file_path: str) -> str:
    """
    تستدعي وكيلة فحص جودة البيانات المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم
    فحص ملف بيانات، التحقق من الترميز، أو تحليل جودة نص عربي خام.
    المدخل: file_path (مسار الملف الكامل).
    """
    try:
        inspector = create_data_inspector_agent()
        task = create_inspection_task(inspector, file_path)
        crew = Crew(agents=[inspector], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f" خطأ أثناء تفويض وكيل الفحص: {str(e)}"


@tool("Delegate to Preprocessor")
def delegate_to_preprocessor(file_path: str, task_type: str) -> str:
    """
    تستدعي وكيلة تنظيف النصوص العربية المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم
    تنظيف بيانات، إزالة تشكيل، تطبيع نصوص، أو تجهيز ملف لمهمة تدريب.
    المدخلات (الاثنين إلزاميين، لازم تمرّريهم دايماً):
    - file_path: مسار الملف.
    - task_type: نوع المهمة، مثل 'classification' أو 'tts'. لو المستخدم ما
      حدد نوع المهمة صراحة، مرّري 'classification' كقيمة افتراضية معقولة.
    """
    try:
        preprocessor = create_preprocessor_agent()
        task = create_preprocessing_task(preprocessor, file_path, task_type=task_type)
        crew = Crew(agents=[preprocessor], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f" خطأ أثناء تفويض وكيل التنظيف: {str(e)}"


@tool("Delegate to Model Advisor")
def delegate_to_advisor(file_path: str, task_type: str) -> str:
    """
    تستدعي المستشارة المتخصصة في اختيار النماذج اللغوية العربية. استخدمي هذه الأداة
    لما يسأل المستخدم عن أنسب نموذج (AraBERT, CAMeLBERT, MarBERT) لمهمته.
    المدخلات (الاثنين إلزاميين، لازم تمرّريهم دايماً):
    - file_path: مسار الملف.
    - task_type: نوع المهمة. لو ما حدد المستخدم، مرّري 'classification'.
    """
    try:
        advisor = create_model_advisor_agent()
        task = create_advisory_task(advisor, file_path, task_type=task_type)
        crew = Crew(agents=[advisor], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f" خطأ أثناء تفويض وكيل الاستشارة: {str(e)}"


@tool("Delegate to Trainer")
def delegate_to_trainer(file_path: str, model_hf_path: str, text_column: str,
                         label_column: str, num_labels: int) -> str:
    """
    تستدعي مهندسة التدريب المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم تحضير
    بيانات للتدريب، تقسيم داتاسيت، أو توليد سكريبت تدريب (Fine-tuning) كامل.
    المدخلات (كلها إلزامية، لازم تمرّريها دايماً، حتى لو بقيم افتراضية معقولة):
    - file_path: مسار الملف.
    - model_hf_path: مسار النموذج على Hugging Face (اسألي المستخدم لو ما حدده).
    - text_column: اسم عمود النص، افتراضياً 'text' لو ما حدد المستخدم.
    - label_column: اسم عمود التصنيف، افتراضياً 'label' لو ما حدد المستخدم.
    - num_labels: عدد الفئات، افتراضياً 2 لو ما حدد المستخدم.
    """
    try:
        trainer = create_trainer_agent()
        task = create_training_prep_task(
            trainer, file_path, model_hf_path=model_hf_path,
            text_column=text_column, label_column=label_column, num_labels=num_labels,
        )
        crew = Crew(agents=[trainer], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f" خطأ أثناء تفويض وكيل التدريب: {str(e)}"


@tool("Delegate to Evaluator")
def delegate_to_evaluator(predictions_file: str, text_column: str,
                           true_label_column: str,
                           predicted_label_column: str) -> str:
    """
    تستدعي محللة الأداء المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم تقييم أداء
    نموذج مدرَّب، حساب مقاييس (Accuracy, F1)، أو تحليل أخطاء التصنيف.
    المدخلات (كلها إلزامية، لازم تمرّريها دايماً، حتى لو بقيم افتراضية معقولة):
    - predictions_file: ملف فيه القيم الحقيقية والمتوقعة.
    - text_column: افتراضياً 'text' لو ما حدد المستخدم.
    - true_label_column: افتراضياً 'true_label' لو ما حدد المستخدم.
    - predicted_label_column: افتراضياً 'predicted_label' لو ما حدد المستخدم.
    """
    try:
        evaluator = create_evaluator_agent()
        task = create_evaluation_task(
            evaluator, predictions_file, text_column=text_column,
            true_label_column=true_label_column, predicted_label_column=predicted_label_column,
        )
        crew = Crew(agents=[evaluator], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f" خطأ أثناء تفويض وكيل التقييم: {str(e)}"


@tool("Delegate to Dialect Agent")
def delegate_to_dialect_agent(user_message: str, dialect: str) -> str:
    """
    تستدعي خبيرة اللهجات العربية المحكية عشان تردي على المستخدم بنفس لهجته.
    استخدميها لو المستخدم كتب بلهجة عامية واضحة أو طلب الرد بلهجة معينة.
    أمثلة كلمات تدل على اللهجة: جزائرية (واش، كيفاش، بزاف)، تونسية (شنية،
    عسلامة، برشا)، مغربية (واخا، دابا، ديال)، مصرية (إزيك، إيه، خالص)،
    خليجية (شلونك، وايد، الحين).
    المدخلات: user_message (رسالة المستخدم كما هي)، dialect (وحدة من:
    'جزائرية'، 'تونسية'، 'مغربية'، 'مصرية'، 'خليجية' — بالضبط بهذا الشكل).
    """
    try:
        if dialect not in AVAILABLE_DIALECTS:
            available = "، ".join(AVAILABLE_DIALECTS)
            return f"لهجة غير مدعومة. اللهجات المتاحة حالياً: {available}"
        return run_dialect_turn(user_message=user_message, dialect=dialect)
    except Exception as e:
        return f" خطأ أثناء تفويض وكيلة اللهجات: {str(e)}"


@tool("Analyze Image")
def analyze_image(image_path: str, question: str = "شو اللي تشوفينه في هذي الصورة؟ جاوبي بأسلوب محادثة طبيعي ومباشر، فقرة أو فقرتين قصار، من غير عناوين أو نقاط أو ترقيم.") -> str:
    """
    تحلّل صورة مرفوعة من المستخدم بصرياً وتصف محتواها أو تجاوب عن سؤال محدد بخصوصها،
    باستخدام نموذج رؤية حاسوبية (Vision). استخدمي هذه الأداة أي وقت المستخدم يرفع
    صورة (png, jpg, jpeg, webp) ويسأل عنها أو يطلب وصفها/تحليلها.
    المدخلات: image_path (مسار الصورة الكامل الموجود على القرص)،
    question (سؤال محدد عن الصورة، اختياري — افتراضياً وصف عام).
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return f"لم أجد الصورة في المسار: {image_path}"

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return f"الملف '{image_path}' مش صورة مدعومة (المسموح: png, jpg, jpeg, webp)."

        mime = "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")

        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        # نستخدم نموذج Vision مباشرة عبر crewai.LLM.call بدل الاعتماد على
        # multimodal=True في CrewAI (فيها مشاكل معروفة بتعامل الصورة كنص عادي).
        vision_llm = get_llm(model_name=VISION_MODEL_NAME, temperature=0.3)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64_data}"},
                    },
                ],
            }
        ]

        response = vision_llm.call(messages=messages)
        return str(response)
    except Exception as e:
        return f" خطأ أثناء تحليل الصورة: {str(e)}"


def create_orchestrator_agent(temperature: float = 0.4):
    llm = get_llm(temperature=temperature)

    agent = Agent(
        role="المستشار الرئيسي لـ AlmostachAR",
        goal=(
            "مساعدة المستخدم بمعالجة اللغة العربية الطبيعية: أجيبي مباشرة على "
            "الأسئلة العامة بالفصحى، أو استخدمي الأداة المناسبة عند الحاجة "
            "الفعلية (بيانات، صورة، أو رد بلهجة عامية)."
        ),
        backstory=(
            "مستشارة ذكاء اصطناعي لهندسة NLP العربي. عندك أدوات لفحص/تنظيف "
            "البيانات، اختيار النماذج، التدريب، التقييم، تحليل الصور، والرد "
            "بلهجة عامية (جزائرية/تونسية/مغربية/مصرية/خليجية) لو المستخدم "
            "كتب بلهجة. جاوبي بالفصحى مباشرة على أي سؤال عام بدون أدوات. "
            "استخدمي أداة فقط لما الطلب يحتاج فعلياً ملف بيانات، صورة "
            "مرفوعة، أو المستخدم يكتب/يطلب لهجة معينة — حتى بدون طلب صريح "
            "لو لاحظتِ لهجة واضحة بكلامه. لخّصي نتائج الأدوات بإيجاز طبيعي."
        ),
        tools=[
            delegate_to_inspector,
            delegate_to_preprocessor,
            delegate_to_advisor,
            delegate_to_trainer,
            delegate_to_evaluator,
            delegate_to_dialect_agent,
            analyze_image,
        ],
        llm=llm,
        verbose=True,
    )
    return agent


def create_chat_task(agent, user_message: str, conversation_history: str = "",
                      file_path: str = "", depth: str = "مفصل", specialty_hint: str = ""):
    """
    ينشئ مهمة محادثة واحدة بناءً على رسالة المستخدم الحالية، مع سياق المحادثة
    السابقة (اختياري)، مسار ملف مرفوع (اختياري)، عمق الإجابة المطلوب
    (موجز/مفصل)، وتلميح تخصص اختياري يفضّله المستخدم حالياً.
    """
    context_section = ""
    if conversation_history:
        context_section += f"\nسياق المحادثة السابقة (للاستئناس فقط):\n{conversation_history}\n"

    if specialty_hint:
        context_section += f"\nالمستخدم مركّز حالياً على: {specialty_hint} (استأنسي بهذا، بدون ما تتجاهلي طلبه الفعلي لو كان مختلف).\n"

    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            context_section += (
                f"\nالمستخدم رفع صورة متاحة في هذا المسار: {file_path}\n"
                "لو سؤاله يخص هذي الصورة أو طلب وصفها/تحليلها، استخدمي أداة "
                "'Analyze Image' مباشرة بالمسار أعلاه.\n"
            )
        else:
            context_section += f"\nملف مرفوع من المستخدم متاح في هذا المسار: {file_path}\n"

    depth_instruction = (
        "أجيبي بإيجاز شديد (جملتين لثلاث جمل بحد أقصى)، بدون تفاصيل زايدة."
        if depth == "موجز"
        else "أجيبي بتفصيل كافٍ يغطي النقاط المهمة بشكل واضح ومنظم."
    )

    task = Task(
        description=(
            f"رسالة المستخدم الحالية: \"{user_message}\"\n"
            f"{context_section}\n"
            f"مستوى التفصيل المطلوب: {depth_instruction}\n\n"
            "افهمي قصد المستخدم وردي بشكل طبيعي ومباشر. إذا كان السؤال عاماً، "
            "جاوبي من معرفتك مباشرة بدون استخدام أي أداة. إذا كان الطلب يتطلب "
            "عملية فعلية على ملف بيانات، استخدمي أداة التفويض المناسبة "
            "(تأكدي من وجود مسار ملف صالح قبل الاستدعاء؛ لو ما فيش ملف مرفوع "
            "واحتجتيه، اطلبي من المستخدم رفعه بدل افتراض مسار وهمي). إذا كان "
            "الملف المرفوع صورة، استخدمي أداة تحليل الصور بدل أدوات البيانات."
        ),
        expected_output=(
            "رد طبيعي ومباشر بالعربية، مناسب لواجهة محادثة، يجاوب على طلب "
            "المستخدم أو يلخص نتيجة الوكيل المتخصص أو تحليل الصورة بشكل مفهوم، "
            "وبمستوى التفصيل المطلوب."
        ),
        agent=agent,
    )
    return task


import re
import time


def _parse_retry_wait_seconds(error_text: str, default_wait: int) -> int:
    """
    Groq بيرجع بنص الخطأ وقت الانتظار الفعلي المطلوب، مثلاً:
    'Please try again in 6m 11.52s'. نحاول نقرأه عشان ننتظر بالضبط
    المدة الصحيحة بدل رقم ثابت تخميني.
    """
    match = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", error_text)
    if match:
        minutes = int(match.group(1)) if match.group(1) else 0
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds) + 2  # هامش أمان بسيط
    return default_wait


def run_chat_turn(user_message: str, conversation_history: str = "", file_path: str = "",
                   temperature: float = 0.4, depth: str = "مفصل", specialty_hint: str = "",
                   max_retries: int = 4, retry_wait_seconds: int = 20):
    """
    نقطة الدخول الرئيسية: تشغّل دورة محادثة واحدة وترجع رد المدير كنص.
    تُستخدم مباشرة من واجهة Streamlit. تعيد المحاولة تلقائياً عند تجاوز حد
    الاستخدام (Rate Limit) من Groq، وتحترم وقت الانتظار الفعلي اللي يطلبه
    Groq نفسه بدل انتظار ثابت.
    """
    orchestrator = create_orchestrator_agent(temperature=temperature)
    task = create_chat_task(orchestrator, user_message, conversation_history, file_path,
                             depth=depth, specialty_hint=specialty_hint)
    crew = Crew(agents=[orchestrator], tasks=[task], verbose=True)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            error_text = str(e)
            last_error = error_text
            is_rate_limit = "RateLimitError" in error_text or "rate_limit" in error_text.lower()

            if is_rate_limit and attempt < max_retries:
                wait_time = _parse_retry_wait_seconds(error_text, retry_wait_seconds)
                time.sleep(wait_time)
                continue
            else:
                break

    if last_error and ("RateLimitError" in last_error or "rate_limit" in last_error.lower()):
        return (
            "⏳ عذراً، وصلنا لحد الاستخدام المسموح مؤقتاً من مزوّد النموذج (Groq) "
            "(الخطة المجانية محدودة جداً بعدد التوكنات بالدقيقة). جربي تبعتي "
            "رسالتك مرة أخرى بعد دقيقة تقريباً."
        )
    return f" حدث خطأ غير متوقع: {last_error}"


if __name__ == "__main__":
    print(" AlmostachAR — وضع المحادثة التجريبي (اكتبي 'خروج' للإنهاء)\n")

    history = ""
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "sample.csv"
    )

    while True:
        user_input = input("أنتِ: ").strip()
        if user_input.lower() in ["خروج", "exit", "quit"]:
            print("مع السلامة! ")
            break

        response = run_chat_turn(user_input, conversation_history=history, file_path=test_file)
        print(f"\nAlmostachAR: {response}\n")

        history += f"المستخدم: {user_input}\nAlmostachAR: {response}\n\n"