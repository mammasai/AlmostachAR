# وكيل المدير الذي يوجّه المحادثة للوكلاء المتخصصين

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm

from crewai import Agent, Task, Crew
from crewai.tools import tool

from agents.data_inspector import create_data_inspector_agent, create_inspection_task
from agents.preprocessor import create_preprocessor_agent, create_preprocessing_task
from agents.model_advisor import create_model_advisor_agent, create_advisory_task
from agents.trainer import create_trainer_agent, create_training_prep_task
from agents.evaluator import create_evaluator_agent, create_evaluation_task

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
def delegate_to_preprocessor(file_path: str, task_type: str = "classification") -> str:
    """
    تستدعي وكيلة تنظيف النصوص العربية المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم
    تنظيف بيانات، إزالة تشكيل، تطبيع نصوص، أو تجهيز ملف لمهمة تدريب.
    المدخلات: file_path، task_type (مثال: 'classification' أو 'tts').
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
def delegate_to_advisor(file_path: str, task_type: str = "classification") -> str:
    """
    تستدعي المستشارة المتخصصة في اختيار النماذج اللغوية العربية. استخدمي هذه الأداة
    لما يسأل المستخدم عن أنسب نموذج (AraBERT, CAMeLBERT, MarBERT) لمهمته.
    المدخلات: file_path، task_type.
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
def delegate_to_trainer(file_path: str, model_hf_path: str, text_column: str = "text",
                         label_column: str = "label", num_labels: int = 2) -> str:
    """
    تستدعي مهندسة التدريب المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم تحضير
    بيانات للتدريب، تقسيم داتاسيت، أو توليد سكريبت تدريب (Fine-tuning) كامل.
    المدخلات: file_path، model_hf_path (مسار النموذج على Hugging Face)،
    text_column، label_column، num_labels.
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
def delegate_to_evaluator(predictions_file: str, text_column: str = "text",
                           true_label_column: str = "true_label",
                           predicted_label_column: str = "predicted_label") -> str:
    """
    تستدعي محللة الأداء المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم تقييم أداء
    نموذج مدرَّب، حساب مقاييس (Accuracy, F1)، أو تحليل أخطاء التصنيف.
    المدخل: predictions_file (ملف فيه القيم الحقيقية والمتوقعة).
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

def create_orchestrator_agent():
    llm = get_llm(temperature=0.4)

    agent = Agent(
        role="المستشار الرئيسي لـ AlmostachAR",
        goal=(
            "التحدث مع المستخدم بشكل طبيعي وودود كمساعد محادثة، وفهم قصده بدقة، "
            "ثم اتخاذ القرار الصحيح: إما الإجابة مباشرة إذا كان السؤال عاماً "
            "(مثل شرح مفهوم في Arabic NLP)، أو تفويض المهمة للوكيل المتخصص المناسب "
            "إذا كانت تتطلب فحص أو معالجة أو تحليل ملف بيانات فعلي."
        ),
        backstory=(
            "أنتِ الواجهة الرئيسية لمنصة AlmostachAR — مستشار ذكاء اصطناعي متكامل "
            "لهندسة معالجة اللغة العربية الطبيعية. عندك فريق من خمس خبيرات متخصصات "
            "تحت إمرتك: خبيرة فحص البيانات، خبيرة تنظيف النصوص، مستشارة اختيار "
            "النماذج، مهندسة التدريب، ومحللة الأداء. مهمتك إنك تكوني الوسيطة الذكية: "
            "لما المستخدم يسأل سؤال عام (مثلاً 'شنو الفرق بين AraBERT وCAMeLBERT؟')، "
            "جاوبي مباشرة من معرفتك بدون ما تستدعي أي أداة. لكن لما المستخدم يطلب "
            "عملية فعلية على ملف (فحص، تنظيف، تدريب، تقييم)، استخدمي أداة التفويض "
            "المناسبة واستدعي الخبيرة المختصة. لو الطلب يحتاج أكثر من خبيرة بالتسلسل "
            "(مثلاً فحص ثم تنظيف)، استدعيهم وحدة وحدة بالترتيب المنطقي. تحدثي دائماً "
            "بالعربية بأسلوب طبيعي ومباشر، ولخصي تقارير الخبيرات بشكل واضح ومفهوم "
            "للمستخدم بدل نسخها حرفياً بكل تفاصيلها التقنية."
        ),
        tools=[
            delegate_to_inspector,
            delegate_to_preprocessor,
            delegate_to_advisor,
            delegate_to_trainer,
            delegate_to_evaluator,
        ],
        llm=llm,
        verbose=True,
    )
    return agent

def create_chat_task(agent, user_message: str, conversation_history: str = "",
                      file_path: str = ""):
    """
    ينشئ مهمة محادثة واحدة بناءً على رسالة المستخدم الحالية، مع سياق المحادثة
    السابقة (اختياري) ومسار ملف مرفوع (اختياري) إن وُجد.
    """
    context_section = ""
    if conversation_history:
        context_section += f"\nسياق المحادثة السابقة (للاستئناس فقط):\n{conversation_history}\n"
    if file_path:
        context_section += f"\nملف مرفوع من المستخدم متاح في هذا المسار: {file_path}\n"

    task = Task(
        description=(
            f"رسالة المستخدم الحالية: \"{user_message}\"\n"
            f"{context_section}\n"
            "افهمي قصد المستخدم وردي بشكل طبيعي ومباشر. إذا كان السؤال عاماً، "
            "جاوبي من معرفتك مباشرة بدون استخدام أي أداة. إذا كان الطلب يتطلب "
            "عملية فعلية على ملف بيانات، استخدمي أداة التفويض المناسبة "
            "(تأكدي من وجود مسار ملف صالح قبل الاستدعاء؛ لو ما فيش ملف مرفوع "
            "واحتجتيه، اطلبي من المستخدم رفعه بدل افتراض مسار وهمي)."
        ),
        expected_output=(
            "رد طبيعي ومباشر بالعربية، مناسب لواجهة محادثة، يجاوب على طلب "
            "المستخدم أو يلخص نتيجة الوكيل المتخصص المُستدعى بشكل مفهوم."
        ),
        agent=agent,
    )
    return task

import time

def run_chat_turn(user_message: str, conversation_history: str = "", file_path: str = "",
                   max_retries: int = 3, retry_wait_seconds: int = 15):
    """
    نقطة الدخول الرئيسية: تشغّل دورة محادثة واحدة وترجع رد المدير كنص.
    تُستخدم مباشرة من واجهة Streamlit. تعيد المحاولة تلقائياً عند تجاوز حد
    الاستخدام (Rate Limit) من Groq بدل ما تفشل مباشرة.
    """
    orchestrator = create_orchestrator_agent()
    task = create_chat_task(orchestrator, user_message, conversation_history, file_path)
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
                time.sleep(retry_wait_seconds)
                continue
            else:
                break

    if last_error and ("RateLimitError" in last_error or "rate_limit" in last_error.lower()):
        return (
            "⏳ عذراً، وصلنا لحد الاستخدام المسموح مؤقتاً من مزوّد النموذج (Groq). "
            "جربي تبعتي رسالتك مرة أخرى بعد دقيقة تقريباً."
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