# الملف الرئيسي الذي يربط الوكلاء الخمسة بالتسلسل

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crewai import Crew

from agents.data_inspector import create_data_inspector_agent, create_inspection_task
from agents.preprocessor import create_preprocessor_agent, create_preprocessing_task
from agents.model_advisor import create_model_advisor_agent, create_advisory_task
from agents.trainer import create_trainer_agent, create_training_prep_task
from agents.evaluator import create_evaluator_agent, create_evaluation_task

WAIT_BETWEEN_AGENTS_SECONDS = 25  # فاصل زمني بين كل وكيل والتالي لتفادي rate limit

def _run_single_agent_task(agent, task, step_name: str):
    """يشغّل وكيل واحد كـ Crew مستقلة، ويرجع النتيجة كنص."""
    print(f"\n{'='*60}")
    print(f" الخطوة: {step_name}")
    print(f"{'='*60}\n")

    mini_crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = mini_crew.kickoff()
    return result

def run_full_pipeline(
    raw_file_path: str,
    task_type: str = "classification",
    text_column: str = "text",
    label_column: str = "label",
    num_labels: int = 2,
    predictions_file_path: str = None,
):
    """
    يشغّل خط الأنابيب الكامل: فحص → تنظيف → اقتراح نموذج → تحضير تدريب → تقييم.
    كل خطوة تشتغل بشكل منفصل مع فاصل زمني لتفادي rate limits.
    """

    results = {}

    inspector = create_data_inspector_agent()
    task1 = create_inspection_task(inspector, raw_file_path)
    results["inspection"] = _run_single_agent_task(inspector, task1, "1/5 - فحص جودة البيانات")

    print(f"\n⏳ استراحة {WAIT_BETWEEN_AGENTS_SECONDS} ثانية لتفادي حد الاستخدام...")
    time.sleep(WAIT_BETWEEN_AGENTS_SECONDS)

    preprocessor = create_preprocessor_agent()
    task2 = create_preprocessing_task(preprocessor, raw_file_path, task_type=task_type)
    results["preprocessing"] = _run_single_agent_task(preprocessor, task2, "2/5 - تنظيف النصوص")

    print(f"\n⏳ استراحة {WAIT_BETWEEN_AGENTS_SECONDS} ثانية لتفادي حد الاستخدام...")
    time.sleep(WAIT_BETWEEN_AGENTS_SECONDS)

    advisor = create_model_advisor_agent()
    task3 = create_advisory_task(advisor, raw_file_path, task_type=task_type)
    results["advisory"] = _run_single_agent_task(advisor, task3, "3/5 - اقتراح النموذج المناسب")

    print(f"\n⏳ استراحة {WAIT_BETWEEN_AGENTS_SECONDS} ثانية لتفادي حد الاستخدام...")
    time.sleep(WAIT_BETWEEN_AGENTS_SECONDS)

    trainer = create_trainer_agent()
    task4 = create_training_prep_task(
        trainer,
        raw_file_path,
        model_hf_path="aubmindlab/bert-base-arabertv2",
        text_column=text_column,
        label_column=label_column,
        num_labels=num_labels,
    )
    results["training_prep"] = _run_single_agent_task(trainer, task4, "4/5 - تحضير التدريب")

    print(f"\n⏳ استراحة {WAIT_BETWEEN_AGENTS_SECONDS} ثانية لتفادي حد الاستخدام...")
    time.sleep(WAIT_BETWEEN_AGENTS_SECONDS)

    if predictions_file_path is None:
        predictions_file_path = os.path.join(
            os.path.dirname(raw_file_path), "..", "processed", "sample_predictions.csv"
        )
        predictions_file_path = os.path.normpath(predictions_file_path)

    evaluator = create_evaluator_agent()
    task5 = create_evaluation_task(
        evaluator,
        predictions_file_path,
        text_column=text_column,
        true_label_column="true_label",
        predicted_label_column="predicted_label",
    )
    results["evaluation"] = _run_single_agent_task(evaluator, task5, "5/5 - تقييم الأداء")

    return results

if __name__ == "__main__":
    default_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "raw", "sample.csv"
    )

    if not os.path.exists(default_file):
        print(f" الملف الافتراضي غير موجود في: {default_file}")
        sys.exit(1)

    all_results = run_full_pipeline(
        raw_file_path=default_file,
        task_type="classification",
        text_column="text",
        label_column="label",
        num_labels=2,
    )

    print("\n" + "=" * 60)
    print(" ملخص نتائج خط الأنابيب الكامل (AlmostachAR):")
    print("=" * 60)
    for step_name, step_result in all_results.items():
        print(f"\n--- {step_name} ---")
        print(step_result)