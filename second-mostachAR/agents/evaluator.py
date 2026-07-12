"""
 تقييم أداء النموذج المدرَّب وكتابة تقرير استشاري نهائي
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm

from crewai import Agent, Task, Crew
from crewai.tools import tool


def _read_any_file(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    else:
        raise ValueError("صيغة الملف غير مدعومة. استخدمي CSV أو Excel.")


@tool("Classification Metrics Calculator")
def compute_classification_metrics(predictions_file: str, true_label_column: str = "true_label",
                                     predicted_label_column: str = "predicted_label") -> str:
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
        )

        df = _read_any_file(predictions_file)

        if true_label_column not in df.columns or predicted_label_column not in df.columns:
            return (
                f"الأعمدة المطلوبة غير موجودة. الأعمدة المتاحة: {list(df.columns)}. "
                f"مطلوب: {true_label_column}, {predicted_label_column}"
            )

        y_true = df[true_label_column]
        y_pred = df[predicted_label_column]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        detailed_report = classification_report(y_true, y_pred, zero_division=0)

        report = "مقاييس التقييم:\n"
        report += f"- الدقة الإجمالية (Accuracy): {accuracy:.2%}\n"
        report += f"- الدقة الموزونة (Precision): {precision:.2%}\n"
        report += f"- الاستدعاء الموزون (Recall): {recall:.2%}\n"
        report += f"- مقياس F1 الموزون: {f1:.2%}\n\n"
        report += "تقرير تفصيلي لكل فئة:\n"
        report += detailed_report

        return report

    except Exception as e:
        return f"خطأ أثناء حساب المقاييس: {str(e)}"


@tool("Confusion Matrix Analyzer")
def analyze_confusion_matrix(predictions_file: str, true_label_column: str = "true_label",
                               predicted_label_column: str = "predicted_label") -> str:
    try:
        from sklearn.metrics import confusion_matrix

        df = _read_any_file(predictions_file)

        if true_label_column not in df.columns or predicted_label_column not in df.columns:
            return f"الأعمدة المطلوبة غير موجودة. الأعمدة المتاحة: {list(df.columns)}"

        y_true = df[true_label_column]
        y_pred = df[predicted_label_column]

        labels = sorted(y_true.unique().tolist())
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        report = "مصفوفة الالتباس (Confusion Matrix):\n\n"
        report += "الفئات: " + ", ".join(str(l) for l in labels) + "\n\n"

        header = "الحقيقي \\ المتوقع".ljust(20) + "".join(str(l).ljust(10) for l in labels)
        report += header + "\n"
        for i, label in enumerate(labels):
            row = str(label).ljust(20) + "".join(str(cm[i][j]).ljust(10) for j in range(len(labels)))
            report += row + "\n"

        max_confusion = 0
        confused_pair = None
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i][j] > max_confusion:
                    max_confusion = cm[i][j]
                    confused_pair = (labels[i], labels[j])

        if confused_pair and max_confusion > 0:
            report += (
                f"\nأكثر خلط ملاحظ: النموذج توقع '{confused_pair[1]}' بينما الحقيقة "
                f"كانت '{confused_pair[0]}' في {max_confusion} حالة/حالات."
            )
        else:
            report += "\nلا يوجد خلط ملحوظ بين الفئات."

        return report

    except Exception as e:
        return f"خطأ أثناء تحليل مصفوفة الالتباس: {str(e)}"


@tool("Misclassification Sampler")
def sample_misclassified_examples(predictions_file: str, text_column: str = "text",
                                    true_label_column: str = "true_label",
                                    predicted_label_column: str = "predicted_label",
                                    num_samples: int = 5) -> str:
    try:
        df = _read_any_file(predictions_file)

        required_cols = [text_column, true_label_column, predicted_label_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"أعمدة ناقصة: {missing}. الأعمدة المتاحة: {list(df.columns)}"

        misclassified = df[df[true_label_column] != df[predicted_label_column]]

        if len(misclassified) == 0:
            return "لا توجد أمثلة مُصنَّفة خطأ — النموذج حقق دقة 100% على بيانات الاختبار."

        sample = misclassified.head(num_samples)

        report = f"عينة من الأخطاء ({len(misclassified)} خطأ من أصل {len(df)}):\n\n"
        for idx, row in sample.iterrows():
            text_preview = str(row[text_column])[:100]
            report += f"- النص: \"{text_preview}...\"\n"
            report += f"  الحقيقي: {row[true_label_column]} | المتوقع: {row[predicted_label_column]}\n\n"

        error_rate = len(misclassified) / len(df)
        report += f"نسبة الخطأ الإجمالية: {error_rate:.2%}"

        return report

    except Exception as e:
        return f"خطأ أثناء سحب عينة الأخطاء: {str(e)}"


def create_evaluator_agent():
    llm = get_llm(temperature=0.3)

    agent = Agent(
        role="محللة أداء النماذج العربية",
        goal=(
            "تقييم أداء النموذج المدرَّب بدقة عبر مقاييس كمية (Accuracy, Precision, "
            "Recall, F1)، وتحليل الأخطاء الشائعة، وكتابة تقرير استشاري واضح بالعربية "
            "يشرح النتائج ويقترح تحسينات عملية."
        ),
        backstory=(
            "أنتِ خبيرة في تقييم نماذج تعلم الآلة، خصوصاً في سياق معالجة اللغة "
            "العربية الطبيعية. لا تكتفين بعرض الأرقام فقط، بل تفسرينها وتربطينها "
            "بأمثلة حقيقية من الأخطاء، وتقترحين خطوات عملية للتحسين (مثل جمع بيانات "
            "أكتر لفئة معينة، أو مراجعة جودة التسميات). أسلوبك دقيق وصادق، ولا تجمّلين "
            "نتائج ضعيفة."
        ),
        tools=[
            compute_classification_metrics,
            analyze_confusion_matrix,
            sample_misclassified_examples,
        ],
        llm=llm,
        verbose=True,
    )
    return agent


def create_evaluation_task(agent, predictions_file: str, text_column: str = "text",
                            true_label_column: str = "true_label",
                            predicted_label_column: str = "predicted_label"):
    task = Task(
        description=(
            f"قيّمي أداء النموذج بناءً على ملف التنبؤات الموجود في: {predictions_file}\n"
            f"عمود النص: {text_column}\n"
            f"عمود القيمة الحقيقية: {true_label_column}\n"
            f"عمود القيمة المتوقعة: {predicted_label_column}\n\n"
            "الخطوات المطلوبة بالترتيب:\n"
            "1. استخدمي أداة 'Classification Metrics Calculator' لحساب المقاييس الأساسية.\n"
            "2. استخدمي أداة 'Confusion Matrix Analyzer' لتحليل أنماط الخلط بين الفئات.\n"
            "3. استخدمي أداة 'Misclassification Sampler' لسحب أمثلة توضيحية على الأخطاء.\n"
            "4. اكتبي تقريراً استشارياً نهائياً بالعربية يلخص: الأداء العام، أهم أنماط "
            "الأخطاء، وتوصيات عملية محددة لتحسين النموذج (مثل جمع بيانات إضافية، "
            "مراجعة جودة التصنيف اليدوي، أو تجربة نموذج آخر)."
        ),
        expected_output=(
            "تقرير استشاري بالعربية يغطي: المقاييس الكمية، تحليل مصفوفة الالتباس، "
            "أمثلة على الأخطاء، وتوصيات عملية واضحة ومحددة للتحسين."
        ),
        agent=agent,
    )
    return task


if __name__ == "__main__":
    test_predictions_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "sample_predictions.csv"
    )

    if not os.path.exists(test_predictions_path):
        print(f"ملف التنبؤات التجريبي غير موجود في: {test_predictions_path}")
        print("سيتم إنشاء ملف تجريبي بسيط للتحقق من عمل الوكيل...")

        sample_data = pd.DataFrame({
            "text": [
                "هذا المنتج ممتاز جداً",
                "الخدمة كانت سيئة للغاية",
                "تجربة عادية لا بأس بها",
                "أنصح الجميع بتجربة هذا",
                "لن أشتري من هنا مرة أخرى",
                "المنتج جيد لكن التوصيل تأخر",
                "رائع ومذهل حقاً",
                "لم يعجبني إطلاقاً",
            ],
            "true_label": [1, 0, 1, 1, 0, 1, 1, 0],
            "predicted_label": [1, 0, 1, 1, 1, 1, 1, 0],
        })

        os.makedirs(os.path.dirname(test_predictions_path), exist_ok=True)
        sample_data.to_csv(test_predictions_path, index=False, encoding="utf-8-sig")
        print(f"تم إنشاء ملف تجريبي في: {test_predictions_path}")

    evaluator = create_evaluator_agent()
    task = create_evaluation_task(evaluator, test_predictions_path)

    crew = Crew(
        agents=[evaluator],
        tasks=[task],
        verbose=True,
    )

    result = crew.kickoff()
    print("\n" + "=" * 50)
    print("النتيجة النهائية:")
    print("=" * 50)
    print(result)