# وكيل تحضير وتوليد سكريبت تدريب النموذج

import os
import sys
import json
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
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        return pd.DataFrame({"text": lines})
    else:
        raise ValueError("صيغة الملف غير مدعومة. استخدمي CSV, Excel, أو TXT.")

@tool("Dataset Splitter")
def split_dataset(file_path: str, text_column: str = None, label_column: str = None,
                   train_ratio: float = 0.8, output_dir: str = None) -> str:
    """
    يقسّم الداتاسيت إلى مجموعتي تدريب (train) واختبار (test) بنسبة محددة،
    ويحفظهما كملفين منفصلين جاهزين للتدريب.
    المدخلات: file_path, text_column (اختياري), label_column (اختياري),
    train_ratio (نسبة التدريب، افتراضي 0.8), output_dir (اختياري)
    """
    try:
        df = _read_any_file(file_path)

        if text_column is None or text_column not in df.columns:
            text_columns = df.select_dtypes(include="object").columns
            if len(text_columns) == 0:
                return " لم يتم العثور على عمود نصي."
            text_column = text_columns[0]

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        if output_dir is None:
            output_dir = os.path.dirname(file_path)

        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")

        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

        return (
            f" تم تقسيم الداتاسيت بنجاح:\n"
            f"- عدد صفوف التدريب: {len(train_df)} → {train_path}\n"
            f"- عدد صفوف الاختبار: {len(test_df)} → {test_path}\n"
            f"- عمود النص المستخدم: {text_column}"
        )

    except Exception as e:
        return f" خطأ أثناء تقسيم الداتاسيت: {str(e)}"

@tool("Training Script Generator")
def generate_training_script(model_hf_path: str, train_path: str, test_path: str,
                              text_column: str = "text", label_column: str = "label",
                              num_labels: int = 2, output_script_path: str = None,
                              epochs: int = 3, batch_size: int = 8) -> str:
    """
    يكتب سكريبت بايثون كامل وجاهز للتشغيل يقوم بـ Fine-tuning للنموذج المحدد
    باستخدام مكتبة Hugging Face Transformers، ويحفظه كملف .py.
    المدخلات: model_hf_path (مسار النموذج على Hugging Face)، train_path، test_path،
    text_column، label_column، num_labels (عدد الفئات)، epochs، batch_size
    """
    try:
        if output_script_path is None:
            output_script_path = os.path.join(
                os.path.dirname(train_path), "run_training.py"
            )

        script_dir = os.path.dirname(output_script_path)
        if script_dir and not os.path.exists(script_dir):
            os.makedirs(script_dir, exist_ok=True)

        script_content = f'''"""
سكريبت تدريب تم توليده تلقائياً بواسطة Training Agent
النموذج: {model_hf_path}
"""

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_PATH = "{model_hf_path}"
TRAIN_PATH = r"{train_path}"
TEST_PATH = r"{test_path}"
TEXT_COLUMN = "{text_column}"
LABEL_COLUMN = "{label_column}"
NUM_LABELS = {num_labels}
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
OUTPUT_DIR = "./trained_model"

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

def main():
    print("تحميل التوكنايزر والنموذج...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=NUM_LABELS
    )

    print("تحميل البيانات...")
    train_dataset, test_dataset = load_data()

    def tokenize_function(examples):
        return tokenizer(
            examples[TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    if LABEL_COLUMN in train_dataset.column_names:
        train_dataset = train_dataset.rename_column(LABEL_COLUMN, "labels")
        test_dataset = test_dataset.rename_column(LABEL_COLUMN, "labels")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("بدء التدريب...")
    trainer.train()

    print(f"حفظ النموذج المدرَّب في {{OUTPUT_DIR}}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("انتهى التدريب بنجاح!")

if __name__ == "__main__":
    main()
'''

        with open(output_script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        return (
            f" تم توليد سكريبت التدريب بنجاح في: {output_script_path}\n"
            f"النموذج المستخدم: {model_hf_path}\n"
            f"عدد الحقب (epochs): {epochs}\n"
            f"حجم الدفعة (batch size): {batch_size}\n"
            f" ملاحظة: هذا السكريبت يحتاج عمود اسمه '{label_column}' في البيانات "
            f"يحتوي على الفئات (labels) كأرقام صحيحة تبدأ من 0."
        )

    except Exception as e:
        return f" خطأ أثناء توليد سكريبت التدريب: {str(e)}"

@tool("Training Config Validator")
def validate_training_readiness(file_path: str, label_column: str = None) -> str:
    """
    يتحقق إذا كانت البيانات جاهزة فعلياً للتدريب (وجود عمود labels، عدد كافٍ من الصفوف،
    توازن الفئات) قبل توليد سكريبت التدريب، لتفادي أخطاء لاحقة.
    """
    try:
        df = _read_any_file(file_path)
        issues = []

        if len(df) < 10:
            issues.append(f"عدد الصفوف قليل جداً ({len(df)}). يُفضّل 50 صف على الأقل للتجربة، وآلاف للتدريب الحقيقي.")

        if label_column and label_column in df.columns:
            label_counts = df[label_column].value_counts()
            if len(label_counts) < 2:
                issues.append(f"عمود الفئات '{label_column}' يحتوي على فئة واحدة فقط، وهذا غير كافٍ للتصنيف.")
            else:
                min_class = label_counts.min()
                max_class = label_counts.max()
                if max_class / min_class > 5:
                    issues.append(
                        f"عدم توازن كبير بين الفئات (الأكبر {max_class} مقابل الأصغر {min_class}). "
                        "قد يحتاج الأمر Data Augmentation أو موازنة."
                    )
        elif label_column:
            issues.append(f"عمود الفئات '{label_column}' غير موجود في الملف. الأعمدة المتاحة: {list(df.columns)}")
        else:
            issues.append("لم يتم تحديد عمود الفئات (label_column). لا يمكن تقييم جاهزية التدريب بالكامل بدونه.")

        if not issues:
            return f" البيانات جاهزة للتدريب. عدد الصفوف: {len(df)}."
        else:
            report = " ملاحظات قبل التدريب:\n"
            for issue in issues:
                report += f"- {issue}\n"
            return report

    except Exception as e:
        return f" خطأ أثناء التحقق من جاهزية البيانات: {str(e)}"

def create_trainer_agent():
    llm = get_llm(temperature=0.2)

    agent = Agent(
        role="مهندسة تدريب النماذج العربية",
        goal=(
            "تحضير البيانات وتقسيمها، والتحقق من جاهزيتها للتدريب، وتوليد سكريبت "
            "تدريب (Fine-tuning) كامل وقابل للتشغيل بناءً على النموذج الموصى به."
        ),
        backstory=(
            "أنتِ مهندسة تعلم آلي متخصصة في تدريب نماذج Transformers على بيانات عربية. "
            "تعرفين إن التدريب بدون تحقق مسبق من جودة البيانات مضيعة للوقت والموارد، "
            "فدائماً تتحققين أولاً قبل توليد أي سكريبت. تكتبين كود نظيف وموثّق وجاهز "
            "للتشغيل المباشر بدون أخطاء."
        ),
        tools=[split_dataset, generate_training_script, validate_training_readiness],
        llm=llm,
        verbose=True,
    )
    return agent

def create_training_prep_task(agent, file_path, model_hf_path, text_column="text",
                               label_column="label", num_labels=2):
    task = Task(
        description=(
            f"حضّري عملية تدريب نموذج بناءً على المعطيات التالية:\n"
            f"- ملف البيانات: {file_path}\n"
            f"- النموذج المُختار: {model_hf_path}\n"
            f"- عمود النص: {text_column}\n"
            f"- عمود الفئات: {label_column}\n"
            f"- عدد الفئات: {num_labels}\n\n"
            "الخطوات المطلوبة بالترتيب:\n"
            "1. استخدمي أداة 'Training Config Validator' للتحقق من جاهزية البيانات.\n"
            "2. إذا كانت البيانات جاهزة (أو حتى لو فيها ملاحظات بسيطة)، استخدمي أداة "
            "'Dataset Splitter' لتقسيم البيانات لتدريب واختبار.\n"
            "3. استخدمي أداة 'Training Script Generator' لتوليد سكريبت التدريب الكامل.\n"
            "4. اكتبي ملخصاً نهائياً بالعربية يوضح: نتيجة التحقق، تفاصيل التقسيم، "
            "ومسار سكريبت التدريب النهائي، وكيفية تشغيله."
        ),
        expected_output=(
            "ملخص بالعربية يغطي: حالة جاهزية البيانات، تفاصيل تقسيم التدريب/الاختبار، "
            "ومسار سكريبت التدريب الجاهز للتشغيل."
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
        print(f" الملف التجريبي غير موجود في: {test_file_path}")
    else:
        trainer = create_trainer_agent()
        # مثال: نموذج AraBERT اللي اقترحه model_advisor في التجربة السابقة
        task = create_training_prep_task(
            trainer,
            test_file_path,
            model_hf_path="aubmindlab/bert-base-arabertv2",
            text_column="text",
            label_column="label",
            num_labels=2,
        )

        crew = Crew(
            agents=[trainer],
            tasks=[task],
            verbose=True,
        )

        result = crew.kickoff()
        print("\n" + "=" * 50)
        print("النتيجة النهائية:")
        print("=" * 50)
        print(result)