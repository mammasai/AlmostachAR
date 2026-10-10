import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_PATH = "aubmindlab/bert-base-arabertv2"
TRAIN_PATH = r"C:\\Users\\LAPTA\\Desktop\\second-mostachAR\\data\\processed\\train.csv"
TEST_PATH = r"C:\\Users\\LAPTA\\Desktop\\second-mostachAR\\data\\processed\\test.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUM_LABELS = 2
EPOCHS = 3
BATCH_SIZE = 8
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

    print(f"حفظ النموذج المدرَّب في {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("انتهى ")

if __name__ == "__main__":
    main()
