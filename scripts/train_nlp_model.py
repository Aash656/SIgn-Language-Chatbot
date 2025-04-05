from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
import pandas as pd
import torch

# Load your CSVs into Hugging Face Datasets format
train_df = pd.read_csv("data/nlp/train.csv")
eval_df = pd.read_csv("data/nlp/eval.csv")

dataset = DatasetDict({
    "train": load_dataset("csv", data_files="data/nlp/train.csv")["train"],
    "validation": load_dataset("csv", data_files="data/nlp/eval.csv")["train"]
})

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize
max_input_length = 64
max_target_length = 64

def preprocess(example):
    inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=max_input_length)
    targets = tokenizer(example["target"], padding="max_length", truncation=True, max_length=max_target_length)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="models/nlp/model_args",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# Trainer setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save final model
trainer.save_model("models/nlp_model/T5model")
tokenizer.save_pretrained("models/nlp_model/T5model")
