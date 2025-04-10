import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
import evaluate
import torch
import numpy as np
import re

# Load ASLG-PC12 dataset from Hugging Face and split into train/validation
raw_dataset = load_dataset("achrafothman/aslg_pc12")["train"]
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize
max_input_length = 128
max_target_length = 128

def clean_gloss(gloss):
    gloss = re.sub(r'X-', '', gloss)
    gloss = re.sub(r'[-\n]+', ' ', gloss)
    gloss = re.sub(r'\s+', ' ', gloss).strip()
    return gloss

def preprocess(example):
    cleaned_gloss = clean_gloss(example["gloss"])
    input_text = f"translate gloss to english: {cleaned_gloss}"
    target_text = example["text"].strip()

    inputs = tokenizer(input_text, padding="max_length", truncation=True,
                       max_length=max_input_length, return_token_type_ids=False)

    targets = tokenizer(target_text, padding="max_length", truncation=True,
                        max_length=max_target_length, return_token_type_ids=False)

    labels = targets["input_ids"]
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    inputs["labels"] = labels

    return inputs

tokenized_dataset = dataset.map(preprocess, batched=False)
global_tokenizer = tokenizer


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
    fp16=False,
    max_grad_norm=1.0,
    report_to="wandb",
    run_name=f"t5-asl-run-{int(time.time())}"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator
)

trainer.train()
trainer.save_model("models/nlp_model/T5model")
tokenizer.save_pretrained("models/nlp_model/T5model")
