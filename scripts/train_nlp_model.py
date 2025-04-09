import time
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
import evaluate
import torch
import numpy as np

# Function to clean gloss
def clean_gloss(gloss):
    try:
        gloss = re.sub(r'X-', '', gloss)  # Clean unwanted X- tokens
        gloss = re.sub(r'DESC-', '', gloss)  # Clean DESC- tokens
        gloss = re.sub(r'\n', ' ', gloss)  # Replace newline with space
        return gloss.strip()
    except Exception as e:
        print(f"Error processing gloss: {e}")
        return gloss  # Return the gloss as is in case of error

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
max_input_length = 64
max_target_length = 64

def preprocess(example):
    try:
        example["gloss"] = clean_gloss(example["gloss"])  # Clean gloss text
    except Exception as e:
        print(f"Error processing gloss: {e}")  # Log the error
        example["gloss"] = ""  # Default to empty string or handle differently

    inputs = tokenizer(example["gloss"], padding="max_length", truncation=True,
                       max_length=max_input_length, return_token_type_ids=False)
    targets = tokenizer(example["text"], padding="max_length", truncation=True,
                        max_length=max_target_length, return_token_type_ids=False)
    inputs["labels"] = targets["input_ids"]

    if all(token_id == tokenizer.pad_token_id for token_id in inputs["input_ids"]):
        print("⚠️ Warning: Found an all-padding input sequence.")

    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True, batch_size=32)  # Adjust batch_size as necessary

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # If predictions are logits, get argmax to get token IDs
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert to token IDs if they're still logits
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    # Replace -100 in labels to pad_token_id (to decode properly)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = bleu.compute(
        predictions=[pred.split() for pred in decoded_preds],
        references=[[label.split()] for label in decoded_labels]
    )

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_score["bleu"],
        "rougeL": rouge_result["rougeL"]
    }

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
    fp16=False,
    max_grad_norm=1.0,
    report_to="wandb",
    run_name=f"t5-asl-run-{int(time.time())}"
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
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save final model
trainer.save_model("models/nlp_model/T5model")
tokenizer.save_pretrained("models/nlp_model/T5model")
