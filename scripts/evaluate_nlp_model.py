from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import evaluate
import numpy as np
import re

# Load trained model + tokenizer
model_dir = "models/nlp_model/T5model"
model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

# Load dataset
raw_dataset = load_dataset("achrafothman/aslg_pc12")["train"]
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
validation_data = dataset["test"]

# Gloss cleaning
def clean_gloss(gloss):
    gloss = re.sub(r'X-', '', gloss)
    gloss = re.sub(r'[-\n]+', ' ', gloss)
    gloss = re.sub(r'\s+', ' ', gloss).strip()
    return gloss

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Evaluation loop
predictions = []
references = []

for example in validation_data:
    gloss = clean_gloss(example["gloss"])
    input_text = f"translate gloss to english: {gloss}"
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    label = example["text"].strip()
    
    predictions.append(pred.split())
    references.append([label.split()])

# Compute metrics
bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=[" ".join(p) for p in predictions],
                            references=[" ".join(r[0]) for r in references])

print("BLEU:", bleu_score["bleu"])
print("ROUGE-L:", rouge_score["rougeL"])
