from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import evaluate
import re

# Load the model and tokenizer from Hugging Face repository
model_name = "Aash656/t5-gloss2english"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load dataset
raw_dataset = load_dataset("achrafothman/aslg_pc12")["train"]
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
validation_data = dataset["test"]

# Gloss cleaning function
def clean_gloss(gloss):
    gloss = re.sub(r'X-', '', gloss)
    gloss = re.sub(r'[-\n]+', ' ', gloss)
    gloss = re.sub(r'\s+', ' ', gloss).strip()
    return gloss

# Metrics for evaluation
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Evaluation loop
predictions = []
references = []
batch_size = 8  # Optional, not used in loop below (single example at a time)

# Start processing and add progress logging
for i, example in enumerate(validation_data):
    gloss = clean_gloss(example["gloss"])
    input_text = f"translate gloss to english: {gloss}"
    
    # Tokenize inputs
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Generate predictions
    outputs = model.generate(**inputs, max_length=128)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    label = example["text"].strip()
    
    predictions.append(pred)
    references.append([label])  # Keep as list of list for BLEU

    if i % 100 == 0:
        print(f"Processed {i}/{len(validation_data)} examples")

# Compute evaluation metrics
bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=[r[0] for r in references])

# Output the results
print("BLEU:", bleu_score["bleu"])
print("ROUGE-L:", rouge_score["rougeL"])
