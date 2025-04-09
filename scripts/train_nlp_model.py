from datasets import load_dataset
from transformers import T5Tokenizer
from tqdm import tqdm

# Step 1: Load your dataset (adjust accordingly)
dataset = load_dataset("your_dataset_name")

# Step 2: Initialize the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Step 3: Preprocess function to tokenize and handle padding/truncation
def preprocess(examples):
    # Tokenizing the 'text' column of your dataset
    tokenized_input = tokenizer(
        examples['text'],  # Assuming 'text' is the column name in your dataset
        padding=True,       # Pad sequences to the longest sequence in the batch
        truncation=True,    # Truncate sequences that exceed max_length
        max_length=512      # Fixed length for consistency (adjust as needed)
    )
    return tokenized_input

# Step 4: Map the preprocess function to your dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Step 5: Check if the tokenization worked (optional)
print(tokenized_dataset)

# Step 6: Define your training function (if you want to use the tokenized dataset)
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

# Example of model initialization (adjust according to your needs)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Step 7: Training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Output directory
    num_train_epochs=3,               # Number of epochs
    per_device_train_batch_size=16,   # Batch size for training
    per_device_eval_batch_size=16,    # Batch size for evaluation
    warmup_steps=500,                 # Warmup steps
    weight_decay=0.01,                # Weight decay to prevent overfitting
    logging_dir="./logs",             # Directory for logs
    logging_steps=10,                 # Log every 10 steps
)

# Step 8: Trainer setup (for fine-tuning)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # Adjust to your dataset split
    eval_dataset=tokenized_dataset["test"],    # Adjust to your dataset split
)

# Step 9: Start training (or you can call trainer.evaluate() if just testing)
trainer.train()

# Optionally save the model after training
model.save_pretrained("./final_model")
