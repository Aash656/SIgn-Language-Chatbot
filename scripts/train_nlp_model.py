from datasets import load_dataset
from transformers import T5Tokenizer

# Load dataset (modify according to your dataset)
dataset = load_dataset("your_dataset_name")

# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Preprocess function to tokenize the text with padding and truncation
def preprocess(examples):
    # Tokenize the text column
    tokenized_input = tokenizer(
        examples['text'],  # Assuming 'text' is the column name
        padding=True,       # Pad sequences to the longest sequence in the batch
        truncation=True,    # Truncate sequences that exceed max_length
        max_length=512      # Optional: Set a fixed length to avoid varying lengths
    )
    return tokenized_input

# Apply the preprocess function to the dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Check the dataset structure after preprocessing
print(tokenized_dataset)

