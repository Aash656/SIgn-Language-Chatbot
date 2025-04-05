from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load fine-tuned model and tokenizer
model_path = "models/nlp_model/T5model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load input text (one sentence per line)
with open("outputs/output_from_sign_model.txt", "r") as f:
    asl_sentences = f.readlines()

results = []

# Predict
for sentence in asl_sentences:
    input_ids = tokenizer(sentence.strip(), return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=64)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    results.append(prediction)

# Save to final_output.txt
with open("outputs/final_output.txt", "w") as f:
    for line in results:
        f.write(line + "\n")

print("✅ Output saved to final_output.txt")
