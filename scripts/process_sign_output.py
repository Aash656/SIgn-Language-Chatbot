import ast

# Load class labels
with open("models/sign_model/labels.txt", "r") as f:
    labels = {int(line.split()[0]): line.split()[1] for line in f.readlines()}

# Read model output as list
with open("outputs/predicted_labels.txt", "r") as f:
    predicted_labels = ast.literal_eval(f.read())

# Convert labels to a string, handling special cases
output_string = ""
for label in predicted_labels:
    if label == "[SPACE]":
        output_string += " "
    elif label == "[NOTHING]":
        continue  # Skip
    else:
        output_string += label

# Save the result to a file
with open("outputs/output_from_sign_model.txt", "w") as f:
    f.write(output_string)

print("Output created!")
