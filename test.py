import openai
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set your API key (recommend using environment variables for safety)
openai.api_key = "your-api-key"

# Load the test dataset
test_file_path = "test_chat_formatted_data.jsonl"  # Replace with your test dataset path
test_data = []
with open(test_file_path, "r") as file:
    for line in file:
        test_data.append(json.loads(line))

# Define the fine-tuned model ID
fine_tuned_model_id = "ft:gpt-3.5-turbo-0125:personal::AXirM1Dy"

# Get predictions from the fine-tuned model
true_labels = []
predicted_labels = []

count = 0
for entry in test_data:
    count += 1
    print(count)
    if count > 5000:  # Stop after 5000 entries
        break

    # Extract the true label from the assistant's message in the test dataset
    true_label = entry["messages"][-1]["content"]
    true_labels.append(true_label)

    # Check for repetitive patterns
    messages = entry["messages"][:-1]  # Exclude the assistant's response
    user_message = messages[-1]["content"]
    if "Analyze the following comment: Analyze the following comment:" in user_message:
        user_message = user_message.replace("Analyze the following comment: Analyze the following comment:", "Analyze the following comment:")

    # Update the messages with the cleaned prompt
    messages[-1]["content"] = user_message

    # Send the prompt to the fine-tuned model
    try:
        response = openai.ChatCompletion.create(
            model=fine_tuned_model_id,
            messages=messages,
            max_tokens=50,
            temperature=0
        )
        predicted_label = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error on entry {count}: {e}")
        predicted_label = "unknown"  # Default label if prediction fails

    predicted_labels.append(predicted_label)

# Map the labels into binary categories (e.g., hate vs. non-hate) if necessary
def map_to_binary(label):
    return "non-hate" if "non-toxic" in label else "hate"

binary_true_labels = [map_to_binary(label) for label in true_labels]
binary_predicted_labels = [map_to_binary(label) for label in predicted_labels]

# Ensure lengths match before computing metrics
if len(binary_true_labels) != len(binary_predicted_labels):
    raise ValueError("Mismatched lengths between true and predicted labels!")

# Compute metrics
precision = precision_score(binary_true_labels, binary_predicted_labels, pos_label="hate")
recall = recall_score(binary_true_labels, binary_predicted_labels, pos_label="hate")
f1 = f1_score(binary_true_labels, binary_predicted_labels, pos_label="hate")

# Confusion matrix for FPR and FNR
tn, fp, fn, tp = confusion_matrix(binary_true_labels, binary_predicted_labels, labels=["non-hate", "hate"]).ravel()

# False Positive Rate (FPR)
fpr = fp / (fp + tn)

# False Negative Rate (FNR)
fnr = fn / (fn + tp)

# Print the results
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("F1-Score:", f1)
print("False Positive Rate (FPR):", fpr)
print("False Negative Rate (FNR):", fnr)

# Generate confusion matrix
cm = confusion_matrix(binary_true_labels, binary_predicted_labels, labels=["non-hate", "hate"])

# Print the confusion matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["non-hate", "hate"], yticklabels=["non-hate", "hate"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(binary_true_labels, binary_predicted_labels, target_names=["non-hate", "hate"]))
print("Unique labels in true labels:", set(binary_true_labels))
print("Unique labels in predicted labels:", set(binary_predicted_labels))
