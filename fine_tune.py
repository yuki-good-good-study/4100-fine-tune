import openai

# Set your API key
openai.api_key = "your-api-key"

# Upload training and validation files
train_file = openai.File.create(file=open("train_chat_formatted_data.jsonl"), purpose="fine-tune")
valid_file = openai.File.create(file=open("test_chat_formatted_data.jsonl"), purpose="fine-tune")

print("Training file uploaded with ID:", train_file["id"])
print("Validation file uploaded with ID:", valid_file["id"])

# Start the fine-tuning process
response = openai.FineTuningJob.create(
    training_file=train_file["id"],
    validation_file=valid_file["id"],
    model="gpt-3.5-turbo",  # Choose the base model (e.g., "ada", "babbage", "curie", "davinci")
)

print("Fine-tuning job created with ID:", response["id"])
