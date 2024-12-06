import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the CSV file
file_path = "train.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Split into train and test datasets (80:20 split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to convert a single row into chat format
def convert_row_to_chat_format(row):
    # System instruction
    system_message = {
        "role": "system",
        "content": "You are an assistant that analyzes comments for harmful language."
    }
    
    # User message
    user_message = {
        "role": "user",
        "content": f"Analyze the following comment: '{row['comment_text']}'"
    }
    
    # Assistant message (expected completion)
    analysis = []
    if row['severe_toxic']:
        analysis.append("severely toxic")
    if row['toxic'] and not row['severe_toxic']:
        analysis.append("toxic")
    if row['obscene']:
        analysis.append("obscene")
    if row['threat']:
        analysis.append("threatening")
    if row['insult']:
        analysis.append("insulting")
    if row['identity_hate']:
        analysis.append("identity-based hate speech")
    
    if not analysis:
        analysis_content = "This comment is non-toxic and inclusive."
    else:
        analysis_content = f"This comment is {', '.join(analysis)}."
    
    assistant_message = {
        "role": "assistant",
        "content": analysis_content
    }
    
    # Combine all messages into a dictionary
    return {
        "messages": [system_message, user_message, assistant_message]
    }

# Convert train and test datasets into chat format
train_chat_data = [convert_row_to_chat_format(row) for _, row in train_data.iterrows()]
test_chat_data = [convert_row_to_chat_format(row) for _, row in test_data.iterrows()]

# Save to JSONL files
def save_jsonl(data, output_path):
    with open(output_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

save_jsonl(train_chat_data, "train_chat_formatted_data.jsonl")
save_jsonl(test_chat_data, "test_chat_formatted_data.jsonl")

print("Chat-formatted train and test data saved as 'train_chat_formatted_data.jsonl' and 'test_chat_formatted_data.jsonl'.")
