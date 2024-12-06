import openai

# Set your OpenAI API key
openai.api_key = "your-api-key"

# Replace with your fine-tuning job ID
job_id = "your-job-id"  # Replace with the actual job ID

# Get detailed job information
response = openai.FineTuningJob.retrieve(id=job_id)

# Print the detailed response
print("Fine-tuning job details:", response)