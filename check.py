import openai

openai.api_key = "your-api-key"

# List all available models
models = openai.Model.list()
print("Available models:", [model["id"] for model in models["data"]])