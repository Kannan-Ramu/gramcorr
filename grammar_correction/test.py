from transformers import T5ForConditionalGeneration, T5Tokenizer

# Replace "path/to/your/saved/model" with the actual directory path where your trained model was saved.
model_directory = "model"

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_directory)
model = T5ForConditionalGeneration.from_pretrained(model_directory)
# Example input text
input_text = "grammar:  Me and him goes to the parks every Saturdayss ."

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text based on the input
output_ids = model.generate(input_ids)

# Decode the output token IDs to get the generated text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
