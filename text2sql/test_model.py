from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./exports/qwen-0.5b-text2sql-merged")
model = AutoModelForCausalLM.from_pretrained("./exports/qwen-0.5b-text2sql-merged")

# Test query
prompt = "Convert the following natural language query to SQL: Show me all products with price greater than 100"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)