# Text2SQL Fine-tuning with LLaMA Factory

Fine-tune Qwen models for Text-to-SQL generation using the Spider dataset, optimized for RTX 3060 (6GB VRAM) and other consumer GPUs.

## 🎯 Results

- **Base Qwen-0.5B**: ~20% accuracy on Spider dataset
- **Fine-tuned**: ~35-50% accuracy 
- **Training Time**: 20-30 minutes on RTX 3060

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (6GB+ VRAM recommended)
- 16GB+ system RAM

## 🚀 Quick Start

### 1. Installation

```bash
# Clone this repository
git clone https://github.com/rzeta-10/text2sql-finetuning
cd text2sql-finetuning

# Install dependencies
pip install -r requirements.txt

# Install LLaMA Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
cd ..
```

### 2. Prepare Dataset

```bash
# Download and format Spider dataset
python3 prepare_spider_data.py
```

### 3. Fine-tune Model

```bash
llamafactory-cli train text2sql/text2sql_config.yaml
```

### 4. Export Fine-tuned Model

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_name_or_path ./saves/qwen-0.5b-text2sql \
    --template qwen \
    --finetuning_type lora \
    --export_dir ./exports/qwen-0.5b-text2sql-merged \
    --export_size 2 \
    --export_legacy_format False
```

### 5. Test Your Model

```bash
python3 compare_models.py
```

## 📁 Repository Structure

```
text2sql-finetuning/
├── README.md
├── text2sql/
│   ├── compare_models.py
│   ├── custom_test_cases.py
│   ├── prepare_spider_data.py
│   ├── spider_train.json
│   ├── spider_val.json
│   ├── test_model.py
│   └── text2sql_config.yaml
```

## 🔧 Configuration Options

### My RTX 3060 (6GB VRAM) Settings
- **Model**: Qwen2.5-0.5B-Instruct
- **Quantization**: 4-bit
- **Batch Size**: 1 (with gradient accumulation)
- **Sequence Length**: 1024 tokens
- **LoRA Rank**: 16


## 🛠 Troubleshooting

### CUDA Out of Memory
```bash
# Try ultra-low memory config
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
cutoff_len: 512
lora_rank: 8
max_samples: 2000
```

### Slow Training
- Ensure GPU drivers are updated
- Close other applications
- Monitor temperatures with `nvidia-smi`
- Use `fp16` instead of `bf16` for RTX 30-series

### Poor Results
- Increase training epochs (2-3)
- Check data quality
- Verify model paths are correct
- Try different learning rates (1e-4 to 5e-4)

## 📝 Usage Examples

### Basic Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./exports/qwen-0.5b-text2sql-merged")
model = AutoModelForCausalLM.from_pretrained("./exports/qwen-0.5b-text2sql-merged")

# Generate SQL
prompt = "Convert the following natural language query to SQL: Show all customers from New York"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Batch Processing
```python
questions = [
    "Show all customers from New York",
    "Count total orders placed last month", 
    "Find top 5 best selling products"
]

for question in questions:
    prompt = f"Convert the following natural language query to SQL: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200, temperature=0.1)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"SQL: {sql}\n")
```

## 🔍 Model Comparison

Run the comparison script to evaluate improvements:

```bash
python3 compare_models.py
```

**Sample Output:**
```
Example 1: Show all customers from New York
Ground Truth SQL: SELECT * FROM customers WHERE city = 'New York'

Base Model Output (0.45s):
  SELECT customers city New York FROM

Fine-tuned Model Output (0.42s):
  SELECT * FROM customers WHERE city = 'New York'

Exact Match - Base: False, Fine-tuned: True
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Spider Dataset](https://yale-lily.github.io/spider) by Yale University
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning framework
- [Qwen Models](https://huggingface.co/Qwen) by Alibaba Cloud