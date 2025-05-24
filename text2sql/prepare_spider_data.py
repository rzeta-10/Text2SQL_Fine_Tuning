from datasets import load_dataset
import json

# Load Spider dataset
dataset = load_dataset('xlangai/spider')

# Convert to LLaMA Factory format
def format_spider_example(example):
    # Get database schema info
    db_id = example['db_id']
    question = example['question']
    sql = example['query']
    
    # Create instruction
    instruction = f"Convert the following natural language query to SQL for the {db_id} database"
    
    return {
        "instruction": instruction,
        "input": question,
        "output": sql
    }

# Process training data
train_data = [format_spider_example(ex) for ex in dataset['train']]

# Save to JSON
with open('spider_train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

print(f"Saved {len(train_data)} training examples to spider_train.json")

# Process validation data for evaluation
val_data = [format_spider_example(ex) for ex in dataset['validation']]
with open('spider_val.json', 'w') as f:
    json.dump(val_data, f, indent=2)

print(f"Saved {len(val_data)} validation examples to spider_val.json")