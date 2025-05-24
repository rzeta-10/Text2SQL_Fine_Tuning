from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
ft_model_path = "./exports/qwen-0.5b-text2sql-merged"

base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
ft_tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
ft_model = AutoModelForCausalLM.from_pretrained(ft_model_path)

test_cases = [
    {
        "question": "Show all customers from New York",
        "expected_keywords": ["SELECT", "customers", "WHERE", "New York"]
    },
    {
        "question": "Count the number of orders placed last month", 
        "expected_keywords": ["COUNT", "orders", "WHERE", "month"]
    },
    {
        "question": "Find the average price of products in electronics category",
        "expected_keywords": ["AVG", "price", "products", "electronics"]
    },
    {
        "question": "List top 5 best selling products",
        "expected_keywords": ["SELECT", "products", "ORDER BY", "LIMIT 5"]
    },
    {
        "question": "Show customers who have never placed an order",
        "expected_keywords": ["SELECT", "customers", "LEFT JOIN", "IS NULL"]
    }
]

def test_model_understanding(model, tokenizer, test_cases):
    results = []
    
    for test in test_cases:
        question = test['question']
        expected_keywords = test['expected_keywords']
        
        # Generate SQL
        prompt = f"Convert the following natural language query to SQL: {question}"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=150, temperature=0.1)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql_output = response[len(prompt):].strip()
        
        # Check if expected keywords are present
        keywords_found = sum(1 for keyword in expected_keywords 
                           if keyword.lower() in sql_output.lower())
        keyword_score = keywords_found / len(expected_keywords)
        
        results.append({
            'question': question,
            'sql_output': sql_output,
            'keyword_score': keyword_score,
            'keywords_found': keywords_found,
            'total_keywords': len(expected_keywords)
        })
        
        print(f"Question: {question}")
        print(f"Generated SQL: {sql_output}")
        print(f"Keyword Score: {keyword_score:.2f} ({keywords_found}/{len(expected_keywords)})")
        print("-" * 50)
    
    avg_score = sum(r['keyword_score'] for r in results) / len(results)
    print(f"\nAverage Keyword Score: {avg_score:.2f}")
    return results, avg_score

# Test both models
print("Testing Base Model:")
base_results, base_score = test_model_understanding(base_model, base_tokenizer, test_cases)

print("\nTesting Fine-tuned Model:")
ft_results, ft_score = test_model_understanding(ft_model, ft_tokenizer, test_cases)

print(f"\nImprovement: {ft_score - base_score:.2f}")