import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import time
from typing import List, Dict

class ModelComparator:
    def __init__(self, base_model_path: str, finetuned_model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base model
        print("Loading base model...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load fine-tuned model
        print("Loading fine-tuned model...")
        self.ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        self.ft_model = AutoModelForCausalLM.from_pretrained(
            finetuned_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate_sql(self, model, tokenizer, question: str, max_length: int = 200) -> str:
        prompt = f"Convert the following natural language query to SQL: {question}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract SQL part (remove the prompt)
        sql_output = response[len(prompt):].strip()
        return sql_output
    
    def compare_on_examples(self, test_examples: List[Dict], num_examples: int = 10):
        results = []
        
        for i, example in enumerate(test_examples[:num_examples]):
            question = example['question']
            ground_truth = example['query']
            
            print(f"\n{'='*60}")
            print(f"Example {i+1}: {question}")
            print(f"Ground Truth SQL: {ground_truth}")
            
            # Get base model prediction
            start_time = time.time()
            base_prediction = self.generate_sql(self.base_model, self.base_tokenizer, question)
            base_time = time.time() - start_time
            
            # Get fine-tuned model prediction
            start_time = time.time()
            ft_prediction = self.generate_sql(self.ft_model, self.ft_tokenizer, question)
            ft_time = time.time() - start_time
            
            print(f"\nBase Model Output ({base_time:.2f}s):")
            print(f"  {base_prediction}")
            print(f"\nFine-tuned Model Output ({ft_time:.2f}s):")
            print(f"  {ft_prediction}")
            
            # Simple accuracy check (exact match)
            base_exact = ground_truth.strip().lower() == base_prediction.strip().lower()
            ft_exact = ground_truth.strip().lower() == ft_prediction.strip().lower()
            
            result = {
                'question': question,
                'ground_truth': ground_truth,
                'base_prediction': base_prediction,
                'ft_prediction': ft_prediction,
                'base_exact_match': base_exact,
                'ft_exact_match': ft_exact,
                'base_time': base_time,
                'ft_time': ft_time
            }
            results.append(result)
            
            print(f"\nExact Match - Base: {base_exact}, Fine-tuned: {ft_exact}")
        
        return results
    
    def calculate_metrics(self, results: List[Dict]):
        base_exact_matches = sum(r['base_exact_match'] for r in results)
        ft_exact_matches = sum(r['ft_exact_match'] for r in results)
        total = len(results)
        
        base_accuracy = base_exact_matches / total * 100
        ft_accuracy = ft_exact_matches / total * 100
        
        avg_base_time = sum(r['base_time'] for r in results) / total
        avg_ft_time = sum(r['ft_time'] for r in results) / total
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Total Examples: {total}")
        print(f"Base Model Accuracy: {base_accuracy:.1f}% ({base_exact_matches}/{total})")
        print(f"Fine-tuned Model Accuracy: {ft_accuracy:.1f}% ({ft_exact_matches}/{total})")
        print(f"Improvement: {ft_accuracy - base_accuracy:.1f} percentage points")
        print(f"Average Base Model Time: {avg_base_time:.2f}s")
        print(f"Average Fine-tuned Time: {avg_ft_time:.2f}s")
        
        return {
            'base_accuracy': base_accuracy,
            'ft_accuracy': ft_accuracy,
            'improvement': ft_accuracy - base_accuracy,
            'base_avg_time': avg_base_time,
            'ft_avg_time': avg_ft_time
        }

# Usage example
if __name__ == "__main__":
    # Load test data
    print("Loading Spider test data...")
    dataset = load_dataset('xlangai/spider')
    test_examples = list(dataset['validation'])  # Convert to list of dicts
    
    # Initialize comparator
    comparator = ModelComparator(
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        finetuned_model_path="./exports/qwen-0.5b-text2sql-merged"
    )
    
    # Run comparison
    results = comparator.compare_on_examples(test_examples, num_examples=20)
    metrics = comparator.calculate_metrics(results)
    
    # Save detailed results
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to model_comparison_results.json")