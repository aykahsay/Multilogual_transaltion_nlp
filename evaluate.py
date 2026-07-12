import pandas as pd
import torch
import evaluate
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned MT Model")
    parser.add_argument("--model_path", type=str, default="models/psa-en-sw-finetuned", help="Path to the fine-tuned model")
    parser.add_argument("--test_data", type=str, default="data/test.csv", help="Path to test data")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = MarianTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Loading test data from {args.test_data}...")
    df_test = pd.read_csv(args.test_data)
    
    # We will evaluate on a subset if the test set is too large to save time in POC
    # Let's take 200 random samples for evaluation
    if len(df_test) > 200:
        df_test = df_test.sample(200, random_state=42)
        
    inputs = df_test["English"].tolist()
    references = df_test["Kiswahili"].tolist()
    
    print("Generating translations...")
    predictions = []
    
    # Process in batches
    batch_size = 16
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        
        encoded = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**encoded, max_length=128)
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        
        if i % (batch_size * 5) == 0:
            print(f"Translated {i}/{len(inputs)}")

    print("\nComputing metrics...")
    # Load sacrebleu metric from Hugging Face evaluate library
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    
    # Sacrebleu expects references as a list of lists (multiple references per prediction)
    refs_formatted = [[ref] for ref in references]
    
    bleu_results = sacrebleu.compute(predictions=predictions, references=refs_formatted)
    chrf_results = chrf.compute(predictions=predictions, references=refs_formatted)
    
    print("\n--- Evaluation Results ---")
    print(f"BLEU Score: {bleu_results['score']:.2f}")
    print(f"chrF Score: {chrf_results['score']:.2f}")
    
    # Show a few examples
    print("\n--- Example Translations ---")
    for i in range(3):
        print(f"Input (EN):  {inputs[i]}")
        print(f"Target (SW): {references[i]}")
        print(f"Pred (SW):   {predictions[i]}")
        print("-" * 30)

if __name__ == "__main__":
    main()
