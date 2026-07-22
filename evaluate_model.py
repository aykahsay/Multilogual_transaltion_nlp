import os
import re
import argparse
import torch
import pandas as pd
import evaluate as hf_evaluate  # Avoid collision
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Multilingual MT Model")
    parser.add_argument("--model_path", type=str, default="models/psa-multilingual-nllb-finetuned", help="Path to the fine-tuned model")
    parser.add_argument("--test_data", type=str, default="data/test.csv", help="Path to test data")
    args = parser.parse_args()
    
    # Graceful fallback to pre-trained model if fine-tuned model doesn't exist or is empty
    model_checkpoint = args.model_path
    if not os.path.exists(model_checkpoint) or (os.path.isdir(model_checkpoint) and len(os.listdir(model_checkpoint)) == 0):
        print(f"Model directory '{model_checkpoint}' not found or empty. Falling back to pre-trained 'facebook/nllb-200-distilled-600M' for evaluation.")
        model_checkpoint = "facebook/nllb-200-distilled-600M"
        
    print(f"Loading tokenizer and model from '{model_checkpoint}'...")
    tokenizer = NllbTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    # If the base model is used, ensure we still add 'guz_Latn' so tokenization doesn't fail
    if model_checkpoint == "facebook/nllb-200-distilled-600M":
        print("Adding 'guz_Latn' token to the base tokenizer...")
        tokenizer.add_special_tokens({'additional_special_tokens': ['guz_Latn']})
        model.resize_token_embeddings(len(tokenizer))
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Loading test data from {args.test_data}...")
    df_test = pd.read_csv(args.test_data)
    
    # We will test on these representative translation directions
    eval_directions = [
        ('English', 'Kiswahili', 'eng_Latn', 'swh_Latn'),
        ('English', 'Somali', 'eng_Latn', 'som_Latn'),
        ('English', 'Dholuo', 'eng_Latn', 'luo_Latn'),
        ('English', 'Ekegusii', 'eng_Latn', 'guz_Latn'),
        ('Kiswahili', 'Ekegusii', 'swh_Latn', 'guz_Latn'),
        ('Ekegusii', 'Kiswahili', 'guz_Latn', 'swh_Latn')
    ]
    
    # Load metrics from Hugging Face evaluate library
    print("Loading metrics...")
    sacrebleu = hf_evaluate.load("sacrebleu")
    chrf = hf_evaluate.load("chrf")
    
    results = []
    
    print("\nStarting evaluation across translation directions...")
    
    for src_col, tgt_col, src_lang, tgt_lang in eval_directions:
        print(f"\n--- Evaluating: {src_col} -> {tgt_col} ({src_lang} -> {tgt_lang}) ---")
        
        # Filter rows having both non-null values
        df_filtered = df_test[[src_col, tgt_col]].dropna()
        df_filtered = df_filtered[(df_filtered[src_col].str.strip() != "") & (df_filtered[tgt_col].str.strip() != "")]
        
        if len(df_filtered) == 0:
            print("No test samples available for this direction. Skipping.")
            continue
            
        # Sample to save time in local execution/testing (maximum 50 samples per direction)
        if len(df_filtered) > 50:
            df_sample = df_filtered.sample(50, random_state=42)
        else:
            df_sample = df_filtered
            
        inputs = df_sample[src_col].tolist()
        references = df_sample[tgt_col].tolist()
        
        predictions = []
        batch_size = 8
        
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            
            # Tokenize inputs
            encoded = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                # Force target language code start token
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
                outputs = model.generate(
                    **encoded, 
                    max_length=128, 
                    forced_bos_token_id=forced_bos_token_id
                )
                
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
            
        # Format references for SacreBLEU (list of lists)
        refs_formatted = [[ref] for ref in references]
        
        # Compute scores
        bleu_results = sacrebleu.compute(predictions=predictions, references=refs_formatted)
        chrf_results = chrf.compute(predictions=predictions, references=refs_formatted)
        
        print(f"Processed {len(inputs)} sentences.")
        print(f"BLEU: {bleu_results['score']:.2f} | chrF: {chrf_results['score']:.2f}")
        
        results.append({
            'Direction': f"{src_col} -> {tgt_col}",
            'Samples': len(inputs),
            'BLEU': bleu_results['score'],
            'chrF': chrf_results['score']
        })
        
        # Show sample outputs
        print("\n  Sample Translations:")
        for idx in range(min(2, len(inputs))):
            print(f"  Input:  {inputs[idx]}")
            print(f"  Target: {references[idx]}")
            print(f"  Pred:   {predictions[idx]}")
            print("  " + "-" * 20)
            
    # Print overall summary table
    print("\n" + "=" * 45)
    print("           OVERALL EVALUATION SUMMARY")
    print("=" * 45)
    print(f"{'Direction':<22} | {'Samples':<7} | {'BLEU':<6} | {'chrF':<6}")
    print("-" * 45)
    for r in results:
        print(f"{r['Direction']:<22} | {r['Samples']:<7} | {r['BLEU']:<6.2f} | {r['chrF']:<6.2f}")
    print("=" * 45)

if __name__ == "__main__":
    main()
