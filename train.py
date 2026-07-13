import pandas as pd
import os
import torch
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset

def load_data(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

def preprocess_function(examples, tokenizer, max_length=128):
    inputs = examples["English"]
    targets = examples["Kiswahili"]
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, padding="max_length", truncation=True)
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    return model_inputs

def main():
    model_checkpoint = "Helsinki-NLP/opus-mt-en-sw"
    output_dir = "models/psa-en-sw-finetuned"
    
    print(f"Loading tokenizer and model: {model_checkpoint}")
    tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    # Load dataset splits
    print("Loading datasets...")
    train_dataset = load_data("data/train.csv")
    dev_dataset = load_data("data/dev.csv")
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    tokenized_dev = dev_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=dev_dataset.column_names)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3, # 3 epochs for a simple POC
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        push_to_hub=False,
    )
    
    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
