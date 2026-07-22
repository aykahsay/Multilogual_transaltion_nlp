import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datasets import Dataset

def main():
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    output_dir = "models/nllb-en-guz"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading pretrained tokenizer & model from {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang="eng_Latn", tgt_lang="swh_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Load datasets
    train_df = pd.read_csv(os.path.join("data", "train_guz.csv"))
    dev_df = pd.read_csv(os.path.join("data", "dev_guz.csv"))

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    max_input_length = 128
    max_target_length = 128

    def preprocess_function(examples):
        inputs = [ex for ex in examples["English"]]
        targets = [ex for ex in examples["Ekegusii"]]
        
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        
        # Tokenize targets with target language setting
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_dev = dev_dataset.map(preprocess_function, batched=True, remove_columns=dev_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting NLLB-200 Ekegusii Fine-Tuning...")
    trainer.train()

    print(f"Saving fine-tuned Ekegusii model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
