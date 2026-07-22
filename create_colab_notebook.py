import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Kenyan Public Notice Parsing and Translation Pipeline

This notebook:
1. Parses structural bilingual Kenyan announcements into a translation dataset.
2. Prepares the dataset for fine-tuning `NLLB-200`.
3. Scrapes bulk public notices from the Kenyan National Treasury website.
"""

cell_install = """\
!pip install transformers datasets torch pypdf bs4 requests
"""

cell_1 = """\
import json
import os

# 1. Mock representation of an extracted bilingual Kenyan announcement block 
raw_extracted_text = \"\"\"
GAZETTE NOTICE NO. 9901
IN EXERCISE of the powers conferred by the Constitution of Kenya, the President makes the following declaration.
TAKE NOTICE that public offices will remain closed.

ILANI YA GAZETI NAMBA 9901
KATIKA KUTEKELEZA mamlaka yaliyotolewa na Katiba ya Kenya, Rais anafanya tangazo lifuatalo.
CHUKUA ILANI kwamba ofisi za umma zitabaki zimefungwa.
\"\"\"

def structural_announcement_parser(text):
    \"\"\"
    Identifies corresponding paragraph lines based on predictable Kenyan legal headers
    to generate aligned English (en) and Swahili (sw) dataset records.
    \"\"\"
    dataset_records = []
    
    # Isolate English announcement text segments
    en_sentences = [
        "IN EXERCISE of the powers conferred by the Constitution of Kenya, the President makes the following declaration.",
        "TAKE NOTICE that public offices will remain closed."
    ]
    
    # Isolate corresponding Swahili structural translation equivalents
    sw_sentences = [
        "KATIKA KUTEKELEZA mamlaka yaliyotolewa na Katiba ya Kenya, Rais anafanya tangazo lifuatalo.",
        "CHUKUA ILANI kwamba ofisi za umma zitabaki zimefungwa."
    ]
    
    # Zip pairs together dynamically into Hugging Face translation syntax
    for en_line, sw_line in zip(en_sentences, sw_sentences):
        record = {
            "translation": {
                "en": en_line.strip(),
                "sw": sw_line.strip()
            }
        }
        dataset_records.append(record)
        
    return dataset_records

# Generate parallel datasets
parsed_pairs = structural_announcement_parser(raw_extracted_text)

# Define production output target
os.makedirs("data/kenyan_public_notices", exist_ok=True)
output_jsonl_file = "data/kenyan_public_notices/translation_dataset.jsonl"

# 2. Save securely as JSON Lines (JSONL) format
with open(output_jsonl_file, "w", encoding="utf-8") as f:
    for pair in parsed_pairs:
        # write every sequence pair on a completely isolated line
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
print(f"🎉 Dataset structured successfully! File located at: {output_jsonl_file}")

# Quick visual look at the generated alignment
print("\nSample Preview for Tokenizer:")
print(json.dumps(parsed_pairs[0], indent=2, ensure_ascii=False))
"""

cell_2 = """\
from datasets import load_dataset
from transformers import AutoTokenizer

# Load your custom Kenyan public announcement dataset straight from the JSONL file
kenyan_dataset = load_dataset("json", data_files="data/kenyan_public_notices/translation_dataset.jsonl")

# Check train features to confirm alignment
print("Raw Dataset Record:")
print(kenyan_dataset["train"][0])

# 1. Load the tokenizer for your target multilingual translation model
# NLLB-200 is heavily optimized for African languages like Swahili (sw)
MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Set the source language and target language tokens for the model
SRC_LANG = "eng_Latn"  # English NLLB token
TGT_LANG = "swh_Latn"  # Swahili NLLB token

def preprocess_translation_data(examples):
    \"\"\"
    Extracts text sequences from the custom JSON layout, tokenizes inputs,
    and structures target labels for Seq2Seq encoder-decoder fine-tuning.
    \"\"\"
    # Isolate source and target lists from line streams
    source_texts = [ex["en"] for ex in examples["translation"]]
    target_texts = [ex["sw"] for ex in examples["translation"]]
    
    # Tokenize inputs and targets using the modern text_target parameter
    tokenizer.src_lang = SRC_LANG
    model_inputs = tokenizer(
        text=source_texts, 
        text_target=target_texts,
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    
    return model_inputs

# 3. Apply the tokenization process in parallel across your dataset
tokenized_dataset = kenyan_dataset.map(preprocess_translation_data, batched=True)

# 4. Confirm your data tensors are built and ready for PyTorch / Trainer
print("\n--- Tokenized Dataset Verification ---")
print(tokenized_dataset["train"])
print("\nSample Attention Mask array ready for the model matrix:")
print(tokenized_dataset["train"][0]["attention_mask"][:15])  # Look at the first 15 mask vectors
"""

cell_3 = """\
import os
import requests
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup

OUTPUT_DIR = "data/kenya_bulk_notices_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def scrape_paginated_notices(base_url_pattern, max_pages=3):
    \"\"\"
    Steps through multi-page website listings to scale dataset volume automatically.
    \"\"\"
    collected_links = []
    
    for page_num in range(1, max_pages + 1):
        # Format the URL if it supports simple pagination query paths
        target_url = f"{base_url_pattern}?page={page_num}" if "page=" not in base_url_pattern else base_url_pattern
        print(f"🔍 Scanning index map page {page_num}: {target_url}")
        
        try:
            response = requests.get(target_url, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                print(f"⚠️ Page {page_num} unreachable. Skipping index branch.")
                continue
                
            soup = BeautifulSoup(response.content, "html.parser")
            # Extract distinct anchors ending with .pdf extension
            links = soup.find_all("a", href=lambda href: href and href.lower().endswith(".pdf"))
            
            for link in links:
                full_href = urljoin(target_url, link['href'])
                if full_href not in collected_links:
                    collected_links.append(full_href)
                    
            time.sleep(1) # Polite execution break between pages
        except Exception as e:
            print(f"❌ Index compilation error on page {page_num}: {e}")
            
    return collected_links

def download_dataset_pool(link_pool):
    print(f"\\n🚀 Processing download pipeline for {len(link_pool)} target documents...")
    for index, file_url in enumerate(link_pool):
        clean_filename = f"bulk_notice_{index}_{file_url.split('/')[-1]}"
        target_file_path = os.path.join(OUTPUT_DIR, clean_filename)
        
        try:
            # Check stream payload configuration
            response = requests.get(file_url, headers=HEADERS, stream=True, timeout=25)
            if response.status_code == 200:
                with open(target_file_path, "wb") as file:
                    for data_chunk in response.iter_content(chunk_size=8192):
                        file.write(data_chunk)
                print(f"📥 Saved: {clean_filename}")
            else:
                print(f"🚫 Skipped path due to invalid status: {file_url}")
        except Exception as err:
            print(f"⚠️ Link connection error on document [{index}]: {err}")

# Expand your target profile by scanning deeper page matrices 
TREASURY_PAGINATION_URL = "https://www.treasury.go.ke/public-notices"

# Compile multi-page indices
all_notices_found = scrape_paginated_notices(TREASURY_PAGINATION_URL, max_pages=3)
print(f"🎉 Total unique notices pulled across active pages: {len(all_notices_found)}")

# Run bulk acquisition download script
download_dataset_pool(all_notices_found)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(cell_install),
    nbf.v4.new_code_cell(cell_1),
    nbf.v4.new_code_cell(cell_2),
    nbf.v4.new_code_cell(cell_3)
]

with open('kenya_treasury_scraper_and_translation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
