"""
Extract Data from PDFs to CSV (English, Kiswahili, Domain)
=========================================================
This script reads PDF files, extracts meaningful sentences/notices,
classifies them into domain categories, translates English to Kiswahili,
and outputs a clean CSV file with columns: English, Kiswahili, Domain.

Usage:
    python extract_pdf_to_csv.py --input_dir data --output_csv data/extracted_pdf_data.csv --max_per_pdf 3 --max_pages 15
"""

import os
import re
import sys
import argparse
import time
import pandas as pd
from pypdf import PdfReader
from deep_translator import GoogleTranslator


# ---------------------------------------------------------------------------
# Domain Classification Taxonomy & Rule Engine
# ---------------------------------------------------------------------------
_DOMAIN_MAP = {
    "Disaster/Health": [
        "drought", "flood", "famine", "hunger", "early warning", "bulletin",
        "emergency", "outbreak", "epidemic", "pandemic", "disaster", "relief",
        "humanitarian", "evacuation", "sanitation", "hygiene", "cholera",
        "malnutrition", "food security", "water supply", "rain", "rainfall",
        "climate change", "weather", "ndma", "crisis"
    ],
    "Health": [
        "health", "disease", "vaccine", "vaccination", "hospital", "clinic",
        "medical", "nurse", "doctor", "hiv", "malaria", "tb", "mpox",
        "maternal", "infant", "newborn", "pregnancy", "breastfeeding",
        "nutrition", "mental health", "medicine", "treatment", "patient"
    ],
    "Education": [
        "school", "student", "pupil", "teacher", "exam", "kcse", "kcpe",
        "university", "college", "tvet", "enroll", "curriculum", "tuition",
        "bursary", "scholarship", "textbook", "learning", "literacy",
        "grade 9", "junior secondary", "cbc", "classroom", "education"
    ],
    "Security": [
        "police", "crime", "theft", "fraud", "scam", "trafficking",
        "violence", "gbv", "safety", "shelter", "fire", "road safety",
        "traffic", "law enforcement", "equal rights", "legal reform",
        "equality", "rights", "court", "justice", "security"
    ],
    "Governance": [
        "vote", "election", "iebc", "government", "corruption", "eacc",
        "tax", "kra", "law", "county", "parliament", "citizen", "public notice",
        "registration", "id card", "birth certificate", "national treasury",
        "policy", "reform", "budget", "finance", "public finance", "gazette"
    ],
    "Agriculture": [
        "farm", "crop", "livestock", "seed", "fertilizer", "pest",
        "irrigation", "harvest", "maize", "animal", "veterinary",
        "arid", "pastoral", "farmers", "agribusiness", "produce"
    ],
}


def label_domain(text: str) -> str:
    """Classify input text into one of the target domains based on keyword frequency."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in _DOMAIN_MAP.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best_domain = max(scores, key=scores.get)
    # Default to Disaster/Health if no specific keyword scored
    return best_domain if scores[best_domain] > 0 else "Disaster/Health"


# ---------------------------------------------------------------------------
# PDF Extraction & Sentence Splitting
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str, max_pages: int = 15) -> str:
    """Extract raw text content from up to `max_pages` of a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        num_pages = min(len(reader.pages), max_pages)
        for i in range(num_pages):
            text = reader.pages[i].extract_text()
            if text:
                pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as exc:
        print(f"  [Error reading PDF {os.path.basename(pdf_path)}]: {exc}", flush=True)
        return ""


def clean_and_split_sentences(text: str) -> list[str]:
    """Clean PDF text artifacts and split into candidate sentences/notices."""
    if not text:
        return []

    # Clean whitespace and PDF line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)

    # Split into raw sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

    cleaned = []
    for s in raw_sentences:
        s = s.strip()
        # Keep sentences with reasonable length (e.g., 6 to 50 words)
        words = s.split()
        if 6 <= len(words) <= 50:
            cleaned.append(s)

    return cleaned


# ---------------------------------------------------------------------------
# Translation (English -> Kiswahili)
# ---------------------------------------------------------------------------

def translate_to_kiswahili(translator: GoogleTranslator, text: str) -> str:
    """Translate an English text string to Kiswahili using GoogleTranslator."""
    try:
        translation = translator.translate(text)
        return translation if translation else ""
    except Exception as e:
        print(f"  [Translation Warning]: {e}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process_pdfs(pdf_paths: list[str], max_sentences_per_pdf: int = 3, max_pages: int = 15) -> pd.DataFrame:
    """Extract, filter, label, and translate sentences from PDF files."""
    translator = GoogleTranslator(source='en', target='sw')
    extracted_records = []
    seen_texts = set()

    for idx, pdf_path in enumerate(pdf_paths):
        fname = os.path.basename(pdf_path)
        print(f"[{idx+1}/{len(pdf_paths)}] Processing PDF: {fname}", flush=True)
        raw_text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
        if not raw_text:
            continue

        sentences = clean_and_split_sentences(raw_text)

        count = 0
        for text in sentences:
            if count >= max_sentences_per_pdf:
                break

            # Avoid duplicates
            text_key = text.lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            domain = label_domain(text)
            kiswahili = translate_to_kiswahili(translator, text)

            if kiswahili:
                extracted_records.append({
                    "English": text,
                    "Kiswahili": kiswahili,
                    "Domain": domain
                })
                count += 1
                time.sleep(0.05)  # Slight delay to respect rate limits

    df = pd.DataFrame(extracted_records, columns=["English", "Kiswahili", "Domain"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract data from PDFs into English, Kiswahili, Domain CSV.")
    parser.add_argument("--input_dir", type=str, default="data", help="Directory containing PDF files")
    parser.add_argument("--output_csv", type=str, default="data/extracted_pdf_data.csv", help="Path to save output CSV")
    parser.add_argument("--max_per_pdf", type=int, default=3, help="Max sentences to extract per PDF file")
    parser.add_argument("--max_pages", type=int, default=15, help="Max pages to inspect per PDF file")
    args = parser.parse_args()

    # Collect PDF files from input_dir and optional subdirectories
    pdf_files = []
    if os.path.exists(args.input_dir):
        for root, _, files in os.walk(args.input_dir):
            for f in sorted(files):
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))

    print(f"Found {len(pdf_files)} PDF files in directory '{args.input_dir}'.", flush=True)
    if not pdf_files:
        print("No PDF files found to process.", flush=True)
        return

    df = process_pdfs(pdf_files, max_sentences_per_pdf=args.max_per_pdf, max_pages=args.max_pages)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Successfully extracted {len(df)} records into '{args.output_csv}'.", flush=True)
    print("=" * 60, flush=True)
    print("\nPreview of Extracted Data:", flush=True)
    print(df.head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
