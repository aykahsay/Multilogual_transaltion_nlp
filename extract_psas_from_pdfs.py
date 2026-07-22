"""
PSA Extraction from PDFs — Improved Pipeline
=============================================
Extracts genuine PSA-style sentences from downloaded Kenya government PDF notices
using a two-layer filter (linguistic + ML), then translates to Kiswahili & Somali,
and merges the result directly into the combined dataset via preprocess.py.

PDFs searched:
  - data/*.pdf
  - data/kenya_bulk_notices_2026/*.pdf

Output: data/extracted_pdf_psas.csv  (6-column format matching train/dev/test.csv)
"""

import os
import re
import time
import pickle
import pandas as pd
from pypdf import PdfReader
from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Reuse the linguistic PSA filter from data_collection_pipeline
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_collection_pipeline import (
    is_psa_sentence,
    is_news_headline,
    translate_text,
    label_domain,
    _SUPPORTED_LANGS,
)


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all raw text from a PDF file, page by page."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as exc:
        print(f"  [ERROR] Could not read {os.path.basename(pdf_path)}: {exc}")
        return ""


def split_into_sentences(text: str) -> list[str]:
    """
    Split raw PDF text into individual sentences.
    Handles common PDF artifacts: multiple spaces, line breaks, hyphenation.
    """
    if not text:
        return []

    # Remove common PDF noise
    text = re.sub(r'\s+', ' ', text)                       # collapse whitespace
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)          # fix hyphenated line-breaks
    text = re.sub(r'Page \d+ of \d+', '', text)            # remove page numbers
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)   # remove dates like 01/01/2024

    # Split on sentence-ending punctuation followed by space + capital letter
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

    # Also split on double newlines (section breaks)
    sentences = []
    for chunk in raw:
        for sub in chunk.split('\n'):
            sub = sub.strip()
            if sub:
                sentences.append(sub)

    return sentences


# ---------------------------------------------------------------------------
# ML classifier (optional second layer)
# ---------------------------------------------------------------------------

def load_classifier(
    model_path: str = "psa_classifier.pkl",
    vectorizer_path: str = "tfidf_vectorizer.pkl",
):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return classifier, vectorizer


# ---------------------------------------------------------------------------
# Translate to all supported languages
# ---------------------------------------------------------------------------

def translate_all(text: str) -> dict:
    """Translate to Kiswahili and Somali. Return None for unsupported languages."""
    result = {}
    for lang_name, lang_code in _SUPPORTED_LANGS.items():
        result[lang_name] = translate_text(text, lang_code, lang_name)
        time.sleep(0.15)
    result["Ekegusii"] = None   # Not supported by Google Translate
    result["Dholuo"]   = None   # Not supported by Google Translate
    return result


# ---------------------------------------------------------------------------
# Collect all PDFs from configured directories
# ---------------------------------------------------------------------------

def find_all_pdfs(search_dirs: list[str]) -> list[str]:
    pdf_files = []
    for directory in search_dirs:
        if not os.path.exists(directory):
            print(f"  Directory not found, skipping: {directory}")
            continue
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(directory, fname))
    return pdf_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path = os.path.join("data", "extracted_pdf_psas.csv")

    # Directories to search for PDFs
    pdf_dirs = [
        "data",
        os.path.join("data", "kenya_bulk_notices_2026"),
        os.path.join("data", "kenyan_public_notices"),
    ]

    # ------------------------------------------------------------------
    # Step 1: Find all PDFs
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Discovering PDF files")
    print("=" * 60)
    pdf_files = find_all_pdfs(pdf_dirs)
    print(f"Found {len(pdf_files)} PDF files across all directories.")

    if not pdf_files:
        print("No PDFs found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 2: Load optional ML classifier
    # ------------------------------------------------------------------
    classifier, vectorizer = load_classifier()
    if classifier:
        print("ML classifier loaded — will apply as second filter layer.")
    else:
        print("ML classifier not found — using linguistic filter only.")

    # ------------------------------------------------------------------
    # Step 3: Extract + filter PSA sentences
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Extracting and filtering PSA sentences from PDFs")
    print("=" * 60)

    all_psa_sentences = []

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        print(f"\n  Processing: {fname}")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print("    → Empty or unreadable PDF. Skipping.")
            continue

        sentences = split_into_sentences(raw_text)
        print(f"    → Extracted {len(sentences)} raw sentences")

        # Layer 1: Linguistic PSA filter
        ling_passed = [s for s in sentences if is_psa_sentence(s)]
        print(f"    → Linguistic filter: {len(ling_passed)} passed")

        # Layer 2: ML classifier (if available)
        if classifier and ling_passed:
            features = vectorizer.transform(ling_passed)
            probs = classifier.predict_proba(features)[:, 1]
            ml_passed = [s for s, p in zip(ling_passed, probs) if p >= 0.45]
            print(f"    → ML classifier:   {len(ml_passed)} passed")
        else:
            ml_passed = ling_passed

        # Deduplicate within this PDF
        seen = set()
        unique = []
        for s in ml_passed:
            key = s.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(s.strip())

        print(f"    → Unique PSA sentences kept: {len(unique)}")
        all_psa_sentences.extend(unique)

    # Global deduplication
    seen_global = set()
    final_sentences = []
    for s in all_psa_sentences:
        key = s.strip().lower()
        if key not in seen_global:
            seen_global.add(key)
            final_sentences.append(s.strip())

    print(f"\n{'=' * 60}")
    print(f"Total unique PSA sentences from all PDFs: {len(final_sentences)}")
    print("=" * 60)

    if not final_sentences:
        print("No PSA sentences extracted. Check PDF content or filter thresholds.")
        return

    # ------------------------------------------------------------------
    # Step 4: Translate to all languages
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3: Translating {len(final_sentences)} sentences")
    print("  Kiswahili (sw) + Somali (so) via Google Translate")
    print("  Ekegusii + Dholuo → None (not supported)")
    print("=" * 60)

    records = []
    for idx, text in enumerate(final_sentences):
        langs = translate_all(text)

        # Only keep if Kiswahili translation succeeded
        if langs.get("Kiswahili"):
            records.append({
                "English":   text,
                "Kiswahili": langs["Kiswahili"],
                "Ekegusii":  langs["Ekegusii"],
                "Dholuo":    langs["Dholuo"],
                "Somali":    langs["Somali"],
                "Domain":    label_domain(text),
            })

        if (idx + 1) % 25 == 0:
            sw_ok = sum(1 for r in records if r["Kiswahili"])
            so_ok = sum(1 for r in records if r["Somali"])
            print(f"  [{idx + 1}/{len(final_sentences)}] "
                  f"Kiswahili: {sw_ok}, Somali: {so_ok}")

    print(f"\nTranslated successfully: {len(records)} sentences")
    so_count = sum(1 for r in records if r["Somali"])
    print(f"  Kiswahili: {len(records)}/{len(records)}")
    print(f"  Somali   : {so_count}/{len(records)}")

    # ------------------------------------------------------------------
    # Step 5: Save to CSV (6-column format matching dataset)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Saving output")
    print("=" * 60)

    OUTPUT_COLS = ["English", "Kiswahili", "Ekegusii", "Dholuo", "Somali", "Domain"]
    df = pd.DataFrame(records, columns=OUTPUT_COLS)

    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} PSA rows to '{output_path}'")
    print(f"Columns: {list(df.columns)}")

    # Domain breakdown
    print("\nDomain distribution:")
    print(df["Domain"].value_counts().to_string())

    # Sample preview
    print("\n--- Sample extracted PSA sentences ---")
    for _, row in df.head(8).iterrows():
        print(f"  [{row['Domain']}] {row['English'][:80]}")
        print(f"    SW: {str(row['Kiswahili'])[:80]}")
        print()

    # ------------------------------------------------------------------
    # Step 6: Re-run preprocess.py to merge into train/dev/test
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Merging into dataset via preprocess.py")
    print("=" * 60)
    try:
        import preprocess
        preprocess.main()

        train = pd.read_csv("data/train.csv")
        dev   = pd.read_csv("data/dev.csv")
        test  = pd.read_csv("data/test.csv")
        total = len(train) + len(dev) + len(test)
        print(f"\nFinal dataset size: {total:,} rows "
              f"(train={len(train)}, dev={len(dev)}, test={len(test)})")
        if total >= 5000:
            print("SUCCESS: Dataset exceeds 5,000 rows.")
        else:
            print(f"WARNING: Dataset is {total} rows (target: ≥5,000).")
    except Exception as exc:
        print(f"Preprocessing error: {exc}")
        print("Tip: Run 'python preprocess.py' manually to merge the PDF data.")


if __name__ == "__main__":
    main()
