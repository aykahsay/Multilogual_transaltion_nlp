import os
import glob
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def main():
    pdf_dir = os.path.join("data", "kenya_bulk_notices_2026")
    output_csv = os.path.join("data", "treasury_psas.csv")
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process...")
    
    extracted_data = []
    
    for pdf_path in tqdm(pdf_files):
        text = extract_text_from_pdf(pdf_path)
        
        # Basic filtering to remove extremely short/empty documents
        if len(text) > 100:
            extracted_data.append({
                "Filename": os.path.basename(pdf_path),
                "English": text,
                "Source": "Kenya National Treasury"
            })
            
    df = pd.DataFrame(extracted_data)
    df.to_csv(output_csv, index=False)
    print(f"Successfully extracted {len(df)} PSA documents to {output_csv}")

if __name__ == "__main__":
    main()
