import pandas as pd
from deep_translator import GoogleTranslator
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def translate_sentence(text):
    if not text or not isinstance(text, str):
        return None
    for attempt in range(3):
        try:
            translated = GoogleTranslator(source='en', target='sw').translate(text.strip())
            return translated
        except Exception:
            time.sleep(0.3)
    return None

def main():
    input_path = os.path.join("data", "scraped_psas_english.csv")
    output_path = os.path.join("data", "scraped_psas_translated.csv")
    
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found. Run scrape_psas.py first.")
        return
        
    print(f"Loading sentence rows from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} sentence rows for translation.")
    
    translator = GoogleTranslator(source='en', target='sw')
    
    translated_records = []
    
    def process_row(row):
        en = str(row['English']).strip()
        dom = str(row.get('Domain', 'Disaster/Health'))
        sw = translate_sentence(en)
        if sw:
            return {"English": en, "Kiswahili": sw, "Domain": dom}
        return None

    print("Translating English sentences to Kiswahili concurrently...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                translated_records.append(res)
                
    df_translated = pd.DataFrame(translated_records, columns=["English", "Kiswahili", "Domain"])
    df_translated = df_translated.dropna(subset=["Kiswahili"])
    df_translated = df_translated.drop_duplicates(subset=["English"])
    
    print(f"Successfully translated {len(df_translated)} sentence-level PSA rows.")
    
    df_translated.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved translated sentence-level data to {output_path}")

if __name__ == "__main__":
    main()
