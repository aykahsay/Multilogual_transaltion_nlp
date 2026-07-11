import pandas as pd
from deep_translator import GoogleTranslator
import time
import os
from tqdm import tqdm

def translate_to_kiswahili(text):
    try:
        # Google Translate language code for Swahili is 'sw'
        translated = GoogleTranslator(source='en', target='sw').translate(text)
        return translated
    except Exception as e:
        print(f"Translation error for '{text[:30]}...': {e}")
        return None

def main():
    input_path = os.path.join("data", "scraped_psas_english.csv")
    output_path = os.path.join("data", "scraped_psas_translated.csv")
    
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found. Run scrape_psas.py first.")
        return
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # We don't want to translate everything if it takes too long.
    # Let's target translating exactly what we need to hit 5000 combined.
    # The original CSV has 3153 valid rows. We need about 1850 more.
    target_new_rows = 1900
    
    if len(df) > target_new_rows:
        df = df.head(target_new_rows).copy()
        
    print(f"Translating {len(df)} records to Kiswahili. This will take a while...")
    
    translations = []
    # Using simple iteration with sleep to avoid getting blocked
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sw_text = translate_to_kiswahili(row['English'])
        translations.append(sw_text)
        # Small delay
        if idx % 10 == 0:
            time.sleep(0.5)
            
    df['Kiswahili'] = translations
    
    # Drop failed translations
    df = df.dropna(subset=['Kiswahili'])
    
    print(f"Successfully translated {len(df)} records.")
    
    df.to_csv(output_path, index=False)
    print(f"Saved translated data to {output_path}")

if __name__ == "__main__":
    main()
