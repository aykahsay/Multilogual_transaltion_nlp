import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import time
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

DOMAIN_KEYWORDS = {
    "Health": ["health", "disease", "vaccine", "outbreak", "cholera", "hospital", "clinic", "medical", "hiv", "malaria", "tb", "maternal", "infant", "nutrition", "sanitation", "hygiene"],
    "Disaster/Health": ["drought", "flood", "famine", "hunger", "emergency", "disaster", "relief", "evacuation", "rainfall", "climate", "rains", "water supply", "food security"],
    "Security": ["police", "crime", "safety", "violence", "security", "traffic", "gbv", "shelter", "fire", "law enforcement"],
    "Agriculture": ["crop", "livestock", "farmer", "farm", "seed", "fertilizer", "harvest", "maize", "pest", "veterinary"],
    "Education": ["school", "student", "pupil", "teacher", "exam", "learning", "education", "curriculum", "bursary"],
    "Governance": ["government", "county", "policy", "election", "public notice", "citizen", "parliament", "tax"]
}

HEADLINE_REJECT_PATTERNS = [
    r"^\s*\w[\w\s]+\s+County\s*:\s*Drought",
    r"^\s*\w[\w\s]+\s+County\s*:\s*Flood",
    r"Bulletin\s+for\s+(January|February|March|April|May|June|July|August|September|October|November|December)",
    r"\b(Q1|Q2|Q3|Q4)\s+\d{4}\b",
    r"\bKEY\s+MESSAGE\s+UPDATE\b",
    r"\bCOUNTRY\s+BRIEF\b",
    r"\bSITUATION\s+REPORT\b",
    r"\bOPERATIONAL\s+UPDATE\b",
    r"\bPOLICY\s+BRIEF\b",
    r"\bINFOGRAPHICS?\b",
    r"\bNEWSLETTER\b",
    r"^\s*\w[\w\s,\-]+\s+:\s+\w",
    r"\bVolume\s+\d+",
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    r"^\s*GIEWS\b", r"^\s*DREF\b", r"^\s*WFP\b", r"^\s*UNHCR\b", r"^\s*OCHA\b",
    r"\(IPC Phase \d+\)", r"\bReliefWeb\b", r"\bas of\s+\w+\s+\d{4}\b", r"\[EN/SW\]"
]

def label_domain(text):
    text_lower = text.lower()
    scores = {d: sum(1 for kw in kws if kw in text_lower) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Disaster/Health"

def is_single_genuine_sentence(text):
    if not text or not isinstance(text, str):
        return False
    words = text.split()
    if len(words) < 7 or len(words) > 45:
        return False
    if not re.match(r'^[A-Z]', text) or not text.strip().endswith(('.', '!', '?')):
        return False
    lower = text.lower()
    for pat in HEADLINE_REJECT_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False
    if any(b in lower for b in ["cookie", "privacy policy", "download report", "photo:", "source:", "copyright", "all rights reserved", "published on", "terms of service", "subscribe", "http", "www."]):
        return False
    return True

def fetch_report_sentences(url):
    sents = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            body = soup.find('div', class_=lambda c: c and ('rw-report__content' in c or 'text-formatted' in c or 'report-body' in c)) or soup.find('main')
            if body:
                for p in body.find_all('p'):
                    txt = re.sub(r'\s+', ' ', p.get_text(strip=True)).strip()
                    for sent in re.split(r'(?<=[.!?])\s+(?=[A-Z])', txt):
                        s_clean = re.sub(r'\s+', ' ', sent).strip()
                        if is_single_genuine_sentence(s_clean):
                            sents.append(s_clean)
    except Exception:
        pass
    return sents

def main():
    print("Scraping report URLs from ReliefWeb Kenya...")
    base_url = "https://reliefweb.int/updates"
    report_urls = []
    
    for page in range(3):
        try:
            res = requests.get(base_url, params={"primary_country": 131, "page": page}, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                for art in soup.find_all('article'):
                    h3 = art.find('h3')
                    if h3 and h3.find('a'):
                        link = h3.find('a')['href']
                        if link.startswith('/'):
                            link = 'https://reliefweb.int' + link
                        report_urls.append(link)
        except Exception as e:
            print(f"Listing page error: {e}")

    report_urls = list(set(report_urls))
    print(f"Collected {len(report_urls)} unique report links. Extracting sentence-level rows...")

    sentences = []
    seen = set()
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res_list in executor.map(fetch_report_sentences, report_urls):
            for s in res_list:
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    sentences.append(s)

    print(f"Extracted {len(sentences)} unique single-sentence rows.")

    # Translate to Kiswahili
    print("Translating sentences to Kiswahili line by line...")
    translator = GoogleTranslator(source='en', target='sw')
    
    def translate_pair(sent):
        try:
            sw = translator.translate(sent)
            if sw and len(sw.strip()) > 3:
                return {
                    "English": sent,
                    "Kiswahili": sw.strip(),
                    "Domain": label_domain(sent)
                }
        except Exception:
            pass
        return None

    records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for item in executor.map(translate_pair, sentences):
            if item:
                records.append(item)

    df_out = pd.DataFrame(records, columns=["English", "Kiswahili", "Domain"])
    df_out = df_out.drop_duplicates(subset=["English"])

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Save to scraped_psas_translated.csv
    translated_path = os.path.join(output_dir, "scraped_psas_translated.csv")
    df_out.to_csv(translated_path, index=False, encoding="utf-8")
    print(f"Saved translated sentence rows to {translated_path}")

    # Save English-only version to scraped_psas_english.csv
    df_english = df_out[["Domain", "English"]].copy()
    english_path = os.path.join(output_dir, "scraped_psas_english.csv")
    df_english.to_csv(english_path, index=False, encoding="utf-8")
    print(f"Saved English sentence rows to {english_path}")

if __name__ == "__main__":
    main()
