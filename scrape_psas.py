import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    sentences_list = []
    for retries in range(2):
        try:
            art_resp = requests.get(url, headers=HEADERS, timeout=12)
            if art_resp.status_code != 200:
                continue
            art_soup = BeautifulSoup(art_resp.text, 'html.parser')
            
            body_div = art_soup.find('div', class_=lambda c: c and ('rw-report__content' in c or 'text-formatted' in c or 'report-body' in c)) or art_soup.find('main')
            if not body_div:
                return sentences_list
                
            paragraphs = body_div.find_all('p')
            for p in paragraphs:
                clean_p = re.sub(r'\s+', ' ', p.get_text(strip=True)).strip()
                sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_p)
                for sent in sents:
                    s_clean = re.sub(r'\s+', ' ', sent).strip()
                    if is_single_genuine_sentence(s_clean):
                        sentences_list.append({
                            "Domain": label_domain(s_clean),
                            "English": s_clean
                        })
            break
        except Exception:
            time.sleep(1)
    return sentences_list

def scrape_reliefweb_psas(num_pages=6, max_workers=6):
    """
    Scrapes single, individual sentence-level PSA rows from ReliefWeb.
    """
    base_url = "https://reliefweb.int/updates"
    report_urls = []

    print(f"Fetching report links across {num_pages} pages from ReliefWeb...")
    
    for page in range(num_pages):
        params = {"primary_country": 131, "page": page}
        try:
            response = requests.get(base_url, params=params, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('article')
            for article in articles:
                title_tag = article.find('h3')
                if title_tag and title_tag.find('a'):
                    link = title_tag.find('a')['href']
                    if link.startswith('/'):
                        link = 'https://reliefweb.int' + link
                    report_urls.append(link)
        except Exception as e:
            print(f"Notice on listing page {page}: {e}")

    report_urls = list(set(report_urls))
    print(f"Found {len(report_urls)} report links. Fetching sentence-level rows...")

    psas = []
    seen = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_report_sentences, url) for url in report_urls]
        for future in as_completed(futures):
            res_list = future.result()
            for item in res_list:
                key = item["English"].lower()
                if key not in seen:
                    seen.add(key)
                    psas.append(item)

    return pd.DataFrame(psas)

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    df_scraped = scrape_reliefweb_psas(num_pages=6, max_workers=6)
    
    if df_scraped.empty:
        print("Warning: No sentences were retrieved.")
    else:
        df_scraped = df_scraped.drop_duplicates(subset=["English"])
        print(f"Total unique English sentence rows scraped: {len(df_scraped)}")
        
        output_path = os.path.join(output_dir, "scraped_psas_english.csv")
        df_scraped.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved English sentence rows to {output_path}")

if __name__ == "__main__":
    main()
