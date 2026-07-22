"""
Extract Real Web PSA Sentences (Full Advisory Sentences, No Headlines)
======================================================================
1. Fetches full article bodies from public REST APIs (ReliefWeb Kenya)
2. Filters out headlines/titles using strict linguistic filters (imperative verbs + advisory phrases + length checks)
3. Translates to Kiswahili
4. Assigns Domain
5. Saves clean real PSA sentences to data/real_web_psas.csv
"""

import os
import re
import time
import requests
import pandas as pd
from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Filters & Keywords
# ---------------------------------------------------------------------------

IMPERATIVE_VERBS = [
    "wash", "use", "wear", "stay", "avoid", "report", "seek", "call",
    "contact", "register", "enroll", "attend", "follow", "ensure",
    "protect", "vaccinate", "isolate", "test", "verify", "check",
    "keep", "carry", "do not", "don't", "never", "always", "make sure",
    "remember", "be aware", "beware", "be cautious", "be vigilant",
    "be careful", "be alert", "be ready", "be safe", "please", "kindly",
    "remind", "urge", "encourage", "advise", "warn", "alert",
    "evacuate", "relocate", "suspend", "disinfect", "sanitize", "boil",
    "treat", "prevent", "limit", "reduce", "increase", "maintain", "monitor",
    "observe", "comply", "submit", "apply", "download", "visit", "access"
]

ADVISORY_PHRASES = [
    "are urged", "are advised", "are encouraged", "are reminded",
    "are required", "are requested", "is urged", "is advised",
    "is encouraged", "is reminded", "is required", "is requested",
    "members of the public", "citizens are", "residents are",
    "the public is", "all persons", "all citizens", "all residents",
    "everyone is", "people are", "individuals are", "parents are",
    "farmers are", "drivers are", "travelers are", "expectant mothers",
    "caregivers are", "health workers are", "students are", "schools must",
    "counties must", "hospitals must", "it is mandatory", "it is important",
    "it is critical", "it is vital", "public advisory", "health advisory",
    "safety advisory", "emergency notice"
]

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

_DOMAIN_MAP = {
    "Disaster/Health": [
        "drought", "flood", "famine", "hunger", "early warning",
        "emergency", "outbreak", "epidemic", "pandemic", "disaster", "relief",
        "evacuation", "sanitation", "hygiene", "cholera", "malnutrition",
        "food security", "water supply", "rain", "rainfall", "climate", "weather"
    ],
    "Health": [
        "health", "disease", "vaccine", "vaccination", "hospital", "clinic",
        "medical", "nurse", "doctor", "hiv", "malaria", "tb", "mpox",
        "maternal", "infant", "newborn", "pregnancy", "breastfeed",
        "nutrition", "mental health", "medicine", "treatment", "patient"
    ],
    "Education": [
        "school", "student", "pupil", "teacher", "exam", "kcse", "kcpe",
        "university", "college", "tvet", "enroll", "curriculum", "tuition",
        "bursary", "scholarship", "textbook", "learning", "literacy", "cbc"
    ],
    "Security": [
        "police", "crime", "theft", "fraud", "scam", "trafficking",
        "violence", "gbv", "safety", "shelter", "fire", "road safety",
        "traffic", "law enforcement", "equal rights", "equality", "security"
    ],
    "Governance": [
        "vote", "election", "iebc", "government", "corruption", "tax", "kra",
        "county", "parliament", "citizen", "public notice", "id card", "policy"
    ],
    "Agriculture": [
        "farm", "crop", "livestock", "seed", "fertilizer", "pest",
        "irrigation", "harvest", "maize", "animal", "veterinary", "farmers"
    ],
}

def label_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for domain, keywords in _DOMAIN_MAP.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Disaster/Health"

def is_genuine_psa_sentence(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    
    # 1. Reject obvious headline/title patterns
    for pat in HEADLINE_REJECT_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False
            
    words = text.split()
    if len(words) < 7 or len(words) > 65:
        return False
        
    text_lower = text.lower()
    
    # 2. Must contain an imperative verb OR advisory phrase
    has_imperative = any(text_lower.startswith(v) or f" {v} " in text_lower for v in IMPERATIVE_VERBS)
    has_advisory = any(phrase in text_lower for phrase in ADVISORY_PHRASES)
    
    return has_imperative or has_advisory

# ---------------------------------------------------------------------------
# ReliefWeb Web API Fetcher
# ---------------------------------------------------------------------------

def fetch_web_psa_sentences(max_articles: int = 200) -> list[str]:
    print(f"\n[Fetching Article Bodies from Web API] Requesting up to {max_articles} articles for Kenya...")
    url = "https://api.reliefweb.int/v1/reports"
    payload = {
        "appname": "kenya-psa-nlp-collector",
        "fields": {"include": ["body"]},
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "primary_country.iso3", "value": "KEN"},
                {"field": "format.name", "value": "report"}
            ]
        },
        "sort": ["date.created:desc"],
        "limit": min(100, max_articles)
    }
    
    psa_sentences = []
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            articles = resp.json().get("data", [])
            for art in articles:
                body = art.get("fields", {}).get("body", "")
                if not body:
                    continue
                # Clean HTML / Whitespace
                body_clean = re.sub(r'<[^>]+>', ' ', body)
                body_clean = re.sub(r'\s+', ' ', body_clean).strip()
                
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', body_clean)
                for s in sentences:
                    s_clean = s.strip()
                    if is_genuine_psa_sentence(s_clean):
                        psa_sentences.append(s_clean)
    except Exception as e:
        print(f"Web API fetch error: {e}")
        
    return psa_sentences

def main():
    print("=" * 60)
    print("EXTRACTING REAL FULL PSA SENTENCES FROM THE WEB & FILTERING")
    print("=" * 60)
    
    # Step 1: Filter existing scraped file to remove all headlines/titles
    scraped_file = os.path.join("data", "scraped_psas_translated.csv")
    cleaned_records = []
    seen = set()
    
    if os.path.exists(scraped_file):
        print(f"Purging headlines/titles from {scraped_file}...")
        df_old = pd.read_csv(scraped_file)
        for _, row in df_old.iterrows():
            en = str(row.get("English", "")).strip()
            sw = str(row.get("Kiswahili", "")).strip()
            if is_genuine_psa_sentence(en):
                key = en.lower()
                if key not in seen:
                    seen.add(key)
                    cleaned_records.append({
                        "English": en,
                        "Kiswahili": sw,
                        "Domain": label_domain(en)
                    })
                    
    print(f"Retained {len(cleaned_records)} genuine full PSA sentences from existing scraped file (purged {len(df_old)-len(cleaned_records)} headlines/titles).")
    
    # Step 2: Fetch fresh real sentences from full web article bodies
    web_sentences = fetch_web_psa_sentences(max_articles=200)
    print(f"Extracted {len(web_sentences)} fresh genuine full PSA sentences from web article bodies.")
    
    translator = GoogleTranslator(source='en', target='sw')
    
    # Translate fresh web sentences
    for idx, en in enumerate(web_sentences[:150]):
        key = en.lower()
        if key in seen:
            continue
        seen.add(key)
        
        try:
            sw = translator.translate(en)
            if sw:
                cleaned_records.append({
                    "English": en,
                    "Kiswahili": sw,
                    "Domain": label_domain(en)
                })
                time.sleep(0.05)
        except Exception:
            pass
            
    df_out = pd.DataFrame(cleaned_records, columns=["English", "Kiswahili", "Domain"])
    output_path = os.path.join("data", "real_web_psas.csv")
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    
    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved {len(df_out)} REAL full PSA sentences to '{output_path}'.")
    print("=" * 60)
    
    print("\nSample Real PSA Sentences (First 5):")
    for idx, r in df_out.head(5).iterrows():
        print(f"[{r['Domain']}]")
        print(f"  EN: {r['English']}")
        print(f"  SW: {r['Kiswahili']}\n")

if __name__ == "__main__":
    main()
