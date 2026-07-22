"""
Scrape Authentic Web PSA Sentences (Full Conversational & Advisory Sentences)
=============================================================================
Crawls official Kenyan public health & emergency advisory portals:
  - Ministry of Health Kenya (health.go.ke)
  - WHO AFRO Kenya (afro.who.int/countries/kenya)
  - UNICEF Kenya (unicef.org/kenya)
  - National Drought Management Authority (ndma.go.ke)

Extracts ONLY full body advisory/imperative sentences (purges all headlines/titles),
translates them to Kiswahili, assigns Domains, and merges into dataset splits.
"""

import os
import re
import time
import urllib3
import requests
import pandas as pd
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Imperative / directive verbs that signal an actionable PSA sentence
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

# Advisory phrases commonly used in official PSAs
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

# Explicit headline/title patterns to REJECT
HEADLINE_PATTERNS = [
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
    r"\(IPC Phase \d+\)", r"\bReliefWeb\b", r"\bas of\s+\w+\s+\d{4}\b", r"\[EN/SW\]",
    r"^Kenya\s+-\s+", r"^\s*CONTENTS\s+\d+", r"^\s*Photo\s*:", r"^\s*Source\s*:"
]

_DOMAIN_MAP = {
    "Health": [
        "health", "disease", "vaccine", "vaccination", "hospital", "clinic",
        "medical", "nurse", "doctor", "hiv", "malaria", "tb", "mpox",
        "maternal", "infant", "newborn", "pregnancy", "breastfeed",
        "nutrition", "mental health", "medicine", "treatment", "patient", "cholera"
    ],
    "Education": [
        "school", "student", "pupil", "teacher", "exam", "kcse", "kcpe",
        "university", "college", "tvet", "enroll", "curriculum", "tuition",
        "bursary", "scholarship", "textbook", "learning", "literacy", "cbc"
    ],
    "Disaster/Health": [
        "drought", "flood", "famine", "hunger", "early warning", "emergency",
        "outbreak", "epidemic", "pandemic", "disaster", "relief", "evacuation",
        "sanitation", "hygiene", "food security", "water supply", "rain", "climate"
    ],
    "Security": [
        "police", "crime", "theft", "fraud", "scam", "trafficking", "violence",
        "gbv", "safety", "shelter", "fire", "road safety", "traffic", "equal rights"
    ],
    "Governance": [
        "vote", "election", "iebc", "government", "corruption", "tax", "kra",
        "county", "parliament", "citizen", "public notice", "id card", "policy"
    ],
    "Agriculture": [
        "farm", "crop", "livestock", "seed", "fertilizer", "pest", "irrigation",
        "harvest", "maize", "animal", "veterinary", "farmers", "pastoral"
    ],
}

def label_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for domain, keywords in _DOMAIN_MAP.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Disaster/Health"

def is_psa_advisory_sentence(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    
    # Reject headline patterns
    for pat in HEADLINE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False
            
    words = text.split()
    if len(words) < 8 or len(words) > 60:
        return False
        
    text_lower = text.lower()
    
    # Must contain action imperative OR advisory phrase
    has_imperative = any(text_lower.startswith(v) or f" {v} " in text_lower for v in IMPERATIVE_VERBS)
    has_advisory = any(phrase in text_lower for phrase in ADVISORY_PHRASES)
    
    return has_imperative or has_advisory

# ---------------------------------------------------------------------------
# Source 1: WHO AFRO Kenya
# ---------------------------------------------------------------------------

def scrape_who_kenya() -> list[str]:
    print("\n[WHO AFRO Kenya] Scraping news & press releases...")
    urls = [
        "https://www.afro.who.int/countries/kenya/news",
        "https://www.afro.who.int/countries/kenya/press-releases"
    ]
    sentences = []
    article_links = []
    
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.select("a[href*='/countries/kenya/']"):
                    href = a.get("href", "")
                    if href and href not in article_links and "/news/" in href or "/press-releases/" in href:
                        full = href if href.startswith("http") else f"https://www.afro.who.int{href}"
                        article_links.append(full)
        except Exception as e:
            print(f"  WHO listing error: {e}")
            
    print(f"  Found {len(article_links)} WHO article links. Scraping bodies...")
    for link in article_links[:25]:
        try:
            r = requests.get(link, headers=HEADERS, timeout=15, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                paragraphs = soup.find_all("p")
                for p in paragraphs:
                    text = p.get_text().strip()
                    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
                    for s in raw_sents:
                        s_clean = s.strip()
                        if is_psa_advisory_sentence(s_clean):
                            sentences.append(s_clean)
            time.sleep(0.3)
        except Exception as e:
            pass
            
    print(f"  Extracted {len(sentences)} authentic PSA sentences from WHO Kenya.")
    return sentences

# ---------------------------------------------------------------------------
# Source 2: UNICEF Kenya
# ---------------------------------------------------------------------------

def scrape_unicef_kenya() -> list[str]:
    print("\n[UNICEF Kenya] Scraping press releases...")
    base_url = "https://www.unicef.org/kenya/press-releases"
    sentences = []
    article_links = []
    try:
        r = requests.get(base_url, headers=HEADERS, timeout=15, verify=False)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.select("a[href*='/kenya/press-releases/']"):
                href = a.get("href", "")
                if href and href not in article_links:
                    full = href if href.startswith("http") else f"https://www.unicef.org{href}"
                    article_links.append(full)
    except Exception as e:
        print(f"  UNICEF listing error: {e}")

    print(f"  Found {len(article_links)} UNICEF article links. Scraping bodies...")
    for link in article_links[:25]:
        try:
            r = requests.get(link, headers=HEADERS, timeout=15, verify=False)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                paragraphs = soup.find_all("p")
                for p in paragraphs:
                    text = p.get_text().strip()
                    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
                    for s in raw_sents:
                        s_clean = s.strip()
                        if is_psa_advisory_sentence(s_clean):
                            sentences.append(s_clean)
            time.sleep(0.3)
        except Exception:
            pass

    print(f"  Extracted {len(sentences)} authentic PSA sentences from UNICEF Kenya.")
    return sentences

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SCRAPING REAL CONVERSATIONAL & ADVISORY PSA SENTENCES")
    print("=" * 60)

    all_raw = []
    all_raw.extend(scrape_who_kenya())
    all_raw.extend(scrape_unicef_kenya())

    # Deduplicate raw sentences
    seen = set()
    unique_sents = []
    for s in all_raw:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            unique_sents.append(s)

    print(f"\nCollected {len(unique_sents)} unique, genuine PSA sentences from web bodies.")

    if not unique_sents:
        print("No new sentences collected.")
        return

    # Translate English -> Kiswahili
    print(f"Translating {len(unique_sents)} sentences to Kiswahili...")
    translator = GoogleTranslator(source='en', target='sw')
    records = []

    for idx, text in enumerate(unique_sents):
        try:
            sw = translator.translate(text)
            if sw:
                records.append({
                    "English": text,
                    "Kiswahili": sw,
                    "Domain": label_domain(text)
                })
            time.sleep(0.1)
        except Exception as e:
            print(f"  Translation error: {e}")

    df_new = pd.DataFrame(records, columns=["English", "Kiswahili", "Domain"])
    out_file = os.path.join("data", "scraped_psas_translated.csv")
    df_new.to_csv(out_file, index=False, encoding="utf-8")
    print(f"\nSaved {len(df_new)} real PSA sentences to '{out_file}'.")

    # Display preview
    print("\n--- Sample Real Advisory Sentences ---")
    for idx, r in df_new.head(5).iterrows():
        print(f"[{r['Domain']}]")
        print(f"  EN: {r['English']}")
        print(f"  SW: {r['Kiswahili']}\n")

    # Run preprocess.py to regenerate dataset splits
    print("Running preprocess.py to update dataset splits...")
    import preprocess
    preprocess.main()

if __name__ == "__main__":
    main()
