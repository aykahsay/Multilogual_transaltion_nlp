"""
PSA Data Collection Pipeline — Improved Strategy
=================================================
Collects genuine Public Service Announcement (PSA) sentences for Kenya from
three complementary sources:

  1. ReliefWeb REST API  — fetches full article *bodies*, mines PSA sentences
  2. UNICEF Kenya        — press releases with child/health safety content
  3. NDMA Kenya          — National Drought Management Authority advisories

Each raw sentence passes through a two-layer filter:
  Layer 1: Linguistic PSA detector (imperative/advisory patterns + length)
  Layer 2: ML classifier (TF-IDF + pre-trained PSA classifier)

Kept sentences are translated English → Kiswahili and saved to
data/scraped_psas_translated.csv for downstream merging by preprocess.py.
"""

import os
import re
import time
import pickle
import requests
import pandas as pd
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELIEFWEB_API_URL = "https://api.reliefweb.int/v1/reports"
RELIEFWEB_APP_NAME = "kenya-psa-nlp-collector"

# Imperative / action verbs that signal a directive PSA sentence
IMPERATIVE_VERBS = [
    "wash", "use", "wear", "stay", "avoid", "report", "seek", "call",
    "contact", "register", "enroll", "attend", "follow", "ensure",
    "protect", "vaccinate", "isolate", "test", "verify", "check",
    "keep", "carry", "do not", "don't", "never", "always", "make sure",
    "remember", "be aware", "beware", "be cautious", "be vigilant",
    "be careful", "be alert", "be ready", "be safe", "be wary",
    "please", "kindly", "note that",
    "remind", "urge", "encourage", "advise", "warn", "alert",
    "evacuate", "relocate", "suspend", "ban", "restrict", "close",
    "disinfect", "sanitize", "boil", "treat", "prevent", "limit",
    "reduce", "increase", "maintain", "monitor", "observe", "comply",
    "submit", "apply", "download", "visit", "access", "collect",
]

# Advisory phrases commonly found in PSAs (passive/institutional voice)
ADVISORY_PHRASES = [
    "are urged", "are advised", "are encouraged", "are reminded",
    "are required", "are requested", "is urged", "is advised",
    "is encouraged", "is reminded", "is required", "is requested",
    "members of the public", "citizens are", "residents are",
    "the public is", "all persons", "all citizens", "all residents",
    "everyone is", "people are", "individuals are",
    "parents are", "farmers are", "drivers are", "travelers are",
    "pregnant women are", "pregnant women should", "pregnant women must",
    "expectant mothers", "caregivers are", "health workers are",
    "students are", "schools must", "counties must", "hospitals must",
    "it is mandatory", "it is important", "it is critical", "it is vital",
    "early warning", "public notice", "public advisory",
    "health advisory", "safety advisory", "travel advisory",
    "outbreak alert", "disease alert", "food security alert",
    "drought alert", "flood alert", "emergency notice",
]

# Patterns that mark a string as a news HEADLINE (to be rejected)
NEWS_HEADLINE_PATTERNS = [
    r"^\s*\w[\w\s]+\s+County\s*:\s*Drought",           # "Laikipia County: Drought..."
    r"^\s*\w[\w\s]+\s+County\s*:\s*Flood",              # "Tana River County: Flood..."
    r"Bulletin\s+for\s+(January|February|March|April|May|June|July|August|September|October|November|December)",
    r"\b(Q1|Q2|Q3|Q4)\s+\d{4}\b",                     # "Q3 2024"
    r"\bKEY\s+MESSAGE\s+UPDATE\b",
    r"\bCOUNTRY\s+BRIEF\b",
    r"\bSITUATION\s+REPORT\b",
    r"\bSITUATION\s+UPDATE\b",
    r"\bOPERATIONAL\s+UPDATE\b",
    r"\bRESEARCH\s+BRIEF\b",
    r"\bPOLICY\s+BRIEF\b",
    r"\bFOOD\s+SECURITY\s+OUTLOOK\b",
    r"\bINFOGRAPHICS?\b",
    r"\bNEWSLETTER\b",
    r"^\s*\w[\w\s,\-]+\s+:\s+\w",                      # Generic "Source : Headline" pattern
    r"\bVolume\s+\d+",                                  # "Volume 20 - June 2026"
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b.*?(Report|Update|Brief|Bulletin)",
    r"^\s*GIEWS\b",
    r"^\s*DREF\b",
    r"^\s*WFP\b",
    r"^\s*UNHCR\b",
    r"^\s*OCHA\b",
    r"^\s*ICPAC\b",
    r"\bMDRKE\d+\b",                                   # Red Cross operation codes
    r"\(IPC Phase \d+\)",                              # "(IPC Phase 3)"
    r"\bIPC\s+Phase\b",
    r"\bReliefWeb\b",
    r"\bOperation\s+(Update|Code)\b",
    r"\bLessons\s+(and|Learned|from)\b",
    r"\bCase\s+Study\b",
    r"\bInternational\s+Federation\b",
    r"\(As of\s",                                      # "(As of April 2024)"
    r"\bAnnex\b",
    r"\bAppendix\b",
    r"^\s*\d+\.",                                       # Numbered list items
    r"^[A-Z][A-Z\s&]+:\s",                            # "UNICEF KENYA: ..."
    r"\bJune\s+\d{4}\b$",                             # ends with month/year only
    r"\bDecember\s+\d{4}\b$",
]

# Keywords that raise PSA relevance score (at least one must appear)
PSA_KEYWORDS = [
    # Health & medical
    "health", "prevention", "warning", "alert", "emergency",
    "disease", "protect", "outbreak", "crisis", "vaccin", "clean",
    "water", "sanit", "hygiene", "nutrition", "malnutrition",
    "cholera", "malaria", "tb ", "hiv", "mpox", "covid", "flu",
    "hospital", "clinic", "antenatal", "prenatal", "postnatal",
    "maternal", "infant", "child", "newborn", "pregnancy", "breastfeed",
    "mental", "counsel", "counseling", "treatment", "diagnos",

    # Disaster & environment
    "flood", "drought", "food", "famine", "hunger", "shelter",
    "evacuate", "evacuation", "displacement", "relief", "aid",
    "climate", "weather", "heat", "temperature", "season", "rainfall",
    "fire", "wildfire", "earthquake", "disaster",

    # Security & safety
    "safety", "security", "crime", "fraud", "scam", "trafficking",
    "violence", "gbv", "abuse", "harassment", "theft", "robbery",
    "police", "accident", "road", "traffic", "helmet", "seatbelt",
    "drug", "alcohol", "substance", "addiction",
    "atm", "pin", "password", "banking", "surfing", "ponzi",
    "forgery", "corruption", "bribery", "extortion",

    # Education
    "education", "school", "exam", "pupil", "student", "teacher",
    "enroll", "textbook", "learning", "literacy", "bursary",
    "scholarship", "curriculum", "kcse", "kcpe", "tvet", "university",

    # Governance & civic
    "register", "vote", "election", "iebc", "tax", "fee",
    "deadline", "fine", "permit", "license", "penalty", "comply",
    "government", "county", "nairobi", "national id", "birth cert",

    # Agriculture & livelihoods
    "livestock", "farm", "crop", "pest", "seed", "subsidy",
    "irrigation", "harvest", "agri", "veterinary", "animal",
    "pastoral", "drought", "food security",

    # Financial inclusion
    "loan", "mpesa", "mobile money", "savings", "insurance",
    "pension", "microfinance", "cooperati",
]


# ---------------------------------------------------------------------------
# Helper: Load ML Classifier
# ---------------------------------------------------------------------------

def load_ml_classifier(
    model_path="psa_classifier.pkl",
    vectorizer_path="tfidf_vectorizer.pkl",
):
    """Load pre-trained TF-IDF vectorizer and PSA classifier from disk."""
    print("Loading ML Classifier and Vectorizer...")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "Classifier or Vectorizer not found. "
            "Please run train_psa_classifier.py first."
        )
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return classifier, vectorizer


# ---------------------------------------------------------------------------
# Layer 1a: Reject obvious news headlines
# ---------------------------------------------------------------------------

def is_news_headline(text: str) -> bool:
    """Return True if `text` looks like a news/report title (should be rejected)."""
    for pattern in NEWS_HEADLINE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 1b: Linguistic PSA detector
# ---------------------------------------------------------------------------

def is_psa_sentence(text: str) -> bool:
    """
    Return True if `text` reads like an authentic PSA sentence.

    Criteria (all must pass):
    - Not a news headline
    - Between 6 and 80 words
    - Contains at least one PSA keyword
    - Contains at least one imperative verb OR advisory phrase
    """
    # Reject headlines first
    if is_news_headline(text):
        return False

    words = text.split()
    word_count = len(words)

    # Length gate: too short (noise) or too long (paragraph) are both bad
    if word_count < 6 or word_count > 80:
        return False

    text_lower = text.lower()

    # Must contain at least one domain keyword
    has_keyword = any(kw in text_lower for kw in PSA_KEYWORDS)
    if not has_keyword:
        return False

    # Must have an imperative verb OR advisory phrase
    has_imperative = any(text_lower.startswith(v) or f" {v}" in text_lower for v in IMPERATIVE_VERBS)
    has_advisory = any(phrase in text_lower for phrase in ADVISORY_PHRASES)

    return has_imperative or has_advisory


# ---------------------------------------------------------------------------
# Layer 2: ML classifier filter
# ---------------------------------------------------------------------------

def filter_with_classifier(
    sentences: list,
    classifier,
    vectorizer,
    threshold: float = 0.50,
) -> list:
    """
    Keep only sentences where the ML classifier assigns PSA probability ≥ threshold.
    A lower threshold (0.50) is intentional here because Layer 1 already pre-filters.
    """
    if not sentences:
        return []
    features = vectorizer.transform(sentences)
    probs = classifier.predict_proba(features)[:, 1]
    return [s for s, p in zip(sentences, probs) if p >= threshold]


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def split_body_into_sentences(body: str) -> list:
    """
    Split a long article body into individual sentences.
    Returns only non-empty strings.
    """
    if not body or not isinstance(body, str):
        return []

    # Strip HTML if any slipped through
    body = re.sub(r"<[^>]+>", " ", body)

    # Normalise whitespace
    body = re.sub(r"\s+", " ", body).strip()

    # Split on sentence-ending punctuation followed by space + capital letter
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)

    # Also split on newlines that separate items (common in advisory lists)
    sentences = []
    for chunk in raw:
        sub = [s.strip() for s in chunk.split("\n") if s.strip()]
        sentences.extend(sub)

    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Source 1: ReliefWeb REST API — full article body text
# ---------------------------------------------------------------------------

def fetch_reliefweb_api_bodies(
    num_articles: int = 600,
    country_iso3: str = "KEN",
) -> list:
    """
    Fetch full article body text from the ReliefWeb API for Kenya.
    The API is open (no key required) and returns structured JSON.
    Returns a flat list of raw sentences extracted from article bodies.
    """
    print(f"\n[ReliefWeb API] Fetching up to {num_articles} article bodies for {country_iso3}...")

    # Report types most likely to contain advisory/PSA sentences
    report_types = ["advisory", "situation-report", "news-and-press-release", "report", "manual-and-guideline"]

    all_sentences = []
    page_size = 100  # Max allowed by the API
    fetched = 0

    headers = {"Content-Type": "application/json"}

    for report_type in report_types:
        offset = 0
        type_target = max(1, num_articles // len(report_types))

        while fetched < num_articles:
            payload = {
                "appname": RELIEFWEB_APP_NAME,
                "fields": {
                    "include": ["title", "body", "theme.name"]
                },
                "filter": {
                    "operator": "AND",
                    "conditions": [
                        {"field": "primary_country.iso3", "value": country_iso3},
                        {"field": "format.name",          "value": report_type},
                    ]
                },
                "sort": ["date.created:desc"],
                "limit": min(page_size, num_articles - fetched),
                "offset": offset,
            }

            try:
                resp = requests.post(
                    RELIEFWEB_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=20,
                )
                if resp.status_code != 200:
                    print(f"  API returned {resp.status_code} for type={report_type}. Skipping.")
                    break

                data = resp.json()
                articles = data.get("data", [])
                if not articles:
                    break  # No more results for this type

                for article in articles:
                    fields = article.get("fields", {})
                    body = fields.get("body", "")
                    sentences = split_body_into_sentences(body)
                    all_sentences.extend(sentences)
                    fetched += 1

                offset += len(articles)
                print(f"  [{report_type}] Fetched {fetched} articles, {len(all_sentences)} raw sentences so far.")

                if fetched >= type_target:
                    break

                time.sleep(0.3)  # Polite delay

            except Exception as exc:
                print(f"  ReliefWeb API error: {exc}")
                break

    print(f"[ReliefWeb API] Done. Extracted {len(all_sentences)} raw sentences from {fetched} articles.")
    return all_sentences


# ---------------------------------------------------------------------------
# Source 2: UNICEF Kenya press releases
# ---------------------------------------------------------------------------

def scrape_unicef_kenya_sentences() -> list:
    """
    Scrape UNICEF Kenya press release pages and extract PSA-candidate sentences
    from article body paragraphs.
    """
    print("\n[UNICEF Kenya] Scraping press releases...")
    base_url = "https://www.unicef.org/kenya/press-releases"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NLP-Research-Bot/1.0)"}
    sentences = []

    # Collect article links from the listing page
    article_links = []
    try:
        for page in range(0, 3):  # Fetch up to 3 pages of listings
            params = {"page": page}
            resp = requests.get(base_url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            for a_tag in soup.select("a[href*='/kenya/press-releases/']"):
                href = a_tag.get("href", "")
                if href and href not in article_links:
                    full = href if href.startswith("http") else f"https://www.unicef.org{href}"
                    article_links.append(full)
            time.sleep(0.5)
    except Exception as exc:
        print(f"  UNICEF listing error: {exc}")

    print(f"  Found {len(article_links)} article links. Scraping bodies...")

    # Fetch each article and extract paragraph text
    for url in article_links[:40]:  # Limit to 40 articles
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.select("div.field--name-body p, article p, .content p")
            for p in paragraphs:
                text = p.get_text(separator=" ", strip=True)
                sents = split_body_into_sentences(text)
                sentences.extend(sents)
            time.sleep(0.4)
        except Exception as exc:
            print(f"  Error fetching {url}: {exc}")

    print(f"[UNICEF Kenya] Extracted {len(sentences)} raw sentences.")
    return sentences


# ---------------------------------------------------------------------------
# Source 3: NDMA Kenya drought/disaster advisories
# ---------------------------------------------------------------------------

def scrape_ndma_kenya_sentences() -> list:
    """
    Scrape the Kenya National Drought Management Authority (NDMA) website
    for advisory sentences from drought bulletins and early warning reports.
    """
    print("\n[NDMA Kenya] Scraping drought advisories...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NLP-Research-Bot/1.0)"}
    sentences = []

    # NDMA publishes county drought bulletins — try the main news/resources page
    target_urls = [
        "https://www.ndma.go.ke/index.php/resource-center/early-warning",
        "https://www.ndma.go.ke/index.php/resource-center/drought-monitoring",
        "https://www.ndma.go.ke/index.php/news-media/press-releases",
    ]

    for url in target_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"  NDMA returned {resp.status_code} for {url}. Skipping.")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # Extract all paragraph text
            for elem in soup.select("p, li, .item-page p"):
                text = elem.get_text(separator=" ", strip=True)
                sents = split_body_into_sentences(text)
                sentences.extend(sents)
            time.sleep(0.5)
        except Exception as exc:
            print(f"  NDMA error for {url}: {exc}")

    print(f"[NDMA Kenya] Extracted {len(sentences)} raw sentences.")
    return sentences


# ---------------------------------------------------------------------------
# Source 4: WHO AFRO Kenya health messages
# ---------------------------------------------------------------------------

def scrape_who_afro_kenya_sentences() -> list:
    """
    Scrape WHO Africa Kenya health advisories and press releases.
    """
    print("\n[WHO AFRO Kenya] Scraping health advisories...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NLP-Research-Bot/1.0)"}
    sentences = []

    target_urls = [
        "https://www.afro.who.int/countries/kenya/news",
        "https://www.afro.who.int/countries/kenya",
    ]

    for url in target_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"  WHO AFRO returned {resp.status_code}. Skipping.")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for elem in soup.select("p, li, .field-body p, article p"):
                text = elem.get_text(separator=" ", strip=True)
                sents = split_body_into_sentences(text)
                sentences.extend(sents)
            time.sleep(0.5)
        except Exception as exc:
            print(f"  WHO AFRO error: {exc}")

    print(f"[WHO AFRO Kenya] Extracted {len(sentences)} raw sentences.")
    return sentences


# ---------------------------------------------------------------------------
# Translation — all 4 target languages
# ---------------------------------------------------------------------------

# Languages supported by Google Translate (deep_translator)
_SUPPORTED_LANGS = {
    "Kiswahili": "sw",   # Fully supported ✅
    "Somali":    "so",   # Fully supported ✅
    # Dholuo (luo) and Ekegusii (guz) are NOT supported by Google Translate.
    # They will remain None in the output, matching the original dataset behaviour
    # for scraped rows (preprocess.py fills these columns as None automatically).
}


def translate_text(text: str, lang_code: str, lang_name: str) -> str | None:
    """
    Translate `text` from English to `lang_code` using Google Translate.
    Returns None on failure or if the translation is identical to the input
    (a common sign of an unsupported language silently falling back).
    """
    try:
        result = GoogleTranslator(source="en", target=lang_code).translate(text)
        # Reject if translation came back identical to input (unsupported lang fallback)
        if result and result.strip().lower() != text.strip().lower():
            return result.strip()
        return None
    except Exception as exc:
        print(f"  [{lang_name}] Translation error ('{text[:35]}...'): {exc}")
        return None


def translate_all_languages(text: str) -> dict:
    """
    Translate a single English sentence into all supported target languages.
    Returns a dict with keys: Kiswahili, Ekegusii, Dholuo, Somali.
    Unsupported or failed languages return None.
    """
    translations = {}
    for lang_name, lang_code in _SUPPORTED_LANGS.items():
        translations[lang_name] = translate_text(text, lang_code, lang_name)
        time.sleep(0.15)  # Polite rate-limit between each language call

    # Ekegusii and Dholuo — not supported, leave as None
    translations["Ekegusii"] = None
    translations["Dholuo"]   = None

    return translations


# ---------------------------------------------------------------------------
# Domain labeller
# ---------------------------------------------------------------------------

_DOMAIN_MAP = {
    "Health": [
        "health", "disease", "vaccin", "hospital", "clinic", "medical",
        "nurse", "doctor", "hiv", "malaria", "tb", "cholera", "mpox",
        "outbreak", "epidemic", "pandemic", "infant", "maternal",
        "nutrition", "breastfeed", "sanit", "hygiene", "mental",
    ],
    "Education": [
        "school", "student", "pupil", "teacher", "exam", "kcse", "kcpe",
        "university", "college", "tvet", "enroll", "curriculum", "tuition",
        "bursary", "scholarship", "textbook", "learning", "literacy",
    ],
    "Agriculture": [
        "farm", "crop", "livestock", "seed", "fertilizer", "pest",
        "drought", "irrigation", "harvest", "food security", "maize",
        "animal", "veterinary", "arid", "pastoral",
    ],
    "Governance": [
        "vote", "election", "iebc", "government", "corruption", "eacc",
        "tax", "kra", "court", "law", "county", "parliament", "citizen",
        "registration", "id card", "birth certificate", "national",
    ],
    "Security": [
        "police", "crime", "theft", "fraud", "scam", "trafficking",
        "violence", "gbv", "safety", "shelter", "fire", "road", "traffic",
        "alcohol", "drug", "substance", "child protection",
    ],
}


def label_domain(text: str) -> str:
    """Return the most likely PSA domain based on keyword matching."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in _DOMAIN_MAP.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    # Fall back to Disaster/Health if no domain scored
    return best if scores[best] > 0 else "Disaster/Health"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    output_path = os.path.join("data", "scraped_psas_translated.csv")
    os.makedirs("data", exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Collect raw sentences from all sources
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Collecting raw sentences from all sources")
    print("=" * 60)

    all_raw: list[str] = []

    # Primary source: ReliefWeb API (most reliable, no blocking)
    rw_sentences = fetch_reliefweb_api_bodies(num_articles=600)
    all_raw.extend(rw_sentences)

    # Secondary: UNICEF Kenya
    unicef_sentences = scrape_unicef_kenya_sentences()
    all_raw.extend(unicef_sentences)

    # Secondary: NDMA Kenya
    ndma_sentences = scrape_ndma_kenya_sentences()
    all_raw.extend(ndma_sentences)

    # Secondary: WHO AFRO Kenya
    who_sentences = scrape_who_afro_kenya_sentences()
    all_raw.extend(who_sentences)

    print(f"\nTotal raw sentences collected: {len(all_raw)}")

    # Deduplicate
    seen = set()
    unique_raw = []
    for s in all_raw:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_raw.append(s.strip())
    print(f"After deduplication: {len(unique_raw)} unique sentences")

    # ------------------------------------------------------------------
    # Step 2 — Layer 1: Linguistic PSA filter
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Layer 1 — Linguistic PSA filter")
    print("=" * 60)

    linguistically_filtered = [s for s in unique_raw if is_psa_sentence(s)]
    print(f"Passed linguistic filter: {len(linguistically_filtered)} / {len(unique_raw)}")

    if not linguistically_filtered:
        print("ERROR: No sentences passed the linguistic filter. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 3 — Layer 2: ML classifier filter
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Layer 2 — ML classifier filter")
    print("=" * 60)

    try:
        classifier, vectorizer = load_ml_classifier()
        ml_filtered = filter_with_classifier(
            linguistically_filtered, classifier, vectorizer, threshold=0.45
        )
        print(f"Passed ML classifier: {len(ml_filtered)} / {len(linguistically_filtered)}")
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")
        print("Proceeding without ML filter — linguistic filter only.")
        ml_filtered = linguistically_filtered

    # ------------------------------------------------------------------
    # Step 4 — Translate to all supported languages
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 4: Translating {len(ml_filtered)} sentences to all target languages")
    print("  Supported   : Kiswahili (sw), Somali (so)")
    print("  Unsupported : Ekegusii (guz), Dholuo (luo) — will be None")
    print("=" * 60)

    records = []
    for idx, text in enumerate(ml_filtered):
        # Translate to all languages in one call
        langs = translate_all_languages(text)

        # Only keep the record if at least Kiswahili succeeded
        if langs.get("Kiswahili"):
            records.append({
                "Domain":    label_domain(text),
                "English":   text,
                "Kiswahili": langs["Kiswahili"],
                "Ekegusii":  langs["Ekegusii"],   # None — not supported
                "Dholuo":    langs["Dholuo"],      # None — not supported
                "Somali":    langs["Somali"],      # Translated if successful
            })

        if (idx + 1) % 50 == 0:
            sw_ok = sum(1 for r in records if r["Kiswahili"])
            so_ok = sum(1 for r in records if r["Somali"])
            print(f"  Progress: {idx + 1}/{len(ml_filtered)} — "
                  f"Kiswahili: {sw_ok}, Somali: {so_ok}")

    print(f"\nSuccessfully translated {len(records)} sentences.")
    so_count = sum(1 for r in records if r["Somali"])
    print(f"  Kiswahili filled : {len(records)} / {len(records)}")
    print(f"  Somali filled    : {so_count} / {len(records)}")
    print(f"  Ekegusii filled  : 0 / {len(records)}  (unsupported by Google Translate)")
    print(f"  Dholuo filled    : 0 / {len(records)}  (unsupported by Google Translate)")

    # ------------------------------------------------------------------
    # Step 5 — Save output with full 6-column structure
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Saving output")
    print("=" * 60)

    # Column order matches the original dataset exactly:
    # English, Kiswahili, Ekegusii, Dholuo, Somali, Domain
    OUTPUT_COLUMNS = ["English", "Kiswahili", "Ekegusii", "Dholuo", "Somali", "Domain"]
    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} PSA rows to '{output_path}'.")
    print(f"Columns: {list(df.columns)}")

    # Show sample for manual inspection
    print("\n--- Sample of collected PSA sentences (first 5) ---")
    for _, row in df.head(5).iterrows():
        print(f"  [{row['Domain']}] EN: {row['English'][:70]}")
        print(f"           SW: {str(row['Kiswahili'])[:70]}")
        print(f"           SO: {str(row['Somali'])[:70]}")
        print()

    # ------------------------------------------------------------------
    # Step 6 — Run preprocess.py to regenerate train/dev/test splits
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Running preprocess.py to regenerate dataset splits")
    print("=" * 60)

    try:
        import preprocess
        preprocess.main()

        df_train = pd.read_csv("data/train.csv")
        df_dev   = pd.read_csv("data/dev.csv")
        df_test  = pd.read_csv("data/test.csv")
        total    = len(df_train) + len(df_dev) + len(df_test)

        print(f"\nFinal combined dataset: {total} rows "
              f"(train={len(df_train)}, dev={len(df_dev)}, test={len(df_test)})")

        if total > 5000:
            print("SUCCESS! Dataset exceeds 5000 rows.")
        else:
            print(f"WARNING: Dataset is {total} rows (target: >5000). "
                  "Consider increasing num_articles or lowering the threshold.")
    except Exception as exc:
        print(f"Preprocessing error: {exc}")


if __name__ == "__main__":
    main()
