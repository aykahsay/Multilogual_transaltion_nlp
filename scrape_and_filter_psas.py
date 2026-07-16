import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import os

def load_ml_classifier(model_path="psa_classifier.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    print("Loading ML Classifier and Vectorizer...")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Classifier or Vectorizer not found. Please run train_psa_classifier.py first.")
    
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return classifier, vectorizer

def rule_based_filter(text):
    # Keyword Heuristics
    keywords = ["alert", "health", "safety", "prevention", "warning", 
                "public notice", "disease", "emergency", "mandatory", 
                "guidelines", "protect", "outbreak", "crisis", "vaccin"]
    
    text_lower = text.lower()
    
    # Must contain at least one keyword
    has_keyword = any(kw in text_lower for kw in keywords)
    
    # Must be of reasonable length to be a sentence/announcement
    is_sentence = len(text.split()) > 4
    
    return has_keyword and is_sentence

def scrape_and_filter(classifier, vectorizer, num_pages=5):
    base_url = "https://reliefweb.int/updates"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # ReliefWeb Source IDs:
    # 1503: World Health Organization (WHO)
    # 131: Kenya (Country ID)
    # We will search specifically for WHO updates and general Kenya Government updates
    sources_to_scrape = [
        {"name": "WHO Africa & Global", "params": {"source": 1503, "primary_country": 131}},
        {"name": "Government of Kenya", "params": {"source": 1391, "primary_country": 131}}, # 1391 is Gov Kenya
        {"name": "General Kenya Alerts", "params": {"primary_country": 131}}
    ]
    
    raw_scraped = []
    
    for source in sources_to_scrape:
        print(f"\nStarting Source-Level Scrape ({num_pages} pages) from: {source['name']}")
        for page in range(num_pages):
            params = source['params'].copy()
            params['page'] = page
            
            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article')
                
                for article in articles:
                    title_tag = article.find('h3')
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        raw_scraped.append(title)
            except Exception as e:
                print(f"Error on page {page} for {source['name']}: {e}")
            
    print(f"Total raw sentences scraped: {len(raw_scraped)}")
    
    # Apply Pipeline
    filtered_results = []
    
    print("Applying Pipeline: Rule-Based (Keywords) -> Machine Learning (Classifier)")
    for text in raw_scraped:
        # 1. Rule-Based Filter
        if rule_based_filter(text):
            # 2. ML Classifier Filter
            vec_text = vectorizer.transform([text])
            prob = classifier.predict_proba(vec_text)[0][1] # Probability of being a PSA (class 1)
            
            if prob > 0.85: # High Confidence Threshold
                filtered_results.append({
                    "English": text,
                    "PSA_Probability": round(prob, 4)
                })
                
    return filtered_results

def main():
    try:
        classifier, vectorizer = load_ml_classifier()
    except Exception as e:
        print(e)
        return
        
    final_data = scrape_and_filter(classifier, vectorizer, num_pages=50)
    
    print(f"\nPipeline finished. Kept {len(final_data)} highly-confident PSAs.")
    
    if final_data:
        df = pd.DataFrame(final_data)
        output_file = "high_quality_scraped_psas.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        print("\n--- Sample of highly confident PSAs ---")
        for i, row in df.head(5).iterrows():
            print(f"[Prob: {row['PSA_Probability']:.2f}] {row['English']}")
    else:
        print("No PSAs passed all filters.")

if __name__ == "__main__":
    main()
