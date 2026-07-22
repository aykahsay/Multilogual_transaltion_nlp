import pandas as pd
import pickle
import os

def load_ml_classifier():
    model_path = "psa_classifier.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Classifier or Vectorizer not found.")
    
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return classifier, vectorizer

def main():
    print("Loading extracted Treasury PDFs...")
    df = pd.read_csv("data/treasury_psas.csv")
    classifier, vectorizer = load_ml_classifier()
    
    true_psas_found = []
    
    print("Scanning PDF text for true PSAs using the ML Classifier...\n")
    
    for idx, row in df.iterrows():
        text = str(row['English'])
        filename = row['Filename']
        
        # Break massive PDF texts into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip().split()) > 5]
        
        for p in paragraphs:
            # Check basic rules to avoid pure garbage
            keywords = ["alert", "notice", "public", "warning", "guidelines", "mandatory", "attention"]
            if any(kw in p.lower() for kw in keywords):
                vec_text = vectorizer.transform([p])
                prob = classifier.predict_proba(vec_text)[0][1]
                
                if prob > 0.85:
                    true_psas_found.append({
                        "Filename": filename,
                        "PSA_Probability": prob,
                        "Text_Snippet": p
                    })
                    
    # Sort by highest confidence
    true_psas_found = sorted(true_psas_found, key=lambda x: x['PSA_Probability'], reverse=True)
    
    print(f"Found {len(true_psas_found)} highly confident PSA snippets inside the Treasury PDFs!\n")
    
    print("--- Top 5 Most Confident True PSAs ---")
    for i in range(min(5, len(true_psas_found))):
        psa = true_psas_found[i]
        print(f"\n[{psa['Filename']}] - Prob: {psa['PSA_Probability']:.4f}")
        print(f"Snippet: {psa['Text_Snippet']}")

if __name__ == "__main__":
    main()
