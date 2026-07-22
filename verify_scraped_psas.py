"""
Verify Scraped PSAs using Pre-trained ML Classifier
===================================================
Evaluates every sentence in data/scraped_psas_translated.csv using the
pre-trained TF-IDF vectorizer and ML classifier (psa_classifier.pkl).

Output:
- Calculates average PSA probability
- Filters true PSAs vs news headlines / non-PSAs
- Saves filtered high-quality PSAs to data/high_quality_scraped_psas.csv
"""

import os
import pickle
import pandas as pd


def load_ml_classifier():
    model_path = "psa_classifier.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("psa_classifier.pkl or tfidf_vectorizer.pkl not found in project root.")
    
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return classifier, vectorizer


def main():
    csv_path = os.path.join("data", "scraped_psas_translated.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("=" * 60)
    print("VERIFYING SCRAPED PSA DATASET USING ML CLASSIFIER")
    print("=" * 60)

    print(f"Loading dataset: {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total rows in dataset: {len(df)}")

    # Ensure required columns
    if "English" not in df.columns:
        print("Error: 'English' column missing in dataset.")
        return

    # Load trained ML model
    print("Loading ML classifier and TF-IDF vectorizer...")
    classifier, vectorizer = load_ml_classifier()

    # Predict probabilities for English sentences
    print("Predicting PSA probabilities for all rows...")
    english_texts = df["English"].astype(str).tolist()
    features = vectorizer.transform(english_texts)
    probabilities = classifier.predict_proba(features)[:, 1]

    df["PSA_Probability"] = probabilities

    # Categorize into High Confidence PSAs vs Low Confidence / Non-PSAs
    high_conf_df = df[df["PSA_Probability"] >= 0.50].sort_values(by="PSA_Probability", ascending=False)
    low_conf_df  = df[df["PSA_Probability"] < 0.50].sort_values(by="PSA_Probability", ascending=True)

    print("\n" + "=" * 60)
    print("CLASSIFICATION STATS & SUMMARY")
    print("=" * 60)
    print(f"Average PSA Probability:  {df['PSA_Probability'].mean():.4f}")
    print(f"True PSAs (Prob >= 0.50): {len(high_conf_df)} rows ({len(high_conf_df)/len(df)*100:.1f}%)")
    print(f"Non-PSAs  (Prob <  0.50): {len(low_conf_df)} rows ({len(low_conf_df)/len(df)*100:.1f}%)")

    # Display Top 5 Most Confident PSAs
    print("\n--- TOP 5 HIGH-CONFIDENCE PSA SENTENCES ---")
    for idx, row in high_conf_df.head(5).iterrows():
        print(f"[Prob: {row['PSA_Probability']:.4f} | Domain: {row.get('Domain', 'N/A')}]")
        print(f"  EN: {row['English']}")
        print(f"  SW: {row.get('Kiswahili', 'N/A')}\n")

    # Display 5 Low-Confidence / Non-PSA Sentences
    print("--- SAMPLE LOW-CONFIDENCE / NON-PSA SENTENCES ---")
    for idx, row in low_conf_df.head(5).iterrows():
        print(f"[Prob: {row['PSA_Probability']:.4f} | Domain: {row.get('Domain', 'N/A')}]")
        print(f"  EN: {row['English']}")
        print(f"  SW: {row.get('Kiswahili', 'N/A')}\n")

    # Save high quality filtered dataset
    output_filtered = os.path.join("data", "scraped_psas_verified.csv")
    high_conf_df.to_csv(output_filtered, index=False, encoding="utf-8")
    print(f"Saved {len(high_conf_df)} verified high-confidence PSAs to '{output_filtered}'.")

if __name__ == "__main__":
    main()
