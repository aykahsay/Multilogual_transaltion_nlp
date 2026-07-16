import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from datasets import load_dataset

def get_positive_data(csv_path):
    print("Loading positive examples (PSAs)...")
    df = pd.read_csv(csv_path)
    # The 'English' column contains the PSAs
    psas = df['English'].dropna().tolist()
    # Create labels: 1 for PSA
    labels = [1] * len(psas)
    return psas, labels

def get_negative_data(num_samples):
    print(f"Loading {num_samples} negative examples (News snippets)...")
    # Load ag_news dataset for negative examples (general news)
    dataset = load_dataset("ag_news", split="train")
    
    # Shuffle and select the required number of samples
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    # Extract the text
    news_texts = dataset['text']
    # Create labels: 0 for non-PSA
    labels = [0] * len(news_texts)
    return news_texts, labels

def train_classifier():
    # Paths
    psa_csv_path = "../PSA_KE_Final.csv"
    model_path = "psa_classifier.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    
    if not os.path.exists(psa_csv_path):
        raise FileNotFoundError(f"Cannot find {psa_csv_path}. Please ensure it exists.")

    # 1. Load Data
    psa_texts, psa_labels = get_positive_data(psa_csv_path)
    
    # Get roughly the same number of negative examples to have a balanced dataset
    num_negatives = len(psa_texts)
    neg_texts, neg_labels = get_negative_data(num_negatives)
    
    # Combine
    X_raw = psa_texts + neg_texts
    y = psa_labels + neg_labels
    
    # 2. Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Vectorization (TF-IDF)
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 4. Train Model
    print("Training Logistic Regression classifier...")
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    classifier.fit(X_train, y_train)
    
    # 5. Evaluate Model
    print("Evaluating model...")
    y_pred = classifier.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-PSA", "PSA"]))
    
    # 6. Save Model and Vectorizer
    print("Saving model and vectorizer...")
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("Training complete!")

if __name__ == "__main__":
    train_classifier()
