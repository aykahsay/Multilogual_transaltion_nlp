import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def build_retriever_index(csv_path="PSA_KE_Final.csv", index_path="psa_faiss.index", embeddings_path="gold_psas.npy"):
    print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if not os.path.exists(csv_path):
        # Fallback to the one level up if not in the same folder
        csv_path = "../" + csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find {csv_path}. Please ensure it exists.")
            
    print(f"Loading Gold Standard PSAs from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Using English column as our reference semantics
    psas = df['English'].dropna().tolist()
    
    # We can also add some synthetic strong prototypical PSA queries to boost recall
    prototypical_queries = [
        "This is a public service announcement regarding health and safety.",
        "Emergency warning: Please evacuate the area due to severe flooding.",
        "Ministry of Health vaccination campaign guidelines and schedule.",
        "Cholera outbreak alert and prevention methods.",
        "Official government notice regarding public health measures."
    ]
    
    all_reference_texts = psas + prototypical_queries
    
    print(f"Embedding {len(all_reference_texts)} texts into Dense Vectors...")
    embeddings = model.encode(all_reference_texts, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity (Inner Product)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.array(embeddings).astype('float32')
    
    print("Saving Dense Embeddings to disk...")
    # Just save the numpy array. Since we only have a few thousand vectors, 
    # exact numpy dot product is instantaneous and we don't need FAISS.
    np.save(embeddings_path, embeddings)
    
    # Save the reference texts so we can inspect what it matched against
    with open("gold_psas_texts.txt", "w", encoding="utf-8") as f:
        for t in all_reference_texts:
            f.write(t.replace("\n", " ") + "\n")
            
    print("Dense Retriever Setup Complete! Saved embeddings to", embeddings_path)

if __name__ == "__main__":
    build_retriever_index()
