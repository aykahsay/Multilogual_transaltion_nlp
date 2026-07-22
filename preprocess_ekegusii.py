import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    input_path = os.path.join("data", "_PSA_EnGuz.csv")
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
        
    print(f"Loading Ekegusii dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Rename columns to standard schema
    df = df.rename(columns={"en": "English", "guz": "Ekegusii"})
    df = df[["English", "Ekegusii", "Domain"]].dropna().reset_index(drop=True)
    
    # Clean whitespace
    df["English"] = df["English"].astype(str).str.strip()
    df["Ekegusii"] = df["Ekegusii"].astype(str).str.strip()
    df["Domain"] = df["Domain"].astype(str).str.strip()
    
    # Filter empty strings
    df = df[(df["English"] != "") & (df["Ekegusii"] != "")].drop_duplicates(subset=["English", "Ekegusii"]).reset_index(drop=True)
    
    print(f"Total clean Ekegusii parallel sentences: {len(df)}")
    
    # Split into train (80%), dev (10%), test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_path = os.path.join(output_dir, "train_guz.csv")
    dev_path = os.path.join(output_dir, "dev_guz.csv")
    test_path = os.path.join(output_dir, "test_guz.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    dev_df.to_csv(dev_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"\nEkegusii Dataset Split Summary:")
    print(f"  Train set (80%): {len(train_df)} rows -> {train_path}")
    print(f"  Dev set   (10%): {len(dev_df)} rows -> {dev_path}")
    print(f"  Test set  (10%): {len(test_df)} rows -> {test_path}")

if __name__ == "__main__":
    main()
