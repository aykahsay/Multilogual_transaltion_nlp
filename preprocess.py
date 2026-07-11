import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    original_dataset_path = r"C:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\PSA_KE_Final.csv"
    scraped_dataset_path = os.path.join("data", "scraped_psas_translated.csv")
    output_dir = "data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading original dataset from {original_dataset_path}...")
    df_original = pd.read_csv(original_dataset_path)
    
    columns_to_keep = ["English", "Kiswahili", "Domain"]
    
    # Ensure columns exist in original
    missing_cols = [col for col in columns_to_keep if col not in df_original.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in original dataset: {missing_cols}")
        
    df_original = df_original[columns_to_keep]
    
    # Drop rows with missing values in original
    initial_len = len(df_original)
    df_original = df_original.dropna(subset=["English", "Kiswahili"])
    df_original = df_original[(df_original["English"].str.strip() != "") & (df_original["Kiswahili"].str.strip() != "")]
    print(f"Original valid PSA pairs: {len(df_original)}")
    
    # Check if we have scraped data to add
    if os.path.exists(scraped_dataset_path):
        print(f"Loading scraped dataset from {scraped_dataset_path}...")
        df_scraped = pd.read_csv(scraped_dataset_path)
        
        # Ensure Domain column exists in scraped data
        if "Domain" not in df_scraped.columns:
            df_scraped["Domain"] = "Disaster/Health"
            
        df_scraped = df_scraped[columns_to_keep]
        df_scraped = df_scraped.dropna(subset=["English", "Kiswahili"])
        df_scraped = df_scraped[(df_scraped["English"].str.strip() != "") & (df_scraped["Kiswahili"].str.strip() != "")]
        print(f"Scraped valid PSA pairs: {len(df_scraped)}")
        
        # Combine the two datasets
        df = pd.concat([df_original, df_scraped], ignore_index=True)
    else:
        print("No scraped dataset found. Using only original dataset.")
        df = df_original
        
    print(f"Total combined PSA pairs before deduplication: {len(df)}")
    df = df.drop_duplicates(subset=["English", "Kiswahili"])
    print(f"Total valid PSA pairs after deduplication: {len(df)}")
    
    # Split into train (80%), dev (10%), test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Save to CSV
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print("Preprocessing complete. Data saved to 'data' directory.")

if __name__ == "__main__":
    main()
