import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    original_dataset_path = r"C:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\PSA_KE_Final.csv"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    columns_to_keep = ["English", "Kiswahili", "Domain"]
    dfs = []
    
    # 1. Load original base dataset
    if os.path.exists(original_dataset_path):
        print(f"Loading original dataset from {original_dataset_path}...")
        df_orig = pd.read_csv(original_dataset_path)
        if all(c in df_orig.columns for c in columns_to_keep):
            df_orig = df_orig[columns_to_keep]
            dfs.append(df_orig)
            print(f"  Added {len(df_orig)} rows from original base dataset.")
            
    # 2. Add the 4 Scraped PSA Parts
    part_files = [
        os.path.join("data", "scraped_psas_part1.csv"),
        os.path.join("data", "scraped_psas_part2.csv"),
        os.path.join("data", "scraped_psas_part3.csv"),
        os.path.join("data", "scraped_psas_part4.csv"),
        os.path.join("data", "scraped_psas_translated.csv")
    ]
    
    for pf in part_files:
        if os.path.exists(pf):
            print(f"Loading scraped PSA part from {pf}...")
            df_part = pd.read_csv(pf)
            if "Domain" not in df_part.columns:
                df_part["Domain"] = "Disaster/Health"
            valid_cols = [c for c in columns_to_keep if c in df_part.columns]
            if "English" in valid_cols and "Kiswahili" in valid_cols:
                df_part = df_part[valid_cols]
                dfs.append(df_part)
                print(f"  Added {len(df_part)} rows from {pf}.")

    # 3. Add verified scraped & extracted PDF datasets
    extra_sources = [
        os.path.join("data", "scraped_psas_verified.csv"),
        os.path.join("data", "kenya_bulk_notices_extracted.csv"),
        os.path.join("data", "extracted_pdf_data.csv"),
        os.path.join("data", "extracted_pdf_psas.csv"),
    ]
    
    for src in extra_sources:
        if os.path.exists(src):
            print(f"Loading dataset from {src}...")
            df_extra = pd.read_csv(src)
            if "Domain" not in df_extra.columns:
                df_extra["Domain"] = "Disaster/Health"
            valid_cols = [c for c in columns_to_keep if c in df_extra.columns]
            if "English" in valid_cols and "Kiswahili" in valid_cols:
                df_extra = df_extra[valid_cols]
                dfs.append(df_extra)
                print(f"  Added {len(df_extra)} rows from {src}.")
    
    if not dfs:
        raise ValueError("No valid datasets found to merge!")
        
    # Combine all dataframes
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal combined rows before cleaning: {len(df_combined)}")
    
    # Clean missing / blank values
    df_combined = df_combined.dropna(subset=["English", "Kiswahili"])
    df_combined = df_combined[(df_combined["English"].astype(str).str.strip() != "") & 
                              (df_combined["Kiswahili"].astype(str).str.strip() != "")]
    
    df_combined["English"] = df_combined["English"].astype(str).str.strip()
    df_combined["Kiswahili"] = df_combined["Kiswahili"].astype(str).str.strip()
    df_combined["Domain"] = df_combined["Domain"].fillna("Disaster/Health").astype(str).str.strip()
    
    # Deduplicate based on English and Kiswahili sentences
    df_clean = df_combined.drop_duplicates(subset=["English", "Kiswahili"]).reset_index(drop=True)
    print(f"Total clean verified rows after deduplication: {len(df_clean)}")
    
    # Split into train (80%), dev/validation (10%), test (10%)
    train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print("\nFinal Dataset Split Summary:")
    print(f"  Train set:      {len(train_df)} rows")
    print(f"  Dev/Val set:    {len(dev_df)} rows")
    print(f"  Test set:       {len(test_df)} rows")
    
    # Save split datasets
    train_path = os.path.join(output_dir, "train.csv")
    dev_path = os.path.join(output_dir, "dev.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    dev_df.to_csv(dev_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"\nSuccessfully updated {train_path}, {dev_path}, and {test_path} in 'data/' directory.")

if __name__ == "__main__":
    main()
