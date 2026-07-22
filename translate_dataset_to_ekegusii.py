import os
import pandas as pd

def main():
    print("Mapping Ekegusii translations to primary train/dev/test datasets...")
    
    # 1. Load Ekegusii parallel data map
    guz_path = os.path.join("data", "_PSA_EnGuz.csv")
    if not os.path.exists(guz_path):
        raise FileNotFoundError(f"Missing {guz_path}")
        
    df_guz = pd.read_csv(guz_path)
    df_guz = df_guz.rename(columns={"en": "English", "guz": "Ekegusii"})
    
    # Create normalized lookup dictionary
    lookup = {}
    for _, row in df_guz.iterrows():
        en_norm = str(row["English"]).strip().lower()
        guz_val = str(row["Ekegusii"]).strip()
        if en_norm and guz_val:
            lookup[en_norm] = guz_val
            
    print(f"Loaded {len(lookup)} unique English -> Ekegusii translation mappings.")
    
    # 2. Update train.csv, dev.csv, and test.csv
    splits = ["train.csv", "dev.csv", "test.csv"]
    output_dir = "data"
    
    for sname in splits:
        spath = os.path.join(output_dir, sname)
        if os.path.exists(spath):
            df_split = pd.read_csv(spath)
            
            # Map Ekegusii translations
            ekegusii_col = []
            matched = 0
            
            for _, row in df_split.iterrows():
                en_key = str(row["English"]).strip().lower()
                if en_key in lookup:
                    ekegusii_col.append(lookup[en_key])
                    matched += 1
                else:
                    # Default placeholder if exact match not found
                    ekegusii_col.append("N/A - Pending Fine-Tuned Model Inference")
                    
            df_split["Ekegusii"] = ekegusii_col
            
            # Reorder columns: English, Kiswahili, Ekegusii, Domain
            cols = [c for c in ["English", "Kiswahili", "Ekegusii", "Domain"] if c in df_split.columns]
            df_split = df_split[cols]
            
            df_split.to_csv(spath, index=False, encoding="utf-8")
            print(f"Updated {sname}: {len(df_split)} rows ({matched} matched Ekegusii translations).")

if __name__ == "__main__":
    main()
