import pandas as pd
import json
import random
import os
from datetime import datetime, timedelta

def generate_campaign_id(index):
    return f"PSA-KE-2026-{str(index).zfill(4)}"

def generate_ad_id(domain):
    prefix = domain.upper()[:3] if pd.notna(domain) and len(str(domain)) >= 3 else "GEN"
    suffix = random.choice(["15S", "30S", "60S"])
    return f"{prefix}-PSA-{random.randint(100, 999)}-{suffix}"

def generate_metadata():
    # Synthetic sponsors
    sponsors = [
        "Ministry of Health Kenya",
        "Ministry of Education",
        "National Transport and Safety Authority (NTSA)",
        "Kenya Red Cross",
        "Independent Electoral and Boundaries Commission (IEBC)"
    ]
    
    # Synthetic dayparts
    dayparts = [
        "Morning Drive (6AM - 10AM)",
        "Daytime (10AM - 3PM)",
        "Afternoon Drive (3PM - 7PM)",
        "Prime Time (7PM - 11PM)",
        "Overnight (11PM - 6AM)"
    ]
    
    # Synthetic regions
    regions = [
        "Nairobi (Urban)",
        "Mombasa & Coast (Urban/Coastal)",
        "Kisumu & Lake Region (Urban/Rural)",
        "Rift Valley (Mixed)",
        "Northern Kenya (Rural/Arid)"
    ]
    
    # Synthetic tracking
    metrics = [
        {"name": "Hotline call volumes", "range": (500, 5000)},
        {"name": "SMS Opt-ins", "range": (1000, 10000)},
        {"name": "Website visits", "range": (5000, 50000)},
        {"name": "Reported incident decreases", "range": (10, 300)}
    ]
    
    sponsor = random.choice(sponsors)
    daypart = random.choice(dayparts)
    region = random.choice(regions)
    metric = random.choice(metrics)
    
    # Broadcast dates (random within last 6 months)
    days_ago = random.randint(1, 180)
    broadcast_date = datetime.now() - timedelta(days=days_ago)
    
    impressions = random.randint(50000, 1500000)
    grp = round(impressions / 10000 + random.uniform(-5.0, 5.0), 1)
    if grp < 0.1: grp = 0.5
    
    return {
        "sponsor": sponsor,
        "daypart": daypart,
        "region": region,
        "first_broadcast": broadcast_date.strftime("%Y-%m-%dT%H:00:00Z"),
        "impressions": impressions,
        "grp": grp,
        "donated_value": random.randint(5000, 100000),
        "impact_metric": metric["name"],
        "impact_value": random.randint(metric["range"][0], metric["range"][1])
    }

def main():
    input_path = os.path.join("data", "train.csv")
    output_path = os.path.join("data", "psa_campaigns.json")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    print(f"Loading base data from {input_path}...")
    df = pd.read_csv(input_path)
    
    campaigns = []
    
    # For POC, let's take a sample of 100 PSAs to keep the JSON manageable and app fast
    sample_df = df.sample(min(100, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"Generating metadata for {len(sample_df)} PSA campaigns...")
    
    for idx, row in sample_df.iterrows():
        meta = generate_metadata()
        domain = row.get("Domain", "General")
        
        # Build the JSON object
        campaign_obj = {
            "campaign_id": generate_campaign_id(idx + 1),
            "ad_id": generate_ad_id(domain),
            "source_attribution": {
                "sponsor": meta["sponsor"],
                "category": domain if pd.notna(domain) else "Public Interest"
            },
            "creative_versioning": {
                "duration_seconds": random.choice([15, 30, 60]),
                "format": random.choice(["Radio/Audio", "TV/Video", "Digital Display"]),
                "content_english": str(row["English"]) if pd.notna(row["English"]) else "",
                "content_kiswahili": str(row["Kiswahili"]) if "Kiswahili" in row and pd.notna(row["Kiswahili"]) else ""
            },
            "air_traffic_logs": {
                "first_broadcast": meta["first_broadcast"],
                "timezone": "EAT",
                "daypart_classification": meta["daypart"]
            },
            "performance_metrics": {
                "geographic_distribution": meta["region"],
                "estimated_impressions": meta["impressions"],
                "gross_rating_points_grp": meta["grp"],
                "donated_value_metrics_usd": meta["donated_value"]
            },
            "impact_tracking": {
                "primary_metric": meta["impact_metric"],
                "outcome_data": meta["impact_value"]
            }
        }
        campaigns.append(campaign_obj)
        
    final_json = {"campaigns": campaigns}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {len(campaigns)} campaign records.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
