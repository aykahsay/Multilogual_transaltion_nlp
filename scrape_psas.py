import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def scrape_reliefweb_psas(num_pages=50):
    """
    Scrapes disaster alerts and updates for Kenya from ReliefWeb 
    to be used as Public Service Announcements (PSAs).
    """
    base_url = "https://reliefweb.int/updates"
    psas = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    print(f"Starting to scrape {num_pages} pages from ReliefWeb...")
    
    for page in range(num_pages):
        params = {
            "primary_country": 131, # Kenya
            "page": page
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page}. Status: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all articles
            articles = soup.find_all('article')
            for article in articles:
                # Extract title which often acts as a short alert/PSA
                title_tag = article.find('h3')
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    # Simple filter to ensure it's a meaningful sentence
                    if len(title.split()) > 4:
                        psas.append({
                            "Domain": "Disaster/Health",
                            "English": title
                        })
                        
            print(f"Scraped page {page}. Total collected so far: {len(psas)}")
            
            # Respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
            
    return pd.DataFrame(psas)

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Scrape data
    df_scraped = scrape_reliefweb_psas(num_pages=150) # 150 pages * ~10-20 articles = ~1500-3000 sentences
    
    # Deduplicate
    df_scraped = df_scraped.drop_duplicates(subset=["English"])
    print(f"Total unique English PSAs scraped: {len(df_scraped)}")
    
    output_path = os.path.join(output_dir, "scraped_psas_english.csv")
    df_scraped.to_csv(output_path, index=False)
    print(f"Saved scraped data to {output_path}")

if __name__ == "__main__":
    main()
