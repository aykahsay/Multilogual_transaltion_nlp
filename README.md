# Multilingual Machine Translation for Public Service Announcements (PSAs)

This repository contains the codebase for a proof-of-concept multilingual machine translation (MT) system tailored for Public Service Announcements (PSAs) in Kenya. The project focuses initially on English to Kiswahili translations and features a web application deployed using Streamlit.

## Repository Contents

- `scrape_psas.py`: Web scraper using BeautifulSoup to fetch disaster/health PSAs from ReliefWeb.
- `translate_psas.py`: Script using `deep-translator` to generate Kiswahili pairs from scraped English PSAs.
- `preprocess.py`: Cleans and splits the combined scraped data and the existing PSA dataset into train, validation, and test splits.
- `train.py`: Fine-tunes the lightweight `Helsinki-NLP/opus-mt-en-sw` sequence-to-sequence model using the Hugging Face `transformers` library on the curated PSA dataset.
- `evaluate.py`: Computes objective metrics (BLEU, chrF) using `sacrebleu` on the held-out test set to evaluate translation quality.
- `app.py`: A Streamlit web application that serves as the final deployable digital public good. Users can input a PSA and receive the translation.
- `requirements.txt`: Python dependencies required to run the pipeline and web application.

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Pipeline (Optional if data already exists):**
   ```bash
   python scrape_psas.py
   python translate_psas.py
   python preprocess.py
   ```

3. **Train the Model:**
   ```bash
   python train.py
   ```

4. **Evaluate:**
   ```bash
   python evaluate.py
   ```

5. **Run the Web Application:**
   ```bash
   streamlit run app.py
   ```

## License
MIT License
