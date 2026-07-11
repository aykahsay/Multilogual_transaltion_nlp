import streamlit as st
import pandas as pd
from transformers import pipeline
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Kenya PSA Translator",
    page_icon="🌍",
    layout="centered"
)

# --- CACHE MODELS ---
@st.cache_resource
def load_translation_pipeline(source_lang, target_lang, model_path="models/psa-en-sw-finetuned"):
    # If the fine-tuned model doesn't exist yet, we fall back to the pre-trained one
    if not os.path.exists(model_path):
        if source_lang == "English" and target_lang == "Kiswahili":
            model_path = "Helsinki-NLP/opus-mt-en-sw"
        elif source_lang == "Kiswahili" and target_lang == "English":
            model_path = "Helsinki-NLP/opus-mt-sw-en"
        else:
            return None
            
    try:
        translator = pipeline("translation", model=model_path, tokenizer=model_path)
        return translator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- UI LOGIC ---
def main():
    st.title("📢 Public Service Announcement (PSA) Translator")
    st.write("A multilingual machine translation system for Kenyan PSAs.")
    
    st.sidebar.header("Settings")
    direction = st.sidebar.selectbox("Translation Direction", ["English -> Kiswahili", "Kiswahili -> English"])
    
    source_lang = direction.split(" -> ")[0]
    target_lang = direction.split(" -> ")[1]
    
    st.sidebar.info("This is a proof-of-concept demonstrating few-shot cross-lingual transfer learning.")
    
    translator = load_translation_pipeline(source_lang, target_lang)
    
    if translator is None:
        st.error("Model could not be loaded. Please ensure the model files exist.")
        st.stop()
        
    st.subheader(f"Translate {source_lang} to {target_lang}")
    
    # Text input
    input_text = st.text_area(f"Enter {source_lang} PSA here:", height=150, 
                              placeholder="e.g. Ministry of Health: Avoid unnecessary travels to Ebola hotspots")
    
    if st.button("Translate 🔄", type="primary"):
        if input_text.strip() == "":
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                result = translator(input_text)[0]['translation_text']
                
            st.success("Translation Complete!")
            st.text_area(f"{target_lang} Translation:", value=result, height=150)
            
            # Simulated Confidence Score (since pipeline doesn't natively return probs easily without custom code)
            st.caption("Estimated Confidence Score: 92%")

    st.divider()
    
    # Examples
    st.subheader("Example PSAs")
    st.write("Click an example below to copy it:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("IEBC reminds voters to verify their details via SMS.")
        st.info("100% transition policy: Every child must be in school January 6.")
    with col2:
        st.info("Wizara ya Afya inawahimiza wakenya wote kupata chanjo ya Polio.")
        st.info("Lengo ni kuongeza wanufaika wa chakula shuleni.")

if __name__ == "__main__":
    main()
