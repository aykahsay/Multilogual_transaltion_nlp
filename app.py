import streamlit as st
import pandas as pd
from transformers import pipeline
import os
import json
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PSA Campaign Dashboard",
    page_icon="📢",
    layout="wide"
)

# --- CACHE MODELS ---
@st.cache_resource
def load_translation_pipeline(source_lang, target_lang, model_path="models/psa-en-sw-finetuned"):
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
        return None

# --- LOAD CAMPAIGN DATA ---
@st.cache_data
def load_campaign_data():
    path = os.path.join("data", "psa_campaigns.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.json_normalize(data['campaigns'])
    return pd.DataFrame()

# --- UI LOGIC ---
def main():
    st.title("📢 Public Service Announcement (PSA) Campaign Dashboard")
    st.write("A multilingual system tracking broadcast reach, impact, and translating creatives for under-resourced languages.")
    
    df = load_campaign_data()
    
    tab1, tab2 = st.tabs(["📊 Campaign Analytics", "📝 Creative Translation Studio"])
    
    # --- TAB 1: CAMPAIGN ANALYTICS ---
    with tab1:
        if df.empty:
            st.warning("No campaign data found. Run `generate_psa_metadata.py` first.")
        else:
            st.header("Broadcast Reach & Impact")
            
            # High-Level Metrics
            total_impressions = df['performance_metrics.lifetime_post_total_impressions'].sum()
            total_reach = df['performance_metrics.lifetime_post_total_reach'].sum()
            total_interactions = df['performance_metrics.total_interactions'].sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Lifetime Post Total Impressions", f"{total_impressions:,.0f}")
            col2.metric("Lifetime Post Total Reach", f"{total_reach:,.0f}")
            col3.metric("Total Interactions", f"{total_interactions:,.0f}")
            
            st.divider()
            
            # Interactions breakdown
            col4, col5 = st.columns(2)
            with col4:
                st.subheader("Impressions by Category")
                cat_imp = df.groupby('source_attribution.category')['performance_metrics.lifetime_post_total_impressions'].sum().reset_index()
                cat_imp = cat_imp.set_index('source_attribution.category')
                st.bar_chart(cat_imp)
                
            with col5:
                st.subheader("Interactions Breakdown")
                # Sum likes, comments, shares across all campaigns
                int_breakdown = pd.DataFrame({
                    "Interaction Type": ["Likes", "Comments", "Shares"],
                    "Count": [
                        df['performance_metrics.like_count'].sum(),
                        df['performance_metrics.comment_count'].sum(),
                        df['performance_metrics.share_count'].sum()
                    ]
                }).set_index("Interaction Type")
                st.bar_chart(int_breakdown)
                
            st.divider()
            
            # Detailed Campaign Tracking Log
            st.subheader("Real Digital Tracking Metrics")
            
            # Formatting the dataframe for display
            display_df = df[[
                'campaign_id', 
                'source_attribution.category', 
                'performance_metrics.lifetime_post_total_impressions',
                'performance_metrics.lifetime_post_total_reach',
                'performance_metrics.total_interactions'
            ]].copy()
            
            display_df.columns = ["Campaign ID", "Category", "Impressions", "Reach", "Total Interactions"]
            
            # Search / Filter
            search_category = st.selectbox("Filter by Category:", ["All"] + list(display_df['Category'].unique()))
            if search_category != "All":
                display_df = display_df[display_df['Category'] == search_category]
                
            st.dataframe(display_df, use_container_width=True)

    # --- TAB 2: TRANSLATION ---
    with tab2:
        st.header("Creative Translation Studio")
        st.write("Generate localized creatives for upcoming campaigns.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            direction = st.selectbox("Translation Direction", ["English -> Kiswahili", "Kiswahili -> English"])
            source_lang = direction.split(" -> ")[0]
            target_lang = direction.split(" -> ")[1]
            
            st.info("Uses fine-tuned Helsinki-NLP/opus-mt models.")
            
            # Mock Ad-ID generator for new creatives
            if st.button("Generate Ad-ID for New Creative"):
                st.success(f"New Ad-ID: GEN-PSA-{pd.Timestamp.now().strftime('%M%S')}-15S-SW")
                
        with col2:
            translator = load_translation_pipeline(source_lang, target_lang)
            if translator is None:
                st.error("Model could not be loaded. Please ensure the model files exist.")
            else:
                input_text = st.text_area(f"Enter {source_lang} Ad Script / PSA here:", height=150, 
                                          placeholder="e.g. Ministry of Health: Avoid unnecessary travels to Ebola hotspots")
                
                if st.button("Translate Creative 🔄", type="primary"):
                    if input_text.strip() == "":
                        st.warning("Please enter some text to translate.")
                    else:
                        with st.spinner("Translating..."):
                            result = translator(input_text)[0]['translation_text']
                            
                        st.success("Translation Complete!")
                        st.text_area(f"{target_lang} Localized Script:", value=result, height=150)

if __name__ == "__main__":
    main()
