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
            total_impressions = df['performance_metrics.estimated_impressions'].sum()
            avg_grp = df['performance_metrics.gross_rating_points_grp'].mean()
            total_donated = df['performance_metrics.donated_value_metrics_usd'].sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Estimated Impressions", f"{total_impressions:,.0f}")
            col2.metric("Average Campaign GRP", f"{avg_grp:.1f}")
            col3.metric("Total Donated Media Value", f"${total_donated:,.0f}")
            
            st.divider()
            
            # Geographic Distribution & Dayparts
            col4, col5 = st.columns(2)
            with col4:
                st.subheader("Impressions by Geographic Region")
                region_imp = df.groupby('performance_metrics.geographic_distribution')['performance_metrics.estimated_impressions'].sum().reset_index()
                region_imp = region_imp.set_index('performance_metrics.geographic_distribution')
                st.bar_chart(region_imp)
                
            with col5:
                st.subheader("Campaigns by Daypart")
                daypart_counts = df['air_traffic_logs.daypart_classification'].value_counts()
                st.bar_chart(daypart_counts)
                
            st.divider()
            
            # Detailed Campaign Tracking Log
            st.subheader("Air Traffic & Performance Logs")
            
            # Formatting the dataframe for display
            display_df = df[[
                'ad_id', 
                'source_attribution.sponsor', 
                'air_traffic_logs.first_broadcast',
                'performance_metrics.geographic_distribution',
                'performance_metrics.gross_rating_points_grp',
                'impact_tracking.primary_metric',
                'impact_tracking.outcome_data'
            ]].copy()
            
            display_df.columns = ["Ad-ID", "Sponsor", "Air Date", "DMA / Region", "GRPs", "Impact Metric", "Outcome Value"]
            
            # Search / Filter
            search_sponsor = st.selectbox("Filter by Sponsor:", ["All"] + list(display_df['Sponsor'].unique()))
            if search_sponsor != "All":
                display_df = display_df[display_df['Sponsor'] == search_sponsor]
                
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
