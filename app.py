import streamlit as st
import pandas as pd
from transformers import pipeline
import os
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Multilingual PSA Campaign & Translation Studio",
    page_icon="📢",
    layout="wide"
)

# --- CACHE MODELS ---
@st.cache_resource
def load_translation_pipeline(source_lang, target_lang):
    if source_lang == "English" and target_lang == "Kiswahili":
        model_path = "models/psa-en-sw-finetuned" if os.path.exists("models/psa-en-sw-finetuned") else "Helsinki-NLP/opus-mt-en-sw"
    elif source_lang == "Kiswahili" and target_lang == "English":
        model_path = "Helsinki-NLP/opus-mt-sw-en"
    elif source_lang == "English" and target_lang == "Ekegusii":
        model_path = "models/nllb-en-guz" if os.path.exists("models/nllb-en-guz") else "facebook/nllb-200-distilled-600M"
    else:
        return None
            
    try:
        if "nllb" in model_path:
            translator = pipeline("translation", model=model_path, tokenizer=model_path, src_lang="eng_Latn", tgt_lang="guz_Latn")
        else:
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
    st.title("📢 Public Service Announcement (PSA) Multilingual Translation Studio")
    st.write("A multilingual NLP system supporting Kiswahili and Ekegusii (Gusii) translation for health, safety, and public policy notices in Kenya.")
    
    df = load_campaign_data()
    
    tab1, tab2, tab3 = st.tabs(["📊 Campaign Analytics", "📝 Creative Translation Studio", "📚 Trilingual Parallel Corpus"])
    
    # --- TAB 1: CAMPAIGN ANALYTICS ---
    with tab1:
        if df.empty:
            st.info("No active campaign logs found. Run dataset scripts to view live tracking.")
        else:
            st.header("Broadcast Reach & Impact")
            
            total_impressions = df['performance_metrics.lifetime_post_total_impressions'].sum()
            total_reach = df['performance_metrics.lifetime_post_total_reach'].sum()
            total_interactions = df['performance_metrics.total_interactions'].sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Lifetime Total Impressions", f"{total_impressions:,.0f}")
            col2.metric("Lifetime Total Reach", f"{total_reach:,.0f}")
            col3.metric("Total Interactions", f"{total_interactions:,.0f}")
            
            st.divider()
            
            col4, col5 = st.columns(2)
            with col4:
                st.subheader("Impressions by Category")
                cat_imp = df.groupby('source_attribution.category')['performance_metrics.lifetime_post_total_impressions'].sum().reset_index()
                cat_imp = cat_imp.set_index('source_attribution.category')
                st.bar_chart(cat_imp)
                
            with col5:
                st.subheader("Interactions Breakdown")
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
            
            st.subheader("Real Digital Tracking Metrics")
            display_df = df[[
                'campaign_id', 
                'source_attribution.category', 
                'performance_metrics.lifetime_post_total_impressions',
                'performance_metrics.lifetime_post_total_reach',
                'performance_metrics.total_interactions'
            ]].copy()
            
            display_df.columns = ["Campaign ID", "Category", "Impressions", "Reach", "Total Interactions"]
            
            search_category = st.selectbox("Filter by Category:", ["All"] + list(display_df['Category'].unique()))
            if search_category != "All":
                display_df = display_df[display_df['Category'] == search_category]
                
            st.dataframe(display_df, use_container_width=True)

    # --- TAB 2: TRANSLATION ---
    with tab2:
        st.header("Creative Translation Studio")
        st.write("Generate localized public advisories in **Kiswahili** and **Ekegusii (Gusii)**.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            direction = st.selectbox(
                "Translation Direction", 
                ["English -> Kiswahili", "English -> Ekegusii", "Kiswahili -> English"]
            )
            source_lang = direction.split(" -> ")[0]
            target_lang = direction.split(" -> ")[1]
            
            st.info(f"Targeting: **{target_lang}**")
            
            if st.button("Generate Unique Ad-ID"):
                st.success(f"New Ad-ID: GEN-PSA-{pd.Timestamp.now().strftime('%M%S')}-KEN")
                
        with col2:
            translator = load_translation_pipeline(source_lang, target_lang)
            if translator is None:
                st.warning("Translation model loading... (Ensure model checkpoint is active)")
            else:
                input_text = st.text_area(f"Enter {source_lang} Ad Script / PSA here:", height=150, 
                                          placeholder="e.g. Please wash your hands with soap and clean water to prevent disease.")
                
                if st.button("Translate Creative 🔄", type="primary"):
                    if input_text.strip() == "":
                        st.warning("Please enter some text to translate.")
                    else:
                        with st.spinner(f"Translating to {target_lang}..."):
                            result = translator(input_text)[0]['translation_text']
                            
                        st.success("Translation Complete!")
                        st.text_area(f"{target_lang} Localized Script:", value=result, height=150)

    # --- TAB 3: TRILINGUAL DATASET BROWSER ---
    with tab3:
        st.header("Trilingual Parallel Corpus (English | Kiswahili | Ekegusii)")
        train_path = os.path.join("data", "train.csv")
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            domain_filter = st.selectbox("Select Domain:", ["All"] + list(df_train['Domain'].unique()))
            if domain_filter != "All":
                df_display = df_train[df_train['Domain'] == domain_filter]
            else:
                df_display = df_train
            st.dataframe(df_display, use_container_width=True)
            st.caption(f"Showing {len(df_display)} parallel sentences in {domain_filter} domain.")

if __name__ == "__main__":
    main()
