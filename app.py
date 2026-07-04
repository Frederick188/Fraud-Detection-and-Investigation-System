import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from src.preprocess import preprocess, enrich_dataset
from src.model import train_model
from src.explain import lime_explain
from src.agent import FraudAgent

# Load API keys
load_dotenv()
IPGEO_API_KEY = os.getenv("IPGEO_API_KEY")
CURRENCYFREAKS_API_KEY = os.getenv("CURRENCYFREAKS_API_KEY")

st.set_page_config(page_title="Fraud Detection with LIME + APIs", layout="wide")
st.title("💳 Fraud Detection with LIME Explanation + API Enrichment")

# Upload CSV 
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Manual Row Selection 
    start_idx = st.number_input("Start row index", min_value=0, max_value=len(df)-1, value=0)
    end_idx = st.number_input("End row index", min_value=0, max_value=len(df), value=min(200, len(df)))

    if start_idx < end_idx:
        df_selected = df.iloc[start_idx:end_idx].reset_index(drop=True)
        st.write(f"### Selected Rows ({len(df_selected)})")
        st.dataframe(df_selected.head())

        # Enrich Dataset 
        if st.button("🌐 Enrich Dataset with APIs"):
            with st.spinner("Enriching dataset (IP + Currency)..."):
                enriched_df = enrich_dataset(df_selected)

            # Save enriched dataset to session state
            st.session_state["enriched_df"] = enriched_df

            st.success("Dataset enriched successfully!")
            st.subheader("Enriched Dataset Preview")
            st.dataframe(enriched_df.head())

            st.write("### Summary Statistics")
            st.dataframe(enriched_df.describe())

    # Use Enriched Dataset if available
    if "enriched_df" in st.session_state:
        enriched_df = st.session_state["enriched_df"]

        # Train Model 
        if st.button(" Train Model on Enriched Dataset"):
            X, y, scaler = preprocess(enriched_df)
            model, X_train, X_test, y_test, report = train_model(X, y)
            
            # Save preprocessed data
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["scaler"] = scaler


            # Save to session_state
            st.session_state["model"] = model
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["X_columns"] = X_train.columns
            
            if "agent" not in st.session_state:
                st.session_state.agent = FraudAgent()


            st.success("Model trained successfully on enriched dataset!")
            #st.subheader("Classification Report")
            #st.json(report)

        # LIME Explanation
        if "model" in st.session_state:
            st.subheader("LIME Explainability")

            transaction_index = st.number_input(
                "Select transaction index for LIME (from enriched dataset)",
                min_value=0,
                max_value=len(enriched_df) - 1,
                value=0
            )

            if st.button("Generate LIME Explanation"):
                try:
                    # Preprocess dataset
                    X_all = st.session_state["X"]
                    X_single = X_all.iloc[[transaction_index]]

                    # Generate LIME explanation
                    lime_features = lime_explain(
                        st.session_state["model"],
                        st.session_state["X_train"],
                        X_single,
                        st.session_state["X_train"].columns
                    )

                    # Prediction
                    prediction = st.session_state["model"].predict(X_single)[0]

                    # Prediction probabilities
                    probability = st.session_state["model"].predict_proba(X_single)[0]

                    # Save everything for AI investigation
                    st.session_state["investigation"] = {
                        "transaction": enriched_df.iloc[transaction_index].to_dict(),
                        "processed_features": X_single.iloc[0].to_dict(),
                        "raw_ml_features": X_all.iloc[transaction_index].to_dict(),
                        "prediction": int(prediction),
                        "fraud_probability": float(probability[1]),
                        "lime": lime_features
                    }

                except Exception as e:
                    st.error(f"Error generating LIME explanation: {e}")

            
            st.divider()
            st.header("🤖 Fraud Investigation Assistant")

            question = st.text_input(
                "Ask anything about fraud detection"
            )

            if st.button("Ask AI"):

                if not question:
                    st.warning("Please enter a question.")

                elif "investigation" not in st.session_state:
                    st.warning("Please generate the LIME explanation first.")

                else:

                    with st.spinner("Thinking..."):

                        answer = st.session_state.agent.investigate(
                            question,
                            st.session_state["investigation"]
                        )

                    st.markdown("## 🤖 AI Investigation")
                    st.write(answer)