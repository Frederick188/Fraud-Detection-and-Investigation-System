import lime.lime_tabular
import streamlit as st
import numpy as np

def lime_explain(model, X_train, X_test, feature_names):
    st.subheader("💡 LIME Explainability")
    try:
        if X_test.shape[0] != 1:
            st.error("Please provide a single transaction (1 row) for LIME")
            return

        def proba_func(x):
            prob = model.predict_proba(x)
            if prob.shape[1] == 1:
                prob = np.hstack([1 - prob, prob])
            return prob

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=["Non-Fraud", "Fraud"],
            mode="classification"
        )

        exp = explainer.explain_instance(
            X_test.iloc[0].values,
            proba_func
        )
        
        lime_features = exp.as_list()

        st.components.v1.html(
            exp.as_html(),
            height=800,
            scrolling=True
        )

        return lime_features

    except Exception as e:
        st.error(f"Error generating LIME explanation: {e}")
