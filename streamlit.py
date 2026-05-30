import streamlit as st
import joblib
import pandas as pd

# Load saved model
model = joblib.load("toxicity_model.pkl")

# Page settings
st.set_page_config(
    page_title="Comment Toxicity Detection",
    layout="wide"
)

st.title("🛡️ Comment Toxicity Detection System")

# Tabs
tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])

# -------------------------
# Single Comment Prediction
# -------------------------
with tab1:

    st.subheader("Single Comment Prediction")

    comment = st.text_area(
        "Enter Comment",
        height=150
    )

    if st.button("Predict"):

        if comment.strip() == "":
            st.warning("Please enter a comment.")
        else:

            prediction = model.predict([comment])[0]

            probability = model.predict_proba([comment])[0][1]

            st.write(
                f"### Toxicity Score: {probability:.2f}"
            )

            if prediction == 1:
                st.error("⚠ Toxic Comment")
            else:
                st.success("✅ Non-Toxic Comment")

# -------------------------
# Bulk CSV Prediction
# -------------------------
with tab2:

    st.subheader("Bulk Prediction Using CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data")
        st.dataframe(df.head())

        if "comment_text" not in df.columns:

            st.error(
                "CSV must contain a column named 'comment_text'"
            )

        else:

            predictions = model.predict(
                df["comment_text"]
            )

            probabilities = model.predict_proba(
                df["comment_text"]
            )[:, 1]

            df["Prediction"] = predictions
            df["Toxicity_Score"] = probabilities

            st.write("### Prediction Results")
            st.dataframe(df)

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Results",
                data=csv,
                file_name="toxicity_predictions.csv",
                mime="text/csv"
            )
