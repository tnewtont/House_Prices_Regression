import streamlit as st
import pandas as pd
import joblib
from utils.feature_generator import FeatureGenerator
from utils.winwin import Winsorizer


@st.cache_resource
def load_model():
    # adjust this path if needed
    return joblib.load("model/full_pipeline_BCWin.pickle")

pipeline = load_model()

url = "https://imagescdn.homes.com/i2/8mjY9yBmuF-ohhOfLqNTY0M9QQyiz6EzaSbhEDbhvYk/112/bricktowne-ames-townhomes-ames-ia.jpg?p=1"
css = f"""
<style>
.stApp {{
    background-image: url("{url}");
    background-size: cover;
    background-position: center;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.markdown("<h1 style='color: azure;'> Predicting House Prices With Regression </h1>", unsafe_allow_html=True)
upload = st.file_uploader("Upload raw CSV", type="csv")
if upload:
    df = pd.read_csv(upload)
    preds = pipeline.predict(df)
    output   = pd.DataFrame({"Id": df.get("Id", range(len(df))), "Price": preds})
    st.markdown("<h4 style='color: azure;'> Predictions successfully processed!<br> You may download the results using the button below. </h4>", unsafe_allow_html=True)
    st.dataframe(output)
    st.download_button("Download CSV", output.to_csv(index=False).encode(), "predictions.csv")