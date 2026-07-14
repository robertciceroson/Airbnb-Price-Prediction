import streamlit as st
import pandas as pd

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏙️", layout="wide")
st.title("🏙️ NYC Airbnb Price Predictor")
st.write("Loading data...")

df = pd.read_csv("listings.csv")
df = df[df["price"] > 0]

st.success(f"Data loaded: {len(df):,} listings")
st.dataframe(df[["neighbourhood_group","neighbourhood","room_type","price"]].head(20))
