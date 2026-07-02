import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏙️",
    layout="centered",
)

# ── Load data, preprocess, train — cached for the session ────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    # Load CSV (in repo root)
    df = pd.read_csv("AB_NYC_2019.csv")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    df = df.drop(columns=["id", "name", "host_id", "host_name", "last_review"])
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = df[df["price"] > 0]
    price_cap = df["price"].quantile(0.995)          # ~$800
    df = df[df["price"] <= price_cap]
    min_nights_cap = df["minimum_nights"].quantile(0.99)
    df = df[df["minimum_nights"] <= min_nights_cap]

    # ── Encode categoricals ───────────────────────────────────────────────────
    le_borough       = LabelEncoder()
    le_neighbourhood = LabelEncoder()
    le_room          = LabelEncoder()

    df["borough_enc"]       = le_borough.fit_transform(df["neighbourhood_group"])
    df["neighbourhood_enc"] = le_neighbourhood.fit_transform(df["neighbourhood"])
    df["room_enc"]          = le_room.fit_transform(df["room_type"])

    # ── Neighbourhood → average coords (for UX — no lat/lng sliders) ─────────
    neighbourhood_coords = (
        df.groupby("neighbourhood")[["latitude", "longitude"]].mean()
    )
    # Median prices per neighbourhood (for context display)
    neighbourhood_prices = df.groupby("neighbourhood")["price"].median().round(0)

    # ── Feature matrix ────────────────────────────────────────────────────────
    FEATURES = [
        "borough_enc", "neighbourhood_enc", "room_enc",
        "latitude", "longitude",
        "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count",
        "availability_365",
    ]
    X = df[FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Test-set metrics
    y_pred  = model.predict(X_test)
    r2      = r2_score(y_test, y_pred)
    mae     = mean_absolute_error(y_test, y_pred)

    return (
        model,
        le_borough, le_neighbourhood, le_room,
        df,
        neighbourhood_coords,
        neighbourhood_prices,
        FEATURES,
        r2, mae,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏙️ NYC Airbnb Price Predictor")
st.markdown(
    "Enter your listing details below and click **Predict Price** "
    "to get an estimated nightly rate based on 49,000 NYC Airbnb listings (2019)."
)
st.markdown("---")

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model — training XGBoost on 49K listings (~20 s on first launch)…"):
    (
        model,
        le_borough, le_neighbourhood, le_room,
        df,
        neighbourhood_coords,
        neighbourhood_prices,
        FEATURES,
        r2, mae,
    ) = load_and_train()

# ── Listing details form ──────────────────────────────────────────────────────
st.subheader("📋 Listing Details")

col1, col2 = st.columns(2)

with col1:
    borough = st.selectbox(
        "Borough",
        sorted(df["neighbourhood_group"].unique()),
        index=sorted(df["neighbourhood_group"].unique()).index("Manhattan"),
    )
    neighbourhoods = sorted(
        df[df["neighbourhood_group"] == borough]["neighbourhood"].unique()
    )
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room"],
    )

with col2:
    minimum_nights    = st.slider("Minimum nights required", 1, 30, 2)
    availability      = st.slider("Availability (days / year)", 0, 365, 200)
    number_of_reviews = st.slider("Number of reviews", 0, 300, 20)

col3, col4 = st.columns(2)
with col3:
    reviews_per_month = st.slider("Reviews per month", 0.0, 10.0, 1.0, step=0.1)
with col4:
    host_listings = st.slider("Host total listings", 1, 50, 1)

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True):

    # Look up average coords for selected neighbourhood
    coords = neighbourhood_coords.loc[neighbourhood]
    lat, lng = coords["latitude"], coords["longitude"]

    # Encode inputs
    borough_enc       = int(le_borough.transform([borough])[0])
    neighbourhood_enc = int(le_neighbourhood.transform([neighbourhood])[0])
    room_enc          = int(le_room.transform([room_type])[0])

    input_df = pd.DataFrame(
        [[borough_enc, neighbourhood_enc, room_enc,
          lat, lng,
          minimum_nights, number_of_reviews,
          reviews_per_month, host_listings,
          availability]],
        columns=FEATURES,
    )

    pred = float(model.predict(input_df)[0])

    # ── Result ─────────────────────────────────────────────────────────────
    st.success(f"### Estimated nightly price: **${pred:.0f}**")

    # Context: neighbourhood median
    median_price = neighbourhood_prices.get(neighbourhood, None)
    if median_price:
        diff = pred - median_price
        direction = "above" if diff > 0 else "below"
        st.markdown(
            f"📍 Median price in **{neighbourhood}**: **${median_price:.0f}/night**  "
            f"— your estimate is **${abs(diff):.0f} {direction}** the neighbourhood median."
        )

    # ── Feature importance chart ────────────────────────────────────────────
    with st.expander("📊 What drives the prediction?"):
        importance = model.feature_importances_
        labels = [
            "Borough", "Neighbourhood", "Room type",
            "Latitude", "Longitude",
            "Min nights", "# Reviews",
            "Reviews/month", "Host listings",
            "Availability",
        ]
        sorted_idx = np.argsort(importance)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            [labels[i] for i in sorted_idx],
            importance[sorted_idx],
            color="#2E75B6",
        )
        ax.set_xlabel("Feature importance")
        ax.set_title("XGBoost feature importances")
        st.pyplot(fig)

# ── Model metrics footer ──────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"XGBoost trained on 49K NYC Airbnb listings · "
    f"Test R² = {r2:.2f} · MAE = ${mae:.0f} · "
    f"[GitHub repo](https://github.com/robertciceroson/Airbnb-Price-Prediction)"
)
