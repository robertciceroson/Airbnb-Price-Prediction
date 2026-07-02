import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import datetime
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏙️",
    layout="centered",
)
 
# ── Seasonal multipliers by month ─────────────────────────────────────────────
SEASONAL = {
    1:  0.88,   # January   — off-season
    2:  0.90,   # February  — off-season
    3:  0.97,   # March     — shoulder
    4:  1.05,   # April     — spring uptick
    5:  1.08,   # May       — spring peak
    6:  1.18,   # June      — summer peak
    7:  1.22,   # July      — summer peak
    8:  1.20,   # August    — summer peak
    9:  1.07,   # September — fall shoulder
    10: 1.05,   # October   — fall shoulder
    11: 0.95,   # November  — off-season
    12: 1.02,   # December  — holidays
}
 
SEASON_LABEL = {
    1: "❄️ Off-season", 2: "❄️ Off-season",
    3: "🌸 Shoulder",   4: "🌸 Spring",     5: "🌸 Spring peak",
    6: "☀️ Summer peak", 7: "☀️ Summer peak", 8: "☀️ Summer peak",
    9: "🍂 Fall shoulder", 10: "🍂 Fall shoulder",
    11: "❄️ Off-season", 12: "🎄 Holiday",
}
 
# ── Load data, preprocess, train — cached for the session ────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    df = pd.read_csv("AB_NYC_2019.csv")
 
    # Preprocessing
    df = df.drop(columns=["id", "name", "host_id", "host_name", "last_review"])
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = df[df["price"] > 0]
    price_cap = df["price"].quantile(0.995)
    df = df[df["price"] <= price_cap]
    min_nights_cap = df["minimum_nights"].quantile(0.99)
    df = df[df["minimum_nights"] <= min_nights_cap]
 
    # Encode categoricals
    le_borough       = LabelEncoder()
    le_neighbourhood = LabelEncoder()
    le_room          = LabelEncoder()
 
    df["borough_enc"]       = le_borough.fit_transform(df["neighbourhood_group"])
    df["neighbourhood_enc"] = le_neighbourhood.fit_transform(df["neighbourhood"])
    df["room_enc"]          = le_room.fit_transform(df["room_type"])
 
    # Neighbourhood averages
    neighbourhood_coords = df.groupby("neighbourhood")[["latitude", "longitude"]].mean()
    neighbourhood_prices = df.groupby("neighbourhood")["price"].median().round(0)
 
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
 
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
 
    return (
        model, le_borough, le_neighbourhood, le_room,
        df, neighbourhood_coords, neighbourhood_prices,
        FEATURES, r2, mae,
    )
 
 
# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏙️ NYC Airbnb Price Predictor")
st.markdown(
    "Enter your listing details and check-in date, then click **Predict Price** "
    "to get an estimated nightly rate based on 49,000 NYC Airbnb listings (2019)."
)
st.markdown("---")
 
with st.spinner("Loading model — training XGBoost on 49K listings (~20 s on first launch)…"):
    (
        model, le_borough, le_neighbourhood, le_room,
        df, neighbourhood_coords, neighbourhood_prices,
        FEATURES, r2, mae,
    ) = load_and_train()
 
# ── Form ──────────────────────────────────────────────────────────────────────
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
 
# ── Date range picker ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 Stay Dates")
today = datetime.date.today()
date_range = st.date_input(
    "Select check-in and check-out dates",
    value=(today, today + datetime.timedelta(days=minimum_nights)),
    min_value=datetime.date(2024, 1, 1),
    max_value=datetime.date(2027, 12, 31),
)
 
# Handle partial selection (user clicked only one date)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    checkin_date, checkout_date = date_range
    num_nights = (checkout_date - checkin_date).days
else:
    checkin_date  = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]
    checkout_date = checkin_date + datetime.timedelta(days=minimum_nights)
    num_nights    = minimum_nights
 
num_nights = max(num_nights, 1)  # guard against same-day selection
 
month = checkin_date.month
seasonal_mult = SEASONAL[month]
season_label  = SEASON_LABEL[month]
adj_pct = (seasonal_mult - 1) * 100
adj_str = f"+{adj_pct:.0f}%" if adj_pct >= 0 else f"{adj_pct:.0f}%"
 
col_d1, col_d2, col_d3 = st.columns(3)
col_d1.metric("Check-in",  checkin_date.strftime("%b %d, %Y"))
col_d2.metric("Check-out", checkout_date.strftime("%b %d, %Y"))
col_d3.metric("Nights",    num_nights)
 
st.caption(
    f"{season_label} · Seasonal adjustment: **{adj_str}** "
    f"({checkin_date.strftime('%B')} is "
    f"{'peak' if seasonal_mult > 1.1 else 'shoulder' if seasonal_mult > 0.95 else 'off-season'} in NYC)"
)
 
st.markdown("---")
 
# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True):
 
    coords = neighbourhood_coords.loc[neighbourhood]
    lat, lng = coords["latitude"], coords["longitude"]
 
    borough_enc       = int(le_borough.transform([borough])[0])
    neighbourhood_enc = int(le_neighbourhood.transform([neighbourhood])[0])
    room_enc          = int(le_room.transform([room_type])[0])
 
    input_df = pd.DataFrame(
        [[borough_enc, neighbourhood_enc, room_enc,
          lat, lng, minimum_nights, number_of_reviews,
          reviews_per_month, host_listings, availability]],
        columns=FEATURES,
    )
 
    base_pred     = float(model.predict(input_df)[0])
    adjusted_pred = base_pred * seasonal_mult
    total_cost    = adjusted_pred * num_nights
 
    # Results
    st.success(
        f"### Estimated nightly price: **${adjusted_pred:.0f}** · "
        f"Total for {num_nights} night{'s' if num_nights != 1 else ''}: **${total_cost:,.0f}**"
    )
 
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Base nightly price", f"${base_pred:.0f}")
    with col_b:
        st.metric(
            f"Adjusted ({checkin_date.strftime('%B')})",
            f"${adjusted_pred:.0f}",
            delta=f"{adj_str} seasonal",
        )
    with col_c:
        st.metric(f"Total ({num_nights} nights)", f"${total_cost:,.0f}")
 
    # Neighbourhood median context
    median_price = neighbourhood_prices.get(neighbourhood, None)
    if median_price:
        diff = adjusted_pred - median_price
        direction = "above" if diff > 0 else "below"
        st.markdown(
            f"📍 Median price in **{neighbourhood}**: **${median_price:.0f}/night** — "
            f"your estimate is **${abs(diff):.0f} {direction}** the neighbourhood median."
        )
 
    # Feature importance
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
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"XGBoost trained on 49K NYC Airbnb listings (2019) · "
    f"Test R² = {r2:.2f} · MAE = ${mae:.0f} · "
    f"Seasonal adjustments based on NYC tourism patterns · "
    f"[GitHub repo](https://github.com/robertciceroson/Airbnb-Price-Prediction)"
)
