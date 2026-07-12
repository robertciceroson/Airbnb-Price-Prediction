import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏙️",
    layout="wide",
)
 
# ── Seasonal multipliers by month ─────────────────────────────────────────────
SEASONAL = {
    1:  0.88, 2:  0.90, 3:  0.97, 4:  1.05,
    5:  1.08, 6:  1.18, 7:  1.22, 8:  1.20,
    9:  1.07, 10: 1.05, 11: 0.95, 12: 1.02,
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
    df = pd.read_csv("listings.csv")
    drop_cols = ["id", "name", "host_id", "host_profile_id", "host_name",
                 "last_review", "number_of_reviews_ltm", "license"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna(subset=["neighbourhood_group", "neighbourhood", "room_type"])
    df["calculated_host_listings_count"] = df["calculated_host_listings_count"].fillna(1)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = df[df["price"] > 0]
    df = df[df["price"] <= df["price"].quantile(0.995)]
    df = df[df["minimum_nights"] <= df["minimum_nights"].quantile(0.99)]
 
    le_borough       = LabelEncoder()
    le_neighbourhood = LabelEncoder()
    le_room          = LabelEncoder()
    df["borough_enc"]       = le_borough.fit_transform(df["neighbourhood_group"])
    df["neighbourhood_enc"] = le_neighbourhood.fit_transform(df["neighbourhood"])
    df["room_enc"]          = le_room.fit_transform(df["room_type"])
 
    neighbourhood_coords = df.groupby("neighbourhood")[["latitude", "longitude"]].mean()
    neighbourhood_prices = df.groupby("neighbourhood")["price"].median().round(0)
 
    FEATURES = [
        "borough_enc", "neighbourhood_enc", "room_enc",
        "latitude", "longitude",
        "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count",
        "availability_365",
    ]
    X, y = df[FEATURES], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
 
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    return (
        model, le_borough, le_neighbourhood, le_room,
        df, neighbourhood_coords, neighbourhood_prices,
        FEATURES, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred),
    )
 
# ── Compact CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; max-width: 1200px; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
    div[data-testid="stSlider"] { padding-top: 0; padding-bottom: 0.1rem; }
    h3 { margin-top: 0.3rem !important; margin-bottom: 0.2rem !important; }
    hr { margin: 0.5rem 0 !important; }
    div[data-testid="stMetric"] { padding: 0.3rem 0; }
    div[data-testid="stDateInput"] input {
        background-color: #D4EDDA !important;
        border-color: #28A745 !important;
        color: #155724 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDateInput"] label {
        color: #155724 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] button {
        background-color: #D4EDDA !important;
        border: 1px solid #28A745 !important;
        color: #155724 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #28A745 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>🏙️ NYC Airbnb Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Enter your listing details and check-in date, then click <strong>Predict Price</strong> to get an estimated nightly rate based on current NYC Airbnb listings (June 2026).</p>", unsafe_allow_html=True)
 
with st.spinner("Loading model…"):
    (
        model, le_borough, le_neighbourhood, le_room,
        df, neighbourhood_coords, neighbourhood_prices,
        FEATURES, r2, mae,
    ) = load_and_train()
 
st.divider()
 
# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Price Predictor", "🗺️ Price Map"])
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price Predictor (original)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
 
    col1, col2, col3 = st.columns([1.1, 1.2, 1.0])
 
    with col1:
        st.markdown("**📋 Listing Details**")
        borough = st.selectbox(
            "Borough",
            sorted(df["neighbourhood_group"].unique()),
            index=sorted(df["neighbourhood_group"].unique()).index("Manhattan"),
        )
        neighbourhoods = sorted(df[df["neighbourhood_group"] == borough]["neighbourhood"].unique())
        neighbourhood  = st.selectbox("Neighbourhood", neighbourhoods)
        room_type      = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
 
    with col2:
        st.markdown("**⚙️ Listing Parameters**")
        minimum_nights    = st.slider("Minimum nights required", 1, 30, 2)
        availability      = st.slider("Availability (days / year)", 0, 365, 200)
        number_of_reviews = st.slider("Number of reviews", 0, 300, 20)
        reviews_per_month = st.slider("Reviews per month", 0.0, 10.0, 1.0, step=0.1)
        host_listings     = st.slider("Host total listings", 1, 50, 1)
 
    with col3:
        st.markdown("**📅 Stay Dates**")
        today      = datetime.date.today()
        date_range = st.date_input(
            "Check-in → Check-out",
            value=(today, today + datetime.timedelta(days=minimum_nights)),
            min_value=datetime.date(2024, 1, 1),
            max_value=datetime.date(2027, 12, 31),
        )
 
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            checkin_date, checkout_date = date_range
            num_nights = max((checkout_date - checkin_date).days, 1)
        else:
            checkin_date  = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]
            checkout_date = checkin_date + datetime.timedelta(days=minimum_nights)
            num_nights    = minimum_nights
 
        month         = checkin_date.month
        seasonal_mult = SEASONAL[month]
        season_label  = SEASON_LABEL[month]
        adj_pct       = (seasonal_mult - 1) * 100
        adj_str       = f"+{adj_pct:.0f}%" if adj_pct >= 0 else f"{adj_pct:.0f}%"
 
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Check-in",  checkin_date.strftime("%b %d, %Y"))
        dc2.metric("Check-out", checkout_date.strftime("%b %d, %Y"))
        dc3.metric("Nights",    num_nights)
        st.caption(f"{season_label} · Seasonal adjustment: **{adj_str}** ({checkin_date.strftime('%B')} in NYC)")
 
        st.markdown("")
        predict = st.button("🔍 Predict Price", use_container_width=True)
 
    st.divider()
 
    if predict:
        coords = neighbourhood_coords.loc[neighbourhood]
        lat, lng = coords["latitude"], coords["longitude"]
 
        input_df = pd.DataFrame(
            [[int(le_borough.transform([borough])[0]),
              int(le_neighbourhood.transform([neighbourhood])[0]),
              int(le_room.transform([room_type])[0]),
              lat, lng, minimum_nights, number_of_reviews,
              reviews_per_month, host_listings, availability]],
            columns=FEATURES,
        )
 
        base_pred     = float(model.predict(input_df)[0])
        adjusted_pred = base_pred * seasonal_mult
        total_cost    = adjusted_pred * num_nights
 
        st.markdown(
            f"""<div style="background:#D4EDDA; border:1px solid #28A745; border-radius:6px; padding:4px 14px; display:inline-block;">
            <span style="font-size:0.85rem; font-weight:600; color:#155724;">
            Estimated nightly price: ${adjusted_pred:.0f} &nbsp;·&nbsp;
            Total for {num_nights} night{'s' if num_nights != 1 else ''}: ${total_cost:,.0f}
            </span></div>""",
            unsafe_allow_html=True,
        )
 
        ra, rb, rc = st.columns(3)
        with ra:
            st.metric("Base nightly price", f"${base_pred:.0f}")
        with rb:
            st.metric(f"Adjusted ({checkin_date.strftime('%B')})", f"${adjusted_pred:.0f}",
                      delta=f"{adj_str} seasonal")
        with rc:
            st.metric(f"Total ({num_nights} nights)", f"${total_cost:,.0f}")
 
        median_price = neighbourhood_prices.get(neighbourhood, None)
        if median_price:
            diff      = adjusted_pred - median_price
            direction = "above" if diff > 0 else "below"
            st.markdown(
                f"📍 Median price in **{neighbourhood}**: **${median_price:.0f}/night** — "
                f"your estimate is **${abs(diff):.0f} {direction}** the neighbourhood median."
            )
 
        with st.expander("📊 What drives the prediction?"):
            importance = model.feature_importances_
            labels     = ["Borough", "Neighbourhood", "Room type", "Latitude", "Longitude",
                          "Min nights", "# Reviews", "Reviews/month", "Host listings", "Availability"]
            sorted_idx = np.argsort(importance)
            fig, ax    = plt.subplots(figsize=(6, 4))
            ax.barh([labels[i] for i in sorted_idx], importance[sorted_idx], color="#2E75B6")
            ax.set_xlabel("Feature importance")
            ax.set_title("XGBoost feature importances")
            st.pyplot(fig)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Price Map
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ NYC Airbnb Price Map — Median Nightly Price by Neighbourhood")
    st.caption("Dot color: 🟢 green = lower prices → 🔴 red = higher prices. Dot size = number of listings. Hover for details.")
 
    # Build neighbourhood-level map data
    map_df = (
        df.groupby("neighbourhood")
        .agg(
            lat=("latitude",  "mean"),
            lon=("longitude", "mean"),
            median_price=("price", "median"),
            listing_count=("price", "count"),
            borough=("neighbourhood_group", "first"),
        )
        .reset_index()
    )
    map_df["median_price"] = map_df["median_price"].round(0).astype(int)
    map_df["price_label"]  = map_df["median_price"].apply(lambda p: f"${p:,}/night")
 
    # Borough filter
    all_boroughs = sorted(map_df["borough"].unique())
    selected_boroughs = st.multiselect(
        "Filter by Borough",
        options=all_boroughs,
        default=all_boroughs,
    )
    filtered = map_df[map_df["borough"].isin(selected_boroughs)]
 
    if filtered.empty:
        st.warning("No data — select at least one borough.")
    else:
        fig_map = px.scatter_mapbox(
            filtered,
            lat="lat",
            lon="lon",
            color="median_price",
            size="listing_count",
            size_max=28,
            color_continuous_scale="RdYlGn_r",   # green = cheap, red = expensive
            hover_name="neighbourhood",
            hover_data={
                "borough":       True,
                "price_label":   True,
                "listing_count": True,
                "median_price":  False,
                "lat":           False,
                "lon":           False,
            },
            labels={
                "borough":       "Borough",
                "price_label":   "Median price",
                "listing_count": "# Listings",
            },
            mapbox_style="carto-positron",
            zoom=10,
            center={"lat": 40.7128, "lon": -74.0060},
            height=580,
        )
        fig_map.update_layout(
            coloraxis_colorbar=dict(
                title="Median<br>price ($)",
                tickprefix="$",
                thickness=14,
                len=0.6,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)
 
        # Summary tables
        st.divider()
        c_cheap, c_exp = st.columns(2)
 
        with c_cheap:
            st.markdown("**💚 10 Most Affordable Neighbourhoods**")
            cheap = (
                filtered.nsmallest(10, "median_price")
                [["neighbourhood", "borough", "price_label", "listing_count"]]
                .rename(columns={"neighbourhood": "Neighbourhood", "borough": "Borough",
                                 "price_label": "Median Price", "listing_count": "Listings"})
                .reset_index(drop=True)
            )
            cheap.index += 1
            st.dataframe(cheap, use_container_width=True)
 
        with c_exp:
            st.markdown("**🔴 10 Most Expensive Neighbourhoods**")
            exp = (
                filtered.nlargest(10, "median_price")
                [["neighbourhood", "borough", "price_label", "listing_count"]]
                .rename(columns={"neighbourhood": "Neighbourhood", "borough": "Borough",
                                 "price_label": "Median Price", "listing_count": "Listings"})
                .reset_index(drop=True)
            )
            exp.index += 1
            st.dataframe(exp, use_container_width=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"XGBoost trained on current NYC Airbnb listings (June 2026 — Inside Airbnb) · "
    f"Test R² = {r2:.2f} · MAE = ${mae:.0f} · "
    f"Seasonal adjustments based on NYC tourism patterns · "
    f"[GitHub repo](https://github.com/robertciceroson/Airbnb-Price-Prediction)"
)
