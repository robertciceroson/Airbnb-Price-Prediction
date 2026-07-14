import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
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

FEATURES = [
    "borough_enc", "neighbourhood_enc", "room_enc",
    "latitude", "longitude",
    "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365",
]

# ── Load data + train model (cached — only runs once per server session) ──────
@st.cache_resource(show_spinner=False)
def load_all():
    df = pd.read_csv("listings.csv")
    drop_cols = ["id", "name", "host_id", "host_profile_id", "host_name",
                 "last_review", "number_of_reviews_ltm", "license"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna(subset=["neighbourhood_group", "neighbourhood", "room_type"])
    df["calculated_host_listings_count"] = df["calculated_host_listings_count"].fillna(1)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = df[df["price"] > 0]
    df = df[df["price"] <= df["price"].quantile(0.995)]

    # Encoding maps (deterministic sort — matches prediction encoding)
    borough_map       = {v: i for i, v in enumerate(sorted(df["neighbourhood_group"].unique()))}
    neighbourhood_map = {v: i for i, v in enumerate(sorted(df["neighbourhood"].unique()))}
    room_map          = {v: i for i, v in enumerate(sorted(df["room_type"].unique()))}

    df["borough_enc"]       = df["neighbourhood_group"].map(borough_map)
    df["neighbourhood_enc"] = df["neighbourhood"].map(neighbourhood_map)
    df["room_enc"]          = df["room_type"].map(room_map)

    neighbourhood_coords = df.groupby("neighbourhood")[["latitude", "longitude"]].mean()
    neighbourhood_prices = df.groupby("neighbourhood")["price"].median().round(0)

    # Train GBR directly from data — avoids pickle/numpy version issues
    df_train = df.dropna(subset=FEATURES)
    X = df_train[FEATURES].values.astype(np.float32)
    y = df_train["price"].values.astype(np.float32)
    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42, n_jobs=1, verbose=-1
    )
    model.fit(X, y)

    return (
        model,
        borough_map, neighbourhood_map, room_map,
        df, neighbourhood_coords, neighbourhood_prices,
    )

# ── CSS ───────────────────────────────────────────────────────────────────────
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

with st.spinner("Loading model — first visit trains on ~21K listings, ~20s…"):
    (
        model,
        borough_map, neighbourhood_map, room_map,
        df, neighbourhood_coords, neighbourhood_prices,
    ) = load_all()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Price Predictor
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔍 Price Predictor")

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

if predict:
    coords = neighbourhood_coords.loc[neighbourhood]
    lat, lng = float(coords["latitude"]), float(coords["longitude"])

    row = np.array([[
        float(borough_map.get(borough, 0)),
        float(neighbourhood_map.get(neighbourhood, 0)),
        float(room_map.get(room_type, 0)),
        lat, lng,
        float(minimum_nights), float(number_of_reviews),
        float(reviews_per_month), float(host_listings), float(availability),
    ]], dtype=np.float32)

    base_pred     = float(model.predict(row)[0])
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Price Map
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🗺️ NYC Airbnb Price Map — Median Nightly Price by Neighbourhood")
st.caption("Dot color: 🟢 green = lower prices → 🔴 red = higher prices. Dot size = number of listings. Hover for details.")

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
        lat="lat", lon="lon",
        color="median_price",
        size="listing_count",
        size_max=28,
        color_continuous_scale="RdYlGn_r",
        hover_name="neighbourhood",
        hover_data={
            "borough": True, "price_label": True, "listing_count": True,
            "median_price": False, "lat": False, "lon": False,
        },
        labels={"borough": "Borough", "price_label": "Median price", "listing_count": "# Listings"},
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 40.7128, "lon": -74.0060},
        height=560,
    )
    fig_map.update_layout(
        coloraxis_colorbar=dict(title="Median<br>price ($)", tickprefix="$", thickness=14, len=0.6),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

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
    "LightGBM model trained on current NYC Airbnb listings (June 2026 — Inside Airbnb) · "
    "Served via LightGBM · "
    "Seasonal adjustments based on NYC tourism patterns · "
    "[GitHub repo](https://github.com/robertciceroson/Airbnb-Price-Prediction)"
)
