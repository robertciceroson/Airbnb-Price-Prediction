
import pandas as pd
import plotly.express as px
import datetime

SEASONAL = {
    1: 0.88, 2: 0.90, 3: 0.97, 4: 1.05, 5: 1.08,
    6: 1.18, 7: 1.22, 8: 1.20, 9: 1.07, 10: 1.05,
    11: 0.95, 12: 1.02
}
SEASON_LABEL = {
    1: "❄️ Off-season", 2: "❄️ Off-season", 3: "🌸 Shoulder",
    4: "🌸 Spring", 5: "🌸 Spring peak", 6: "☀️ Summer peak",
    7: "☀️ Summer peak", 8: "☀️ Summer peak", 9: "🍂 Fall shoulder",
    10: "🍂 Fall shoulder", 11: "❄️ Off-season", 12: "🎄 Holiday"
}

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏙️", layout="wide")


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
    neighbourhood_coords = df.groupby("neighbourhood")[["latitude", "longitude"]].mean()
    neighbourhood_prices = df.groupby("neighbourhood")["price"].median().round(0)
    neigh_room_median = df.groupby(["neighbourhood", "room_type"])["price"].median()
    global_median = float(df["price"].median())
    return (neigh_room_median, global_median, df, neighbourhood_coords, neighbourhood_prices)


with st.spinner("Loading NYC Airbnb data..."):
    neigh_room_median, global_median, df, neighbourhood_coords, neighbourhood_prices = load_all()

st.title("🏙️ NYC Airbnb Price Predictor")
st.caption("Market median pricing model — NYC Airbnb listings (June 2026 — Inside Airbnb)")

st.divider()

# --- Price Predictor ---
st.subheader("💰 Predict Your Price")

col1, col2, col3 = st.columns(3)

boroughs = sorted(df["neighbourhood_group"].unique())
with col1:
    borough = st.selectbox("Borough", boroughs,
                           index=boroughs.index("Manhattan") if "Manhattan" in boroughs else 0)

neighbourhoods = sorted(df[df["neighbourhood_group"] == borough]["neighbourhood"].unique())
with col2:
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)

room_types = sorted(df["room_type"].unique())
with col3:
    room_type = st.selectbox("Room Type", room_types)

col4, col5 = st.columns(2)
with col4:
    checkin = st.date_input("Check-in Date", value=datetime.date.today())
with col5:
    nights = st.number_input("Number of Nights", min_value=1, max_value=30, value=3)

# Prediction
base_price = neigh_room_median.get(
    (neighbourhood, room_type),
    neighbourhood_prices.get(neighbourhood, global_median)
)
month = checkin.month
seasonal_mult = SEASONAL[month]
predicted_price = base_price * seasonal_mult
total = predicted_price * nights
season_label = SEASON_LABEL[month]

st.divider()

r1, r2, r3, r4 = st.columns(4)
r1.metric("Estimated Nightly Rate", f"${predicted_price:,.0f}")
r2.metric("Season", season_label)
r3.metric("Seasonal Adjustment", f"{(seasonal_mult - 1) * 100:+.0f}%")
r4.metric(f"Total ({nights} nights)", f"${total:,.0f}")

st.divider()

# --- NYC Map ---
st.subheader("🗺️ NYC Airbnb Price Map")

borough_filter = st.multiselect(
    "Filter by Borough",
    options=boroughs,
    default=boroughs
)

map_df = df[df["neighbourhood_group"].isin(borough_filter)].copy()
map_df = map_df[["neighbourhood_group", "neighbourhood", "room_type",
                  "price", "latitude", "longitude"]].dropna()

if len(map_df) > 5000:
    map_df = map_df.sample(5000, random_state=42)

fig = px.scatter_mapbox(
    map_df,
    lat="latitude",
    lon="longitude",
    color="price",
    color_continuous_scale="RdYlGn_r",
    range_color=[0, df["price"].quantile(0.9)],
    hover_data={"neighbourhood": True, "room_type": True, "price": True,
                "latitude": False, "longitude": False},
    zoom=10,
    center={"lat": 40.7128, "lon": -74.0060},
    height=500,
    mapbox_style="carto-positron",
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Neighbourhood Price Table ---
st.subheader("📊 Median Prices by Neighbourhood")

price_table = (
    df[df["neighbourhood_group"].isin(borough_filter)]
    .groupby(["neighbourhood_group", "neighbourhood", "room_type"])["price"]
    .median()
    .round(0)
    .reset_index()
    .rename(columns={
        "neighbourhood_group": "Borough",
        "neighbourhood": "Neighbourhood",
        "room_type": "Room Type",
        "price": "Median Price ($)"
    })
    .sort_values(["Borough", "Neighbourhood"])
)

st.dataframe(price_table, use_container_width=True, hide_index=True)

st.caption("Market median pricing model — NYC Airbnb listings (June 2026 — Inside Airbnb)")
