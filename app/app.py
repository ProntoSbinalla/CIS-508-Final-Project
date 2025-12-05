import joblib
import pandas as pd
import streamlit as st
import urllib.parse

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = "models/xgb_best_model.joblib"
DATA_URL = (
    "https://media.githubusercontent.com/media/ProntoSbinalla/CIS-508-Final-Project/"
    "refs/heads/main/data/rideshare_kaggle.csv"
)

# These must match the features used in your training pipeline
NUMERIC_FEATURES = [
    "hour",
    "distance",
    "precipIntensity",
    "precipProbability",
    "windGust",
    "windBearing",
    "cloudCover",
    "uvIndex",
    "moonPhase",
    "precipIntensityMax",
    "is_weekend",
    "surge_multiplier",
]

CATEGORICAL_FEATURES = [
    "cab_type",
    "name",
    "source",
    "destination",
    "short_summary",
    "day_name",
    "month_name",
]

WEEKEND_SET = {"Saturday", "Sunday"}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_raw_data():
    """Load the original rideshare CSV from GitHub."""
    return pd.read_csv(DATA_URL)


@st.cache_resource
def build_mappings_and_defaults():
    """
    From the raw data, build:
      - cab_type -> list of service names
      - lists of origins and destinations
      - global medians for numeric features
      - default values for short_summary, day_name, month_name
      - distance medians for each (source, destination) pair
    """
    df = load_raw_data()

    # cab_type -> service names
    service_map = {}
    for cab in df["cab_type"].dropna().unique():
        services = (
            df.loc[df["cab_type"] == cab, "name"]
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )
        service_map[cab] = services

    origins = sorted(df["source"].dropna().unique().tolist())
    destinations = sorted(df["destination"].dropna().unique().tolist())

    # Global numeric medians
    numeric_medians = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            numeric_medians[col] = df[col].median()
        else:
            numeric_medians[col] = 0.0

    # Default categorical values (most common)
    if "short_summary" in df.columns and not df["short_summary"].dropna().empty:
        default_short_summary = df["short_summary"].mode().iloc[0]
    else:
        default_short_summary = "Clear throughout the day"

    if "day_name" in df.columns and not df["day_name"].dropna().empty:
        default_day_name = df["day_name"].mode().iloc[0]
    else:
        default_day_name = "Monday"

    if "month_name" in df.columns and not df["month_name"].dropna().empty:
        default_month_name = df["month_name"].mode().iloc[0]
    else:
        default_month_name = "January"

    # Distance median per (source, destination) pair
    distance_lookup = {}
    if "distance" in df.columns:
        grouped = df.groupby(["source", "destination"])["distance"].median()
        for (src, dst), val in grouped.items():
            distance_lookup[(src, dst)] = float(val)

    return (
        service_map,
        origins,
        destinations,
        numeric_medians,
        default_short_summary,
        default_day_name,
        default_month_name,
        distance_lookup,
    )

# ------------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Single Ride Price Predictor",
    page_icon="🚕",
    layout="centered",
)

st.title("🚕 Single Ride Price Predictor")

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}.")
    st.exception(e)
    st.stop()

# Load mappings and defaults
try:
    (
        service_map,
        origins,
        destinations,
        numeric_medians,
        default_short_summary,
        default_day_name,
        default_month_name,
        distance_lookup,
    ) = build_mappings_and_defaults()
except Exception as e:
    st.error("Error loading data from GitHub or building mappings.")
    st.exception(e)
    st.stop()

st.markdown("### Ride details")

# 1. Platform: Uber or Lyft
cab_options = sorted(service_map.keys())
default_cab_index = cab_options.index("Uber") if "Uber" in cab_options else 0
selected_cab = st.selectbox("Platform", cab_options, index=default_cab_index)

# 2. Service name filtered by platform
service_options = service_map.get(selected_cab, [])
selected_service = st.selectbox("Service type", service_options)

# 3. Origin
selected_origin = st.selectbox("Origin", origins)

# 4. Destination (cannot be same as origin)
destination_options = [d for d in destinations if d != selected_origin]
if not destination_options:
    destination_options = destinations
selected_destination = st.selectbox("Destination", destination_options)

if st.button("Predict price"):
    # Build a single-row DataFrame with exactly the model's feature columns
    data = {}

    # Numeric features
    for col in NUMERIC_FEATURES:
        if col == "distance":
            # Use median distance for this origin–destination pair, fallback to global median
            dist_val = distance_lookup.get(
                (selected_origin, selected_destination),
                numeric_medians.get("distance", 0.0),
            )
            data[col] = [dist_val]
        elif col == "is_weekend":
            data[col] = [1 if default_day_name in WEEKEND_SET else 0]
        else:
            # Hour, surge_multiplier, and all other numerics from global median
            data[col] = [numeric_medians.get(col, 0.0)]

    # Categorical features: selected + defaults
    data["cab_type"] = [selected_cab]
    data["name"] = [selected_service]
    data["source"] = [selected_origin]
    data["destination"] = [selected_destination]
    data["short_summary"] = [default_short_summary]
    data["day_name"] = [default_day_name]
    data["month_name"] = [default_month_name]

    # Assemble in correct order
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    input_df = pd.DataFrame(data)[feature_cols]

    try:
        pred = model.predict(input_df)[0]
        st.metric("Predicted ride price", f"${pred:0.2f}")

        # Google Maps route link (opens in new tab)
        origin_q = urllib.parse.quote(f"{selected_origin}, Boston, MA")
        dest_q = urllib.parse.quote(f"{selected_destination}, Boston, MA")
        maps_url = (
            "https://www.google.com/maps/dir/"
            f"?api=1&origin={origin_q}&destination={dest_q}&travelmode=driving"
        )

        st.markdown("### Route on Google Maps")
        st.markdown(f"[Open this route in Google Maps]({maps_url})")

    except Exception as e:
        st.error("Prediction failed. Check that the feature columns match the training setup.")
        st.exception(e)