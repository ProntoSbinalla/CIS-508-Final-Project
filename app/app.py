import os
import urllib.parse

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = "models/xgb_best_model.joblib"
DATA_URL = (
    "https://media.githubusercontent.com/media/ProntoSbinalla/CIS-508-Final-Project/"
    "refs/heads/main/data/rideshare_kaggle.csv"
)

# Optional: path to a saved feature importance image
FEATURE_IMPORTANCE_IMG = "images/xgb_feature_importances_top20.png"

# These must match the features used in the training pipeline
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
]

CATEGORICAL_FEATURES = [
    "cab_type",
    "name",
    "source",
    "destination",
    "short_summary",
    "day_name",
    "month_name",
    "is_weekend",
    "is_peak_hour",
]

DAY_OPTIONS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

WEEKEND_SET = {"Saturday", "Sunday"}
PEAK_HOURS = [7, 8, 9, 16, 17, 18, 19, 20]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained XGBoost pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_raw_data():
    """Load the original rideshare CSV from GitHub."""
    return pd.read_csv(DATA_URL)


@st.cache_resource
def build_mappings_and_defaults():
    """
    Build helper structures from the raw data:
      - cab_type -> list of service names
      - sorted lists of origins and destinations
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

    # Global numeric medians for the features the model actually uses
    numeric_medians = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            numeric_medians[col] = float(df[col].median())
        else:
            numeric_medians[col] = 0.0

    # Default categorical values (most common in the historical dataset)
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
# Streamlit app layout
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Rideshare Price Prediction",
    page_icon="🚕",
    layout="centered",
)

st.title("🚕 Rideshare Price Prediction App")

st.markdown(
    """
Use this app to estimate the price of a single Uber or Lyft ride in Boston.

The model was trained on the Kaggle **Uber and Lyft Dataset Boston, MA** and
deploys the final XGBoost regressor from the CIS 508 project.
"""
)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}.")
    st.exception(e)
    st.stop()

# Load mappings and defaults from the data
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

st.markdown("### Ride setup")

# 1. Platform: Uber or Lyft
cab_options = sorted(service_map.keys())
if not cab_options:
    st.error("No platforms found in the source data.")
    st.stop()

default_cab_index = cab_options.index("Uber") if "Uber" in cab_options else 0
selected_cab = st.selectbox("Platform", cab_options, index=default_cab_index)

# 2. Service name, filtered by platform
service_options = service_map.get(selected_cab, [])
if not service_options:
    st.error("No services found for the selected platform.")
    st.stop()

selected_service = st.selectbox("Service type", service_options)

# 3. Origin
selected_origin = st.selectbox("Origin", origins)

# 4. Destination (not equal to origin if possible)
destination_options = [d for d in destinations if d != selected_origin]
if not destination_options:
    destination_options = destinations

selected_destination = st.selectbox("Destination", destination_options)

# 5. Time of day and day of week
col1, col2 = st.columns(2)
with col1:
    selected_hour = st.slider(
        "Pickup hour (24 hour clock)",
        min_value=0,
        max_value=23,
        value=9,
        help="Typical peak hours are 7–9 and 16–20.",
    )
with col2:
    default_day_index = (
        DAY_OPTIONS.index(default_day_name)
        if default_day_name in DAY_OPTIONS
        else 0
    )
    selected_day_name = st.selectbox(
        "Day of week",
        DAY_OPTIONS,
        index=default_day_index,
    )

st.caption(
    "Weather related features are kept at typical median values from the historical data "
    "so you can focus on how platform, route and time affect price."
)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("Predict ride price"):
    # Build a single row DataFrame with exactly the model's feature columns
    data = {}

    # Numeric features
    for col in NUMERIC_FEATURES:
        if col == "hour":
            data[col] = [selected_hour]
        elif col == "distance":
            # Use median distance for this origin and destination, fall back to global median
            dist_val = distance_lookup.get(
                (selected_origin, selected_destination),
                numeric_medians.get("distance", 0.0),
            )
            data[col] = [dist_val]
        else:
            data[col] = [numeric_medians.get(col, 0.0)]

    # Categorical features
    data["cab_type"] = [selected_cab]
    data["name"] = [selected_service]
    data["source"] = [selected_origin]
    data["destination"] = [selected_destination]
    data["short_summary"] = [default_short_summary]
    data["day_name"] = [selected_day_name]
    data["month_name"] = [default_month_name]

    # Time flags used by the model
    is_weekend_val = "yes" if selected_day_name in WEEKEND_SET else "no"
    is_peak_hour_val = "yes" if selected_hour in PEAK_HOURS else "no"
    data["is_weekend"] = [is_weekend_val]
    data["is_peak_hour"] = [is_peak_hour_val]

    # Assemble in the correct order
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    input_df = pd.DataFrame(data)[feature_cols]

    try:
        pred = float(model.predict(input_df)[0])
        st.metric("Predicted ride price", f"${pred:0.2f}")

        # Optional debug view for instructors
        with st.expander("Show model inputs"):
            st.write(input_df)

        # Google Maps route link
        origin_q = urllib.parse.quote(f"{selected_origin}, Boston, MA")
        dest_q = urllib.parse.quote(f"{selected_destination}, Boston, MA")
        maps_url = (
            "https://www.google.com/maps/dir/"
            f"?api=1&origin={origin_q}&destination={dest_q}&travelmode=driving"
        )

        st.markdown("### Route on Google Maps")
        st.markdown(f"[Open this route in Google Maps]({maps_url})")

    except Exception as e:
        st.error(
            "Prediction failed. Check that the feature columns here match the "
            "training setup and that the saved model is the full pipeline."
        )
        st.exception(e)

# ------------------------------------------------------------------
# Model explanation section
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("### About this model")

st.markdown(
    """
This app deploys the final **XGBoost Regressor** from the project.

In practice, the model is driven mainly by a small set of features:

- **Service type** (for example Lux Black XL, Black SUV, UberX)
- **Trip distance**
- **Pickup neighborhood** (source)
- **Dropoff neighborhood** (destination)
- **Platform** (Uber versus Lyft)
- **Peak hour flag** (whether the pickup is in a rush hour window)
- **Weekend flag** (weekday versus Saturday or Sunday)

Additional inputs such as weather conditions are still passed to the model
for completeness, but their feature importance is much smaller compared to
the factors above.
"""
)

if os.path.exists(FEATURE_IMPORTANCE_IMG):
    with st.expander("Show feature importance chart"):
        st.image(
            FEATURE_IMPORTANCE_IMG,
            caption="Top 20 feature importances for the XGBoost model",
        )