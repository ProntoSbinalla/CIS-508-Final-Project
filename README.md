# CIS 508 Final Project - Rideshare Price Prediction

End to end machine learning project for CIS 508 (Machine Learning in Business) that predicts Uber and Lyft ride prices in Boston, MA and deploys the final model as an interactive Streamlit web app.

The project follows the course final project guideline from problem framing through deployment and documentation. 

---

## 1. Business Problem and Objective

Rideshare platforms like Uber and Lyft use dynamic pricing that depends on:

* Origin and destination
* Time of day and day of week
* Weather conditions
* Service type (UberX vs UberBlack vs Lyft, etc.)

For passengers and planners, it is useful to estimate the likely ride price before opening the app. For the business, a price prediction model supports:

* Revenue forecasting for different time and location patterns
* What if analysis on weather or demand shocks
* Pricing strategy experiments at specific locations or times

**Goal**
Build a regression model that predicts the **ride price** given trip details and contextual features, and deploy it as a web app that:

* Accepts user inputs (service, route, time, weather)
* Returns an estimated price for a single ride
* Visualizes which features are most important for the model

---

## 2. Dataset

**Source**

* Kaggle: *Uber and Lyft Dataset Boston, MA*
* Time period: 2018-11-26 to 2018-12-18 (roughly 3 weeks)
* Location: Boston, MA
* Records: ~693,000 rides
* Columns: 57

**Key fields**

Trip and platform:

* `cab_type` (Uber or Lyft)
* `name` (service type, for example UberX, Lyft, UberBlack, etc.)
* `source`, `destination` (Boston neighborhoods)
* `distance` (miles)
* `price` (target variable, in USD)
* `surge_multiplier`

Time:

* `timestamp`, `datetime`
* `hour`, `day`, `month`

Weather:

* `short_summary`, `long_summary`
* `temperature`, `apparentTemperature`
* `precipIntensity`, `precipProbability`
* `windSpeed`, `windGust`, `windBearing`
* `cloudCover`, `humidity`, `uvIndex`, `pressure`, `visibility`, `dewPoint`, `ozone`
* Several daily min / max and sunrise / sunset fields

The raw CSV is stored in `data/rideshare_kaggle.csv` in the project and is also accessed directly from GitHub by the Streamlit app.

---

## 3. Repository Structure

A suggested project layout:

```text
.
├── app.py                        # Streamlit app
├── models/
│   └── xgb_best_model.joblib     # Saved XGBoost pipeline
├── data/
│   └── rideshare_kaggle.csv      # Kaggle dataset
├── notebooks/
│   └── CIS 508 - Final Project.ipynb  # EDA and model development
├── README.md
└── (optional) requirements.txt
```

In the notebook, `DATA_PATH = Path("../data/rideshare_kaggle.csv")`, so it expects to live in a `notebooks/` folder with the data one level up under `data/`.

The Streamlit app loads the model from `models/xgb_best_model.joblib` and uses the CSV from the GitHub URL in `DATA_URL`.

---

## 4. Environment and Dependencies

Core libraries used:

* Python 3.x
* `pandas`, `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib`, `seaborn` (for EDA and plots)
* `streamlit` (for deployment)
* `joblib` (for model persistence)
* `mlflow` (for experiment tracking on Databricks in the Colab version)

Example `pip` install:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib mlflow
```

---

## 5. Methodology

The full workflow is implemented in **`CIS 508 - Final Project.ipynb`**.

### 5.1 Data loading and basic inspection

Steps:

1. Load data from `../data/rideshare_kaggle.csv` into a pandas DataFrame.
2. Inspect shape, data types, and sample rows.
3. Check missing values, duplicates, and basic distributions for key fields.
4. Verify date and time fields (`timestamp`, `datetime`, `hour`, `day`, `month`) are consistent.

### 5.2 Cleaning and preprocessing

Main cleaning decisions:

* Drop identifier and purely technical columns that do not help prediction, such as `id`, `timezone`, and some redundant time fields.
* Remove rows with missing target `price`.
* Handle missing values in features (for example using median for numeric columns).
* Remove obvious outliers in `price` or `distance` if any appear (extreme rides with unrealistic cost or distance).

### 5.3 Feature engineering

**Time based features**

* `hour` (0 to 23)
* `day` and `month`
* `day_name` (Monday to Sunday)
* `month_name`
* `is_weekend` (Saturday or Sunday)
* `is_peak_hour` flag, based on rush hour windows (for example morning and evening commute periods)

**Trip features**

* `cab_type` (Uber vs Lyft)
* `name` (service type, for example UberX, Lyft, UberBlack)
* `source`, `destination` (Boston neighborhood level)
* `distance` (miles)

**Weather and context features**

A subset of weather attributes is used in the final app for prediction:

* `precipIntensity`
* `precipProbability`
* `windGust`
* `windBearing`
* `cloudCover`
* `uvIndex`
* `moonPhase`
* `precipIntensityMax`

Together with the engineered time flags, the final model focuses on a compact but informative set of predictors.

### 5.4 Train test split

* Target: `price`
* Features: numeric and categorical columns described above
* Data split into train and test sets (for example 80 percent train and 20 percent test) with a fixed random seed for reproducibility.

### 5.5 Preprocessing and pipelines

Use a `ColumnTransformer` with:

* Numeric features: `StandardScaler`
* Categorical features: `OneHotEncoder(handle_unknown="ignore", drop="first")`

Wrap the preprocessor and estimator into a `Pipeline` so that scaling and encoding are always applied consistently during training and inference.

**Numeric features in the final app**

```python
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
```

**Categorical features in the final app**

```python
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
```

### 5.6 Models trained

For comparison, several regression models are trained, each wrapped in a pipeline that includes the shared preprocessor and a model specific parameter grid:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* K Nearest Neighbors Regressor
* Linear SVR
* MLPRegressor (Neural network)
* XGBRegressor (XGBoost)

Each model is tuned with `GridSearchCV` on the training set (for example depth, learning rate, number of estimators, etc. for XGBoost).

### 5.7 Evaluation

Metrics used on the test set:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Coefficient of determination (R²)

Results are stored in a `model_results` list, converted to a DataFrame, and sorted by RMSE to identify the best performing model.

From the comparison:

* **XGBRegressor** achieves the lowest RMSE and MAE and the highest R² on the held out test set, so it is selected for deployment.
* A bar chart of RMSE by model is generated for quick visual comparison.

### 5.8 Model interpretation

For the final XGBoost model:

* Feature importance is computed and plotted for the top features.
* The Streamlit app includes an optional image of the feature importance chart so instructors and users can see what drives the model.
* In the app text, the model is explained as being driven mainly by:

  * Distance
  * Time of day and day of week
  * Origin and destination
  * Service type
  * Peak hour and weekend flags
  * Weather has a smaller, but still present, effect.

### 5.9 Model persistence

The fitted pipeline that includes the preprocessor and the tuned XGBRegressor is saved with joblib:

```python
import joblib

joblib.dump(best_xgb_pipeline, "models/xgb_best_model.joblib")
```

This file is loaded by `app.py` at runtime.

---

## 6. Streamlit App

The web app is implemented in **`app.py`**.

### 6.1 What the app does

* Loads the trained XGBoost pipeline from `models/xgb_best_model.joblib`.
* Loads the raw dataset from GitHub to:

  * Build lists of valid origins and destinations
  * Build a mapping from cab type to service names
  * Compute global medians for numeric features (used as defaults)
  * Compute median distances for each source to destination pair
  * Extract default values for day, month, and typical weather conditions
* Renders an interactive UI where the user can:

  * Select cab platform and service
  * Select origin and destination
  * Choose pickup time and day
  * Adjust simple context toggles (weekend, peak hour)
  * Adjust weather conditions if desired, or keep median defaults
* Creates a single row DataFrame with features ordered as in training.
* Calls `model.predict` to get the price estimate.
* Displays:

  * The predicted ride price (as a `st.metric`)
  * The final feature vector (in an expander, for debugging or grading)
  * A Google Maps link for the chosen route
  * An explanation block about the model
  * An optional feature importance chart image when present

### 6.2 How to run the app locally

From the project root:

1. Ensure the model file is in place:

   ```text
   models/xgb_best_model.joblib
   ```

   If it is missing, re run the notebook, identify the best XGBoost pipeline, and save it with joblib to that path.

2. Install dependencies:

   ```bash
   pip install streamlit pandas joblib scikit-learn xgboost
   ```

3. Run Streamlit:

   ```bash
   streamlit run app.py
   ```

4. Open the URL that Streamlit prints in your terminal, usually `http://localhost:8501`.

### 6.3 Using the app

Typical workflow:

1. **Pick a platform and service**

   * Select `Uber` or `Lyft`.
   * Select a service such as UberX, UberBlack, Lyft, Lyft XL, etc. The options are filtered by the chosen cab type.

2. **Set the route**

   * Choose an origin neighborhood in Boston.
   * Choose a destination that is different from the origin. If there is no alternate option (edge case), all destinations are shown.

3. **Set time and date context**

   * Slider for pickup hour (24 hour clock).
   * Dropdown for day of week (default taken from typical day in the data).
   * Dropdown for month (default aligned with the dataset period).
   * Toggle for weekend flag.
   * Toggle for peak hour flag.

4. **Set weather (optional)**

   * Weather fields such as `precipIntensity`, `precipProbability`, `windGust`, etc. have default values based on global medians.
   * You can adjust them if you want to simulate a stormy or windy day.

5. **Predict**

   * Click the button to predict the price.
   * The predicted price is displayed.
   * Expand "Show model inputs" to see the exact feature values that were passed into the model.
   * A "Route on Google Maps" section includes a link that opens the origin to destination path in Google Maps.

6. **Model explanation**

   * A final section describes which features the model relies on most.
   * If `feature_importance.png` (or similar) is present in the repo, you can expand "Show feature importance chart" to view the top 20 features ranked by importance.

---

## 7. Notebooks and Experiment Tracking

### 7.1 Main notebook

**`notebooks/CIS 508 - Final Project.ipynb`** contains:

* Data loading and cleaning
* Feature engineering
* Model training (all eight models)
* Model comparison plots
* Feature importance plots for the final model
* Code to persist the final XGBoost pipeline to `models/xgb_best_model.joblib`

### 7.2 Databricks and MLflow (course requirement)

The project is structured so that:

* The same modeling steps can be run on a Databricks cluster.
* MLflow can be used to:

  * Log parameter grids and chosen hyperparameters
  * Log metrics such as RMSE, MAE, and R²
  * Log artifacts such as plots and the trained model

In your Colab or Databricks version, you would typically:

1. Set up MLflow tracking URI to point to your Databricks workspace.
2. Wrap each model training block in `with mlflow.start_run():` and log parameters and metrics.
3. Register or save the best performing model and then export it to the `models/` folder for use in the Streamlit app.

---

## 8. Reproducibility and How To Retrain

To retrain the model or update it with new data:

1. Place the updated dataset in `data/rideshare_kaggle.csv` (same schema).
2. Open `CIS 508 - Final Project.ipynb` in Jupyter or Databricks.
3. Run all cells:

   * EDA
   * Preprocessing and feature engineering
   * Model training and comparison
4. Confirm that XGBRegressor (or another model) is the best based on RMSE and MAE.
5. Save the chosen model pipeline to `models/xgb_best_model.joblib`.
6. Restart the Streamlit app and verify that predictions work.

---

## 9. Business Value and Limitations

**Business value**

* Gives riders or planners a quick benchmark estimate of ride prices without opening the real app.
* Helps operations teams explore how price behaves across:

  * Different services
  * Different origins and destinations
  * Different times and days
  * Different weather conditions
* Can be plugged into dashboards or planning tools to simulate what if scenarios.

**Limitations**

* Data is from a specific city (Boston) and a short time window, so the model is not directly generalizable to other cities or seasons.
* Prices come from historical data and do not account for changes in platform pricing policies over time.
* Certain factors, such as special events, traffic incidents, or real time demand spikes, are not explicitly modeled.

These limitations are discussed in the context of the CIS 508 project rubric, with suggestions for possible extensions like more recent data, other cities, or including event calendars.

---

nice this looks clean.

if you want, I can add a **final section in the README** called
**“Live Demo”** with your deployed Streamlit link included in a professional way.

here’s the exact block you can paste into the README:

---

## Live Demo

The fully deployed Streamlit application for this project is available here:

👉 **[https://meshachsamuel-cis508-finalproject.streamlit.app](https://meshachsamuel-cis508-finalproject.streamlit.app)**

This app loads the trained XGBoost pipeline, allows users to input ride details (service type, route, time, weather) and returns a predicted price in real time.

---