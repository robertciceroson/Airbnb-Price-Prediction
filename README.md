# 🏙️ Airbnb NYC Price Prediction

An end-to-end machine learning pipeline that predicts nightly listing prices for Airbnb properties in New York City using the NYC Airbnb Open Data 2019 dataset.

---

## 📋 Project Overview

| Field | Details |
|---|---|
| **Author** | Robert Cicero Son |
| **Dataset** | NYC Airbnb Open Data 2019 (~49,000 listings, 16 features) |
| **Target Variable** | Nightly listing price (USD) |
| **Best Model** | XGBoost |
| **Best Test R²** | ~0.65 |
| **Best MAE** | ~$35–40 per night |

---

## 🎯 Objective

Build a regression model that accurately predicts the nightly price of an Airbnb listing in NYC based on features such as location, room type, borough, availability, and review history. The model can be used by hosts to price competitively and by analysts to understand what drives listing prices.

---

## 📁 Repository Structure

```
Airbnb-Price-Prediction/
│
├── AB_NYC_2019.csv                        # Raw dataset (NYC Airbnb Open Data 2019)
├── Airbnb_NYC_Price_Prediction.ipynb      # Main notebook — full ML pipeline
├── Airbnb_NYC_Price_Prediction.pdf        # Notebook exported as PDF
├── Airbnb_NYC_Price_Prediction_Report.pdf # Project report
└── README.md                              # This file
```

---

## 🔧 Pipeline Steps

### 1. Data Loading & Exploration
- Auto-detects or downloads the dataset
- Profiles missing values, distributions, and data types

### 2. Data Preprocessing
- Drops irrelevant/leaky columns (`id`, `host_id`, `host_name`, `last_review`, `name`)
- Fills `reviews_per_month` NaN → 0 (no reviews = 0 reviews per month)
- Removes zero-price listings and caps outliers at the 99.5th percentile

### 3. Categorical Encoding
- **One-hot encoding** for `neighbourhood_group` (5 boroughs) and `room_type` (3 types)
- **Target encoding** for `neighbourhood` (~220 unique values) — each neighbourhood is replaced by its mean `log(price)`, computed on training data only to prevent data leakage

### 4. Exploratory Data Analysis (EDA)
- Univariate distributions for price, room type, borough, and numeric features
- Bivariate analysis: price by room type, borough, and numeric features
- Correlation heatmap and geographic price map

### 5. Feature Engineering
New features created from existing data:
| Feature | Description |
|---|---|
| `log_price` | Log-transformed target (reduces skewness) |
| `log_min_nights` | Log-transformed minimum nights |
| `log_num_reviews` | Log-transformed review count |
| `reviews_per_listing` | Reviews ÷ host listing count (demand proxy) |
| `availability_rate` | availability_365 ÷ 365 |
| `has_reviews` | Binary flag — listing has at least one review |
| `is_superhost_proxy` | Binary flag — host has more than 3 listings |
| `engagement_score` | reviews_per_month × availability_rate |
| `lat_lon_product` | Latitude × \|longitude\| (joint spatial signal) |

### 6. Feature Selection
- **Mutual Information** scoring (fit on training data only)
- **Random Forest importance** scoring (fit on training data only)
- Final feature set = union of both methods (features passing either threshold)

### 7. Model Training & Evaluation
10 models evaluated with 5-fold cross-validation:

| Model | CV R² | Test R² |
|---|---|---|
| XGBoost | ~0.64 | ~0.65 |
| LightGBM | ~0.63 | ~0.64 |
| Random Forest | ~0.62 | ~0.63 |
| Extra Trees | ~0.61 | ~0.62 |
| Gradient Boosting | ~0.60 | ~0.61 |
| Decision Tree | ~0.52 | ~0.51 |
| Ridge | ~0.47 | ~0.47 |
| Lasso | ~0.46 | ~0.46 |
| ElasticNet | ~0.46 | ~0.46 |
| Linear Regression | ~0.46 | ~0.46 |

### 8. Hyperparameter Tuning
- `GridSearchCV` on the best-performing model
- 5-fold cross-validation across all parameter combinations

### 9. Model Persistence
Trained model, scaler, and feature names saved to `models/` using `joblib`:
- `models/best_model.pkl`
- `models/scaler.pkl`
- `models/feature_names.pkl`
- `models/results_summary.pkl`

### 10. Interactive Price Predictor
Input a listing's details (location, room type, borough, availability, reviews) and get a predicted nightly price with a confidence range.

---

## 🚀 How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm joblib scipy
```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/robertciceroson/Airbnb-Price-Prediction.git
   cd Airbnb-Price-Prediction
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook Airbnb_NYC_Price_Prediction.ipynb
   ```

3. Run all cells top to bottom (`Kernel → Restart & Run All`)

> The notebook will auto-download the dataset if `AB_NYC_2019.csv` is not found in the working directory.

---

## 📊 Key Findings

- **Room type** is the strongest price driver — entire homes/apartments command a median ~2× the price of private rooms
- **Manhattan** listings price ~40% higher than the NYC median
- **Location** (latitude/longitude) and the neighbourhood target encoding are among the top 5 predictive features
- Log-transforming the target variable reduces skewness from ~20 to ~0.3, significantly improving linear model performance
- Tree-based ensemble methods (XGBoost, LightGBM, Random Forest) substantially outperform linear models on this dataset

---

## 🛠️ Technical Highlights

- **No data leakage** — train/test split is performed before feature selection, and neighbourhood target encoding is computed on training data only
- **RobustScaler** used for linear models (more resistant to outliers than StandardScaler)
- **Residual analysis** confirms approximately unbiased predictions across the price range
- **Input validation** in the interactive predictor guards against out-of-range values

---

## 📄 License

This project uses the [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) dataset, available under the CC0 Public Domain license.
