# 🍕 Food Delivery Time Predictor

[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit%20App-FF4B4B?logo=streamlit&logoColor=white)](https://food-delivery-estimated-time-predictor.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20EDA-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)

🔗 **Live Web Application:** [https://food-delivery-estimated-time-predictor.streamlit.app/](https://food-delivery-estimated-time-predictor.streamlit.app/)

An interactive, end-to-end Machine Learning web application built with **Streamlit** and **Scikit-learn** that predicts food delivery times in minutes using an **Optimized Random Forest Regressor** ($R^2 = 0.96$). Beyond instant inference on a pre-trained benchmark model, the platform includes a self-service **Custom Dataset Pipeline** built specifically to empower local cafes, restaurants, and small vendors to train private ETA models on their own order history—without needing expensive enterprise dispatch tools.

---

## 🎯 Why This Project Exists (Real-World Business Use Case)

Large delivery aggregators like **Zomato** and **Swiggy** rely on costly, proprietary logistics engines to forecast delivery ETAs. In contrast, **local cafes, cloud kitchens, neighborhood restaurants, and small shopkeepers** rarely have access to paid predictive software.

### How Local Businesses Use This Tool
1. **Build a Custom Dataset in 15–30 Days:** Business owners can download the free **Empty Template CSV**, log each delivery step-by-step (distance, prep time, weather, traffic, time of day, vehicle, rider experience, and actual delivery duration), and accumulate enough historical records within **15 to 30 days** of normal operations to train a dedicated model tailored to their own kitchen speed and local roads.
2. **Handle Live Phone Orders With Confidence:** Imagine a customer calls to place an order and asks:
   > *"How long will my food take to arrive? I have other tasks to run."*

   Instead of guessing or drawing a blank—especially when a newer staff member is taking the call—the operator simply selects current local conditions (**traffic level, time of day, weather, distance, preparation time, and courier experience**).
3. **Give Realistic ETA Windows Using Model Error:** If the model predicts **30 minutes** (with an average error of roughly **3–5 minutes**, such as the benchmark model's $\text{MAE} = 3.60\text{ min}$), staff can immediately give the customer a reliable, data-backed estimate like **"around 30 minutes"** or **"30 to 35 minutes."**

---

## 🚀 Key Features

### 1. 🏠 Overview & Guided Workflow
* Explains the business problem, real-world phone-order workflow, and step-by-step onboarding for both demo users and business owners.
* Highlights key model performance metrics ($R^2$, MAE, RMSE) and statistical findings at a glance.

### 2. 🔮 Test The Model (Instant Demo)
* **Real-Time ETA Predictions:** Adjust numerical sliders (`Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs`) and categorical dropdowns (`Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`) to generate instant delivery time estimates.
* **Benchmark Dataset Summary:** Displays overview metrics and reference details for the underlying training dataset (`Food_Delivery_Times.csv`).
* **Interactive Data Quality Report:** Automatically renders target distributions, numerical trend scatter plots, categorical impact rankings, mean delivery-time bar charts, box plots, two-factor interaction heatmaps, and a correlation matrix.

### 3. 📊 Try With Your Own Dataset (Custom ML Pipeline)
* **One-Click Templates:** Download an empty CSV template or a pre-filled sample CSV directly from the interface.
* **Automated Data Cleaning:** Validates column schemas, coerces numeric fields, strips formatting inconsistencies, drops rows with missing targets, and automatically imputes missing predictor values (median for numeric, mode for categorical).
* **Feature Importance Ranking:** Ranks all 7 predictors by their percentage impact on delivery time against a 5% importance threshold.
* **Three Modeling Modes:**
  * ⭐ **Choose Best Features** — Automatically selects top-ranked drivers.
  * 🌐 **Select All Features** — Trains using all 7 valid predictors.
  * 🛠️ **Customized Feature Selection** — Manually pick any subset of features.
* **Live Custom Inference & EDA:** Evaluates the custom model on an 80/20 hold-out split ($R^2$ and MAE), dynamically builds input controls tailored to the user's data ranges/categories, predicts custom ETAs, and generates a dedicated Data Quality Report.

---

## 📈 Model Performance & Methodology

The baseline models were benchmarked and tuned using **`RandomizedSearchCV` (100 iterations, 5-fold cross-validation)** to maximize predictive accuracy and minimize error:

| Metric | Default Baseline | Optimized Random Forest | Improvement |
| :--- | :---: | :---: | :---: |
| **$R^2$ Score** | 0.91 | **0.96** | +0.05 |
| **Mean Absolute Error (MAE)** | 5.11 min | **3.60 min** | -1.51 min |
| **Root Mean Squared Error (RMSE)** | 6.61 min | **4.59 min** | -2.02 min |

* **Best Hyperparameters:** `n_estimators=900`, `max_depth=13`, `max_features=4`, `min_samples_split=5`, `min_samples_leaf=1`
* **Key Statistical Insights:**
  1. **Distance** is the single strongest predictor of total delivery duration.
  2. **Adverse weather** (`Snowy` and `Rainy`) introduces measurable delays compared to `Clear` conditions.
  3. **High traffic congestion** independently extends delivery time regardless of route distance.
  4. **Courier experience** (`7–9 years`) noticeably reduces delivery time compared to newer couriers (`0–1 year`).
  5. All four categorical predictors passed **One-Way ANOVA** testing ($p < 0.05$), confirming statistical significance.

---

## 🗂️ Dataset Schema

The application expects the following standard schema (used by both the benchmark dataset and custom CSV uploads):

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `Order_ID` | Identifier | Unique order ID (automatically excluded from modeling) | `1001` |
| `Distance_km` | Numeric | Road distance between restaurant and customer (km) | `15.57` |
| `Weather` | Categorical | Weather condition (`Clear`, `Rainy`, `Foggy`, `Snowy`, `Windy`) | `Rainy` |
| `Traffic_Level` | Categorical | Road traffic intensity (`Low`, `Medium`, `High`) | `High` |
| `Time_of_Day` | Categorical | Order window (`Morning`, `Afternoon`, `Evening`, `Night`) | `Evening` |
| `Vehicle_Type` | Categorical | Delivery vehicle (`Bike`, `Scooter`, `Car`) | `Scooter` |
| `Preparation_Time_min` | Numeric | Kitchen preparation time (minutes) | `21` |
| `Courier_Experience_yrs` | Numeric | Courier delivery experience (years) | `3.5` |
| `Delivery_Time_min` | Numeric *(Target)* | Total end-to-end delivery duration (minutes) | `71` |

* **Benchmark Dataset Source:** [Kaggle — Food Delivery Time Prediction](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction) (1,000 records).

---

## 🏗️ Repository Structure

```text
food-delivery-time-predictor/
├── app.py                                    # Main Streamlit web application & UI layout
├── eda_engine.py                             # Pure-Python + Plotly Data Quality & EDA report generator
├── pipeline_engine.py                        # CSV validation, cleaning, feature ranking & custom model trainer
├── optimized_rf_model.pkl                    # Serialized pre-trained Random Forest Regressor (R² = 0.96)
├── label_encoders.pkl                        # Saved Scikit-learn LabelEncoders for categorical features
├── Food_Delivery_Times.csv                   # Benchmark dataset (1,000 delivery records)
├── food_delivery_sample_prefilled.csv        # Sample CSV for testing the custom dataset pipeline
├── Food_Delivery_Time_Prediction_model.ipynb # Jupyter Notebook with full EDA, ANOVA & hyperparameter tuning
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

### Architecture & Performance Design
* **Modular Separation:** `app.py` handles UI rendering and state management, `pipeline_engine.py` isolates custom ML preprocessing and training, and `eda_engine.py` builds interactive Plotly figures and plain-English insights without coupling computation to UI reruns.
* **Caching & Deduplication:** Uses `@st.cache_resource` for model/encoder artifacts, `@st.cache_data` for CSV reads, MD5 file-hash deduplication (`cd_sig`) to skip redundant cleaning, and feature-set memoization (`cd_models`) so slider interactions remain instantaneous.

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Pranay-256/YOUR-REPO-NAME.git](https://github.com/Pranay-256/YOUR-REPO-NAME.git)
   cd YOUR-REPO-NAME
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 🔄 Version History

* **v1.1 (November 2026)**
  * Added interactive visualizations and automated plain-English insights via `eda_engine.py`.
  * Added the **Try With Your Own Dataset** pipeline (`pipeline_engine.py`) with automated cleaning, feature importance ranking, and custom Random Forest training for local cafes and small businesses.
  * Added the **Overview** workflow & real-world business use-case tab with responsive equal-height UI cards.
* **v1.0 (September 2026)**
  * Initial release featuring the Optimized Random Forest model ($R^2 = 0.96$), interactive prediction controls, and dark gold/teal Streamlit interface.

---

## 👨‍💻 Contributors

Developed by **Pranay Jha**, **Pradeep Singh Negi**, and **Pranav Jha**.

* **Live App:** [https://food-delivery-estimated-time-predictor.streamlit.app/](https://food-delivery-estimated-time-predictor.streamlit.app/)
* **GitHub:** [@Pranay-256](https://github.com/Pranay-256)
* **LinkedIn:** [Pranay Jha](https://www.linkedin.com/in/pranay-jha-6582a937b/)
