#  AirCast Delhi — AI-Powered Air Quality Forecasting System

> **Smart India Hackathon 2025 | Team VisionX | PS 25178 | Space Technology Theme**

AirCast Delhi is a machine learning system that forecasts **ground-level O₃ (Ozone) and NO₂ (Nitrogen Dioxide)** concentrations for **7 monitoring sites across Delhi**, 24 hours in advance using satellite observations and reanalysis data.

---

##  Problem Statement

Short-term (24-hour, hourly) forecasting of ground-level O₃ and NO₂ using:
- **SAC ISRO reanalysis data**
- **Sentinel-5P satellite observations**

Accurate air quality forecasts enable early warnings, better policy decisions, and protect public health in one of the world's most polluted cities.

---

##  Live Demo

> Dashboard deployment in progress — link will be updated post-deployment.

Run locally with:
```bash
streamlit run app.py
```

---

##  Features

-  **24-hour hourly O₃ and NO₂ predictions**
-  **Site-specific forecasts for 7 Delhi monitoring locations**
-  **Historical pattern analysis and seasonal trends**
-  **Real-time interactive air quality visualization**
-  **Model performance metrics (RMSE, R², RIA)**

---

##  Technical Approach

### Data Sources
| Source | Description |
|---|---|
| SAC ISRO Reanalysis Data | Ground-level atmospheric reanalysis |
| Sentinel-5P (ESA) | Satellite-derived tropospheric gas columns |

### Feature Engineering
- **Lag features** — 24h and 48h historical values
- **Cyclical encoding** — hour of day, day of week, month (sin/cos)
- **Traffic pattern proxies** — rush hour indicators
- **Meteorological variables** — temperature, wind, humidity

### Model Architecture

```
Ensemble Model = XGBoost (60%) + LightGBM (40%)
```

- **XGBoost** — handles feature interactions and non-linear patterns
- **LightGBM** — fast training on large datasets with leaf-wise growth
- **Early stopping** — prevents overfitting on validation set
- **Weighted ensemble** — tuned on validation RMSE

### Model Performance Targets

| Pollutant | RMSE Target | R² Target | RIA Target |
|---|---|---|---|
| O₃ | < 20 μg/m³ | > 0.75 | > 0.80 |
| NO₂ | < 25 μg/m³ | > 0.75 | > 0.80 |

> **RIA** = Refined Index of Agreement — a robust metric for forecast accuracy.

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| ML Models | XGBoost, LightGBM, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Satellite Data | Sentinel-5P (ESA Copernicus) |
| Reanalysis Data | SAC ISRO |

---

##  Project Structure

```
aircast-delhi/
│
├── data/
│   ├── raw/                        # Original CSVs for 7 sites (train + unseen)
│   │   ├── site_1_train_data.csv
│   │   ├── site_1_unseen_input_data.csv
│   │   └── ... (site 2–7)
│   └── lat_lon_sites.txt           # Lat/lon coordinates of monitoring sites
│
├── notebooks/
│   ├── 01_eda_analysis.ipynb       # Exploratory data analysis
│   └── 02_model_training.ipynb     # Model training + evaluation (SIH demo)
│
├── plots/                          # Output visualizations
│   ├── all_sites_comparison.png
│   ├── complete_analysis.png
│   ├── correlation_heatmap.png
│   ├── pollution_analysis.png
│   └── seasonal_patterns.png
│
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

##  Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/kaushalkumar94/aircast-delhi.git
cd aircast-delhi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit dashboard
streamlit run app.py
```

---

##  Key Visualizations

| Plot | Description |
|---|---|
| `all_sites_comparison.png` | O₃/NO₂ levels across all 7 Delhi sites |
| `correlation_heatmap.png` | Feature correlation matrix |
| `seasonal_patterns.png` | Monthly and seasonal pollution trends |
| `pollution_analysis.png` | Pollutant distribution and outlier analysis |
| `complete_analysis.png` | End-to-end model prediction vs actual |

---

##  Team

**Team VisionX** — Smart India Hackathon 2025

| Name | Role |
|---|---|
| Kaushal Kumar | ML Model Development, Feature Engineering, EDA |

---

##  License

This project is open-source under the [MIT License](LICENSE).

---

##  Connect

- 💼 [LinkedIn](https://linkedin.com/in/kaushal-kumar-287898275)
- 💻 [GitHub](https://github.com/kaushalkumar94)
- 📧 kaushalkumar5354@gmail.com
