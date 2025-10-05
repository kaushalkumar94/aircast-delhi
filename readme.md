# AirCast Delhi - Air Quality Forecasting System

🏆 **Smart India Hackathon 2025** | Team VisionX | PS 25178

## 🎯 Problem Statement
Develop AI/ML-based models for short-term forecasting (24 hours) of ground-level O₃ and NO₂ using satellite observations and reanalysis data.

## 🚀 Live Demo
👉 [View Dashboard](https://your-app-url.streamlit.app) _(will be updated after deployment)_

## 📊 Features
- 24-hour hourly O₃ and NO₂ predictions
- Ensemble Model: XGBoost (60%) + LightGBM (40%)
- Real-time air quality visualization
- Site-specific forecasts for 7 Delhi locations
- Historical pattern analysis
- Model performance metrics (RMSE, R², RIA)

## 🔬 Technical Approach
- **Data Sources:** SAC ISRO reanalysis data + Sentinel-5P satellite observations
- **Feature Engineering:** Lag features (24h, 48h), cyclical encoding, traffic patterns
- **Model:** Ensemble of XGBoost + LightGBM with early stopping
- **Metrics:** RMSE, R² Score, Refined Index of Agreement (RIA)

## 🛠️ Technology Stack
- Python 3.9+
- Streamlit
- XGBoost, LightGBM
- Plotly
- Pandas, NumPy, Scikit-learn

## 💻 Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/aircast-delhi.git
cd aircast-delhi

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📈 Model Performance
- **O₃ Model:** RMSE < 20 μg/m³, R² > 0.75, RIA > 0.80
- **NO₂ Model:** RMSE < 25 μg/m³, R² > 0.75, RIA > 0.80

## 👥 Team VisionX
Smart India Hackathon 2025 - Space Technology Theme

## 📄 License
This project was developed for SIH 2025.

---
**Note:** Ensure all `site_*_train_data.csv` files are present in the root directory before running.