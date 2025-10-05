import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AirCast Delhi - SIH 2025",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# Helper functions
@st.cache_data
def load_data():
    """Load and combine all site data"""
    sites = {}
    for i in range(1, 8):
        try:
            df = pd.read_csv(f'site_{i}_train_data.csv')
            df['site_id'] = i
            sites[f'site_{i}'] = df
        except:
            pass

    if sites:
        combined = pd.concat(sites.values(), ignore_index=True)
        return combined
    return None


@st.cache_data
def create_features(df):
    """Feature engineering pipeline - UPDATED to match training code"""
    df = df.copy()

    # Wind features
    df['wind_speed'] = np.sqrt(df['u_forecast'] ** 2 + df['v_forecast'] ** 2)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Date features
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Traffic patterns
    df['morning_rush'] = df['hour'].isin([6, 7, 8]).astype(int)
    df['afternoon_peak'] = df['hour'].isin([12, 13, 14, 15]).astype(int)

    # Sort for lag features
    df = df.sort_values(['site_id', 'date', 'hour'])

    # Lag features (24h and 48h as per training code)
    for lag in [24, 48]:
        df[f'O3_lag_{lag}h'] = df.groupby('site_id')['O3_target'].shift(lag)
        df[f'NO2_lag_{lag}h'] = df.groupby('site_id')['NO2_target'].shift(lag)

    return df.dropna()


def calculate_ria(y_true, y_pred):
    """Calculate Refined Index of Agreement"""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_pred - np.mean(y_true)) +
                          np.abs(y_true - np.mean(y_true))) ** 2)
    return 1 - (numerator / denominator)


@st.cache_resource
def train_models(X_train, y_train_o3, y_train_no2, X_val, y_val_o3, y_val_no2):
    """Train XGBoost and LightGBM models - UPDATED to match training code"""

    # O3 XGBoost model
    o3_xgb = XGBRegressor(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=30
    )
    o3_xgb.fit(X_train, y_train_o3, eval_set=[(X_val, y_val_o3)], verbose=False)

    # O3 LightGBM model
    o3_lgbm = LGBMRegressor(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
    o3_lgbm.fit(X_train, y_train_o3, eval_set=[(X_val, y_val_o3)])

    # NO2 XGBoost model
    no2_xgb = XGBRegressor(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=30
    )
    no2_xgb.fit(X_train, y_train_no2, eval_set=[(X_val, y_val_no2)], verbose=False)

    # NO2 LightGBM model
    no2_lgbm = LGBMRegressor(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
    no2_lgbm.fit(X_train, y_train_no2, eval_set=[(X_val, y_val_no2)])

    return o3_xgb, o3_lgbm, no2_xgb, no2_lgbm


def make_ensemble_predictions(o3_xgb, o3_lgbm, no2_xgb, no2_lgbm, X):
    """Make ensemble predictions - UPDATED with 60-40 weighting"""
    pred_o3_xgb = o3_xgb.predict(X)
    pred_o3_lgbm = o3_lgbm.predict(X)
    pred_o3_ensemble = 0.6 * pred_o3_xgb + 0.4 * pred_o3_lgbm

    pred_no2_xgb = no2_xgb.predict(X)
    pred_no2_lgbm = no2_lgbm.predict(X)
    pred_no2_ensemble = 0.6 * pred_no2_xgb + 0.4 * pred_no2_lgbm

    return pred_o3_ensemble, pred_no2_ensemble


# Load data
data = load_data()

if data is None:
    st.error("Unable to load data. Please ensure CSV files are in the correct directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("###  SIH 2025")
    st.markdown("**Team:** VisionX")
    st.markdown("**PS ID:** 25178")
    st.markdown("**Theme:** Space Technology")
    st.divider()

    page = st.radio("Navigate",
                    [" Home", " Live Predictions", " Data Analysis",
                     " Model Performance", " About"],
                    label_visibility="collapsed")

# Main content
if page == " Home":
    st.markdown('<div class="main-header">AirCast Delhi</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Air Quality Forecasting System</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monitoring Sites", "7 Locations", "Across Delhi")
    with col2:
        st.metric("Data Points", f"{len(data):,}", "Hourly Records")
    with col3:
        st.metric("Forecast Horizon", "24 Hours", "Hourly Updates")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Problem Statement")
        st.info("""
        **Objective:** Develop AI/ML-based models for short-term forecasting (24 hours) of 
        ground-level O₃ and NO₂ using satellite observations and reanalysis data.

        **Challenge:** Integrate sparse satellite data (3% coverage) with meteorological 
        forecasts to predict hourly pollution levels.
        """)

        st.markdown("###  Our Approach")
        st.success("""
        - **Ensemble Learning** with XGBoost + LightGBM (60-40 weighting)
        - **Advanced feature engineering** with temporal and meteorological features
        - **Lag features** (24h, 48h) for temporal dependencies
        - **Smart satellite integration** via forward-fill strategy
        - **Multi-site training** from 7 Delhi locations
        """)

    with col2:
        st.markdown("###  Key Features")
        features = [
            "24-hour hourly O₃ and NO₂ predictions",
            "Ensemble model (XGBoost + LightGBM)",
            "Real-time air quality visualization",
            "Site-specific and city-wide forecasts",
            "Historical pattern analysis",
            "Model performance metrics (RMSE, R², RIA)",
            "Scalable to other Indian cities"
        ]
        for feature in features:
            st.markdown(f"✓ {feature}")

        st.markdown("###  Expected Impact")
        st.warning("""
        - Public health alerts for vulnerable groups
        - Data-driven policy decisions
        - Early warning system for pollution episodes
        - Foundation for multi-city expansion
        """)

elif page == " Live Predictions":
    st.markdown("##  Live 24-Hour Air Quality Forecast")

    # Prepare data
    with st.spinner("Preparing models and data..."):
        processed_data = create_features(data)

        split_date = processed_data['date'].quantile(0.8)
        train_df = processed_data[processed_data['date'] < split_date]
        val_df = processed_data[processed_data['date'] >= split_date]

        exclude_cols = ['O3_target', 'NO2_target', 'date', 'year', 'month', 'day',
                        'NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
        feature_cols = [col for col in processed_data.columns if col not in exclude_cols]

        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        y_train_o3 = train_df['O3_target']
        y_val_o3 = val_df['O3_target']
        y_train_no2 = train_df['NO2_target']
        y_val_no2 = val_df['NO2_target']

        o3_xgb, o3_lgbm, no2_xgb, no2_lgbm = train_models(
            X_train, y_train_o3, y_train_no2, X_val, y_val_o3, y_val_no2
        )

    # Site selection
    selected_site = st.selectbox("Select Monitoring Site",
                                 [f"Site {i}" for i in range(1, 8)])
    site_num = int(selected_site.split()[1])

    # Get last 24 hours for selected site
    site_data = val_df[val_df['site_id'] == site_num].tail(24)

    if len(site_data) > 0:
        site_X = site_data[feature_cols]
        pred_o3, pred_no2 = make_ensemble_predictions(
            o3_xgb, o3_lgbm, no2_xgb, no2_lgbm, site_X
        )

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Hour': site_data['hour'].values,
            'O3_Predicted': pred_o3,
            'O3_Actual': site_data['O3_target'].values,
            'NO2_Predicted': pred_no2,
            'NO2_Actual': site_data['NO2_target'].values
        })

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg O₃ Forecast", f"{forecast_df['O3_Predicted'].mean():.1f} μg/m³")
        with col2:
            st.metric("Max O₃ Forecast", f"{forecast_df['O3_Predicted'].max():.1f} μg/m³")
        with col3:
            st.metric("Avg NO₂ Forecast", f"{forecast_df['NO2_Predicted'].mean():.1f} μg/m³")
        with col4:
            st.metric("Max NO₂ Forecast", f"{forecast_df['NO2_Predicted'].max():.1f} μg/m³")

        # Interactive plot
        fig = make_subplots(rows=2, cols=1, subplot_titles=("O₃ Forecast", "NO₂ Forecast"))

        # O3 plot
        fig.add_trace(go.Scatter(x=forecast_df['Hour'], y=forecast_df['O3_Actual'],
                                 mode='lines+markers', name='O₃ Actual',
                                 line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast_df['Hour'], y=forecast_df['O3_Predicted'],
                                 mode='lines+markers', name='O₃ Predicted (Ensemble)',
                                 line=dict(color='lightblue', width=2, dash='dash')), row=1, col=1)

        # NO2 plot
        fig.add_trace(go.Scatter(x=forecast_df['Hour'], y=forecast_df['NO2_Actual'],
                                 mode='lines+markers', name='NO₂ Actual',
                                 line=dict(color='red', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=forecast_df['Hour'], y=forecast_df['NO2_Predicted'],
                                 mode='lines+markers', name='NO₂ Predicted (Ensemble)',
                                 line=dict(color='lightcoral', width=2, dash='dash')), row=2, col=1)

        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="O₃ (μg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="NO₂ (μg/m³)", row=2, col=1)
        fig.update_layout(height=600, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("View Detailed Hourly Forecast"):
            st.dataframe(forecast_df, use_container_width=True)
    else:
        st.warning("No data available for selected site")

elif page == " Data Analysis":
    st.markdown("##  Historical Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Hourly Patterns", "Seasonal Trends", "Site Comparison"])

    with tab1:
        st.markdown("### Hourly Pollution Patterns")

        hourly_data = data.groupby('hour')[['O3_target', 'NO2_target']].mean().reset_index()

        fig = make_subplots(rows=1, cols=2, subplot_titles=("O₃ Pattern", "NO₂ Pattern"))

        fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['O3_target'],
                                 mode='lines+markers', fill='tozeroy',
                                 line=dict(color='blue', width=3)), row=1, col=1)

        fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['NO2_target'],
                                 mode='lines+markers', fill='tozeroy',
                                 line=dict(color='red', width=3)), row=1, col=2)

        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="O₃ (μg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="NO₂ (μg/m³)", row=1, col=2)
        fig.update_layout(height=400, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key Insights:** O₃ peaks around afternoon (12-15h) due to photochemical reactions. NO₂ shows traffic-related peaks during morning rush (6-8h).")

    with tab2:
        st.markdown("### Seasonal Trends")

        monthly_data = data.groupby('month')[['O3_target', 'NO2_target']].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['O3_target'],
                                 mode='lines+markers', name='O₃',
                                 line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['NO2_target'],
                                 mode='lines+markers', name='NO₂',
                                 line=dict(color='red', width=3)))

        fig.update_layout(xaxis_title="Month", yaxis_title="Concentration (μg/m³)",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key Insights:** O₃ peaks in summer months, NO₂ peaks in winter due to crop burning and temperature inversion.")

    with tab3:
        st.markdown("### Site-wise Comparison")

        site_data = data.groupby('site_id')[['O3_target', 'NO2_target']].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=site_data['site_id'], y=site_data['O3_target'],
                             name='O₃', marker_color='blue'))
        fig.add_trace(go.Bar(x=site_data['site_id'], y=site_data['NO2_target'],
                             name='NO₂', marker_color='red'))

        fig.update_layout(xaxis_title="Site ID", yaxis_title="Average Concentration (μg/m³)",
                          height=400, barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Key Insights:** Sites show 2-3x variation in pollution levels, indicating spatial heterogeneity across Delhi.")

elif page == " Model Performance":
    st.markdown("##  Model Performance Metrics")
    with st.spinner("Training models and calculating metrics..."):
        processed_data = create_features(data)

        split_date = processed_data['date'].quantile(0.8)
        train_df = processed_data[processed_data['date'] < split_date]
        val_df = processed_data[processed_data['date'] >= split_date]

        exclude_cols = ['O3_target', 'NO2_target', 'date', 'year', 'month', 'day',
                        'NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
        feature_cols = [col for col in processed_data.columns if col not in exclude_cols]

        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        y_train_o3 = train_df['O3_target']
        y_val_o3 = val_df['O3_target']
        y_train_no2 = train_df['NO2_target']
        y_val_no2 = val_df['NO2_target']

        o3_xgb, o3_lgbm, no2_xgb, no2_lgbm = train_models(
            X_train, y_train_o3, y_train_no2, X_val, y_val_o3, y_val_no2
        )

        pred_o3, pred_no2 = make_ensemble_predictions(
            o3_xgb, o3_lgbm, no2_xgb, no2_lgbm, X_val
        )

        # Calculate metrics
        o3_rmse = np.sqrt(mean_squared_error(y_val_o3, pred_o3))
        o3_r2 = r2_score(y_val_o3, pred_o3)
        o3_ria = calculate_ria(y_val_o3, pred_o3)

        no2_rmse = np.sqrt(mean_squared_error(y_val_no2, pred_no2))
        no2_r2 = r2_score(y_val_no2, pred_no2)
        no2_ria = calculate_ria(y_val_no2, pred_no2)

    st.info("**Model Architecture:** Ensemble of XGBoost (60%) + LightGBM (40%)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### O₃ Model Performance")
        st.metric("RMSE", f"{o3_rmse:.2f} μg/m³", delta="Lower is better", delta_color="inverse")
        st.metric("R² Score", f"{o3_r2:.3f}", delta="Target: > 0.75")
        st.metric("RIA", f"{o3_ria:.3f}", delta="Target: > 0.80")

    with col2:
        st.markdown("### NO₂ Model Performance")
        st.metric("RMSE", f"{no2_rmse:.2f} μg/m³", delta="Lower is better", delta_color="inverse")
        st.metric("R² Score", f"{no2_r2:.3f}", delta="Target: > 0.75")
        st.metric("RIA", f"{no2_ria:.3f}", delta="Target: > 0.80")

    st.divider()

    # Scatter plots
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(x=y_val_o3, y=pred_o3, labels={'x': 'Actual O₃', 'y': 'Predicted O₃'},
                         title="O₃: Predicted vs Actual")
        fig.add_trace(go.Scatter(x=[0, 200], y=[0, 200], mode='lines',
                                 name='Perfect Prediction', line=dict(color='red', dash='dash')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(x=y_val_no2, y=pred_no2, labels={'x': 'Actual NO₂', 'y': 'Predicted NO₂'},
                         title="NO₂: Predicted vs Actual")
        fig.add_trace(go.Scatter(x=[0, 300], y=[0, 300], mode='lines',
                                 name='Perfect Prediction', line=dict(color='red', dash='dash')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.divider()
    st.markdown("### Feature Importance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        o3_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': o3_xgb.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        fig = px.bar(o3_importance, x='importance', y='feature', orientation='h',
                     title='Top 15 Features for O₃ Prediction',
                     labels={'importance': 'Importance', 'feature': 'Feature'})
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        no2_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': no2_xgb.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        fig = px.bar(no2_importance, x='importance', y='feature', orientation='h',
                     title='Top 15 Features for NO₂ Prediction',
                     labels={'importance': 'Importance', 'feature': 'Feature'})
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Time series comparison
    st.divider()
    st.markdown("### Time Series Validation (Sample)")

    # Take first 168 hours (1 week) for visualization
    sample_size = min(168, len(y_val_o3))
    sample_idx = slice(0, sample_size)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("O₃ Time Series Comparison", "NO₂ Time Series Comparison"))

    # O3 time series
    fig.add_trace(go.Scatter(x=list(range(sample_size)), y=y_val_o3.values[sample_idx],
                             mode='lines', name='O₃ Actual',
                             line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(sample_size)), y=pred_o3[sample_idx],
                             mode='lines', name='O₃ Predicted',
                             line=dict(color='lightblue', width=2, dash='dash')), row=1, col=1)

    # NO2 time series
    fig.add_trace(go.Scatter(x=list(range(sample_size)), y=y_val_no2.values[sample_idx],
                             mode='lines', name='NO₂ Actual',
                             line=dict(color='red', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(sample_size)), y=pred_no2[sample_idx],
                             mode='lines', name='NO₂ Predicted',
                             line=dict(color='lightcoral', width=2, dash='dash')), row=2, col=1)

    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="O₃ (μg/m³)", row=1, col=1)
    fig.update_yaxes(title_text="NO₂ (μg/m³)", row=2, col=1)
    fig.update_layout(height=600, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

else:  # About page
    st.markdown("##  About AirCast Delhi")

    st.markdown("""
    ### Project Overview
    AirCast Delhi is an AI-powered air quality forecasting system developed for Smart India Hackathon 2025.

    ### Technical Approach

    **1. Data Integration**
    - Reanalysis meteorological data from SAC, ISRO
    - Satellite observations from Sentinel-5P (TROPOMI)
    - Ground-truth measurements from CPCB monitoring stations

    **2. Feature Engineering**
    - Cyclical time encoding (hour, month)
    - Traffic pattern indicators (morning rush, afternoon peak)
    - Wind speed calculations
    - Day of week and weekend indicators
    - Lag features (24h, 48h) for temporal dependencies
    - Meteorological interactions

    **3. Model Architecture**
    - **Ensemble approach:** XGBoost (60%) + LightGBM (40%)
    - Separate optimization for O₃ and NO₂
    - Early stopping to prevent overfitting
    - Multi-site pooled training (7 locations)
    - Tree-based method: histogram optimization

    **4. Innovation Points**
    - Smart satellite data integration (handles 97% missing data)
    - Ensemble predictions for robustness
    - Lag features capture temporal dependencies
    - Site-specific and city-wide predictions
    - Scalable to other Indian cities

    ### Model Specifications
    - **Algorithm:** XGBoost + LightGBM Ensemble
    - **Max Depth:** 8
    - **Learning Rate:** 0.05
    - **Estimators:** 300
    - **Subsample:** 0.8
    - **Early Stopping:** 30 rounds
    - **Weighting:** 60% XGBoost + 40% LightGBM

    ### Team VisionX
    - **Problem Statement:** 25178
    - **Theme:** Space Technology
    - **Category:** Software
    - **Hackathon:** Smart India Hackathon 2025

    ### Technology Stack
    - Python 3.9+
    - XGBoost, LightGBM
    - Streamlit
    - Plotly
    - Pandas, NumPy, Scikit-learn

    ### Performance Metrics
    - **RMSE:** Root Mean Square Error (lower is better)
    - **R² Score:** Coefficient of Determination (target > 0.75)
    - **RIA:** Refined Index of Agreement (target > 0.80)

    ### Future Enhancements
    - Real-time API integration
    - Mobile application
    - Multi-city expansion
    - Source attribution module
    - Policy simulation tools
    - Deep learning models (LSTM, Transformers)
    - Uncertainty quantification
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AirCast Delhi | Team VisionX | SIH 2025 | PS 25178</p>
        <p>Ensemble Model: XGBoost (60%) + LightGBM (40%)</p>
    </div>
""", unsafe_allow_html=True)

