"""
================================================================================
FRAMEWORK FOR INTEGRATING INTELLIGENT TRANSPORTATION SYSTEMS FOR SRI LANKA
================================================================================
Master's Thesis - Management of Information Systems
Streamlit Interactive Dashboard with ML-based Predictions
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score)

# Page Configuration
st.set_page_config(
    page_title="Sri Lanka ITS Framework",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# DATA GENERATION
# ================================================================================
@st.cache_data
def generate_sri_lanka_dataset():
    """Generate comprehensive Sri Lanka ITS dataset with 2000+ records"""
    np.random.seed(42)
    n_records = 2000
    
    cities = {
        'Colombo': {'lat': 6.9271, 'lon': 79.8612, 'pop': 752993, 'type': 'Metro'},
        'Kandy': {'lat': 7.2906, 'lon': 80.6337, 'pop': 125400, 'type': 'Urban'},
        'Galle': {'lat': 6.0535, 'lon': 80.2210, 'pop': 99478, 'type': 'Coastal'},
        'Jaffna': {'lat': 9.6615, 'lon': 80.0255, 'pop': 88138, 'type': 'Northern'},
        'Negombo': {'lat': 7.2008, 'lon': 79.8737, 'pop': 142136, 'type': 'Suburban'},
        'Kurunegala': {'lat': 7.4863, 'lon': 80.3647, 'pop': 30315, 'type': 'Urban'},
        'Ratnapura': {'lat': 6.6828, 'lon': 80.3992, 'pop': 52170, 'type': 'Gem_City'},
        'Anuradhapura': {'lat': 8.3114, 'lon': 80.4037, 'pop': 63692, 'type': 'Historic'},
        'Trincomalee': {'lat': 8.5874, 'lon': 81.2152, 'pop': 99135, 'type': 'Port'},
        'Batticaloa': {'lat': 7.7310, 'lon': 81.6747, 'pop': 92332, 'type': 'Eastern'}
    }
    
    transport_modes = {
        'SLTB_Bus': {'capacity': 50, 'speed': (25, 45), 'cost': 3},
        'Private_Bus': {'capacity': 45, 'speed': (30, 50), 'cost': 5},
        'Sri_Lanka_Railways': {'capacity': 800, 'speed': (40, 80), 'cost': 2},
        'Three_Wheeler': {'capacity': 3, 'speed': (20, 40), 'cost': 50},
        'Private_Vehicle': {'capacity': 5, 'speed': (30, 80), 'cost': 15},
        'Motorcycle': {'capacity': 2, 'speed': (30, 70), 'cost': 8}
    }
    
    weather_types = ['Clear', 'Cloudy', 'Light_Rain', 'Heavy_Rain', 'Thunderstorm']
    monsoons = ['Southwest_Monsoon', 'Northeast_Monsoon', 'Inter_Monsoon', 'Dry_Season']
    festivals = ['Normal', 'Sinhala_Tamil_New_Year', 'Vesak', 'Poson', 'Esala_Perahera', 
                 'Deepavali', 'Christmas', 'Long_Weekend']
    
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_records)]
    
    data = []
    for i, date in enumerate(dates):
        city = np.random.choice(list(cities.keys()), 
                               p=[0.25, 0.12, 0.10, 0.08, 0.10, 0.08, 0.07, 0.07, 0.07, 0.06])
        city_info = cities[city]
        
        transport = np.random.choice(list(transport_modes.keys()),
                                    p=[0.20, 0.18, 0.12, 0.22, 0.18, 0.10])
        trans_info = transport_modes[transport]
        
        hour = date.hour
        day_of_week = date.weekday()
        month = date.month
        
        # Monsoon season determination
        if month in [5, 6, 7, 8, 9]:
            monsoon = 'Southwest_Monsoon'
            rain_prob = 0.6
        elif month in [10, 11, 12, 1, 2]:
            monsoon = 'Northeast_Monsoon'
            rain_prob = 0.5
        elif month in [3, 4]:
            monsoon = 'Inter_Monsoon'
            rain_prob = 0.4
        else:
            monsoon = 'Dry_Season'
            rain_prob = 0.2
        
        # Weather based on monsoon
        if np.random.random() < rain_prob:
            weather = np.random.choice(['Light_Rain', 'Heavy_Rain', 'Thunderstorm'], p=[0.5, 0.35, 0.15])
        else:
            weather = np.random.choice(['Clear', 'Cloudy'], p=[0.6, 0.4])
        
        # Rainfall
        rainfall = {'Clear': 0, 'Cloudy': np.random.uniform(0, 2),
                    'Light_Rain': np.random.uniform(2, 15), 
                    'Heavy_Rain': np.random.uniform(15, 50),
                    'Thunderstorm': np.random.uniform(30, 80)}[weather]
        
        temperature = 28 + np.random.normal(0, 2)
        if weather in ['Heavy_Rain', 'Thunderstorm']:
            temperature -= np.random.uniform(2, 5)
        temperature = np.clip(temperature, 22, 36)
        
        humidity = {'Clear': np.random.uniform(55, 75), 'Cloudy': np.random.uniform(60, 80),
                    'Light_Rain': np.random.uniform(70, 90), 
                    'Heavy_Rain': np.random.uniform(80, 95),
                    'Thunderstorm': np.random.uniform(85, 98)}[weather]
        
        is_rush_hour = hour in [7, 8, 9, 17, 18, 19]
        is_weekend = day_of_week >= 5
        
        festival = np.random.choice(festivals, p=[0.85, 0.02, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02])
        
        # Congestion calculation
        base_cong = 0.3
        if city == 'Colombo': base_cong += 0.25
        elif city in ['Kandy', 'Galle', 'Negombo']: base_cong += 0.15
        else: base_cong += 0.05
        
        if is_rush_hour and not is_weekend: base_cong += 0.25
        elif is_rush_hour: base_cong += 0.10
        
        weather_impact = {'Clear': 0, 'Cloudy': 0.02, 'Light_Rain': 0.10, 
                         'Heavy_Rain': 0.25, 'Thunderstorm': 0.35}
        base_cong += weather_impact[weather]
        
        if festival != 'Normal': base_cong += np.random.uniform(0.15, 0.30)
        
        congestion_index = np.clip(base_cong + np.random.normal(0, 0.08), 0, 1)
        
        vehicle_count = int((city_info['pop'] / 100) * (0.5 + congestion_index) * np.random.uniform(0.8, 1.2))
        
        base_speed = np.mean(trans_info['speed'])
        speed_reduction = congestion_index * 0.6 + (rainfall / 100) * 0.2
        avg_speed = max(10, base_speed * (1 - speed_reduction) + np.random.normal(0, 3))
        
        distance = 10
        travel_time = (distance / avg_speed) * 60
        expected_time = (distance / base_speed) * 60
        delay_minutes = max(0, travel_time - expected_time + np.random.normal(0, 2))
        
        occupancy = np.random.uniform(0.4, 0.95) if is_rush_hour else np.random.uniform(0.2, 0.7)
        passenger_count = int(trans_info['capacity'] * occupancy)
        
        incident_prob = 0.02 + congestion_index * 0.05 + (rainfall / 100) * 0.03
        has_incident = np.random.random() < incident_prob
        incident_type = np.random.choice(['None', 'Minor_Accident', 'Major_Accident', 
                                         'Road_Work', 'Vehicle_Breakdown'],
                                        p=[0.85, 0.08, 0.02, 0.03, 0.02]) if has_incident else 'None'
        
        aqi = 50 + congestion_index * 80 + (1 - humidity/100) * 30 + np.random.normal(0, 10)
        cost = distance * trans_info['cost'] * (1 + congestion_index * 0.3)
        
        record = {
            'Record_ID': f'SL-{i+1:05d}',
            'Timestamp': date,
            'Date': date.strftime('%Y-%m-%d'),
            'Time': date.strftime('%H:%M'),
            'Hour': hour,
            'Day_of_Week': day_of_week,
            'Day_Name': date.strftime('%A'),
            'Month': month,
            'Is_Weekend': int(is_weekend),
            'Is_Rush_Hour': int(is_rush_hour),
            'City': city,
            'City_Type': city_info['type'],
            'Latitude': city_info['lat'] + np.random.normal(0, 0.01),
            'Longitude': city_info['lon'] + np.random.normal(0, 0.01),
            'Transport_Mode': transport,
            'Transport_Category': 'Public' if transport in ['SLTB_Bus', 'Private_Bus', 'Sri_Lanka_Railways'] else 'Private',
            'Vehicle_Capacity': trans_info['capacity'],
            'Weather_Condition': weather,
            'Monsoon_Season': monsoon,
            'Rainfall_mm': round(rainfall, 2),
            'Temperature_C': round(temperature, 1),
            'Humidity_Percent': round(humidity, 1),
            'Congestion_Index': round(congestion_index, 4),
            'Congestion_Level': 'Low' if congestion_index < 0.3 else ('Medium' if congestion_index < 0.6 else ('High' if congestion_index < 0.8 else 'Severe')),
            'Vehicle_Count': vehicle_count,
            'Avg_Speed_kmph': round(avg_speed, 2),
            'Travel_Time_Min': round(travel_time, 2),
            'Delay_Minutes': round(delay_minutes, 2),
            'Passenger_Count': passenger_count,
            'Festival_Period': festival,
            'Incident_Type': incident_type,
            'Has_Incident': int(has_incident),
            'Air_Quality_Index': round(aqi, 1),
            'Estimated_Cost_LKR': round(cost, 2)
        }
        data.append(record)
    
    return pd.DataFrame(data)

# ================================================================================
# ML MODELS
# ================================================================================
@st.cache_resource
def train_ml_models(df):
    """Train Random Forest and Gradient Boosting models"""
    
    le_city = LabelEncoder()
    le_transport = LabelEncoder()
    le_weather = LabelEncoder()
    le_monsoon = LabelEncoder()
    le_festival = LabelEncoder()
    le_congestion = LabelEncoder()
    
    df_ml = df.copy()
    df_ml['City_Enc'] = le_city.fit_transform(df_ml['City'])
    df_ml['Transport_Enc'] = le_transport.fit_transform(df_ml['Transport_Mode'])
    df_ml['Weather_Enc'] = le_weather.fit_transform(df_ml['Weather_Condition'])
    df_ml['Monsoon_Enc'] = le_monsoon.fit_transform(df_ml['Monsoon_Season'])
    df_ml['Festival_Enc'] = le_festival.fit_transform(df_ml['Festival_Period'])
    df_ml['Congestion_Level_Enc'] = le_congestion.fit_transform(df_ml['Congestion_Level'])
    df_ml['Is_Anomaly'] = ((df_ml['Congestion_Index'] > 0.7) & (df_ml['Is_Rush_Hour'] == 0)).astype(int)
    
    features = ['Hour', 'Day_of_Week', 'Month', 'Is_Weekend', 'Is_Rush_Hour',
                'City_Enc', 'Transport_Enc', 'Weather_Enc', 'Monsoon_Enc', 
                'Festival_Enc', 'Rainfall_mm', 'Temperature_C', 'Humidity_Percent', 'Vehicle_Count']
    
    X = df_ml[features]
    y_reg = df_ml['Congestion_Index']
    y_clf = df_ml['Congestion_Level_Enc']
    y_anom = df_ml['Is_Anomaly']
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    _, _, y_train_anom, y_test_anom = train_test_split(X, y_anom, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Train models
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_sc, y_train_reg)
    rf_pred = rf_reg.predict(X_test_sc)
    
    gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42)
    gb_reg.fit(X_train_sc, y_train_reg)
    gb_pred = gb_reg.predict(X_test_sc)
    
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_sc, y_train_clf)
    
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    gb_clf.fit(X_train_sc, y_train_clf)
    
    rf_anom = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_anom.fit(X_train_sc, y_train_anom)
    
    gb_anom = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    gb_anom.fit(X_train_sc, y_train_anom)
    
    rf_cv = cross_val_score(rf_reg, X_train_sc, y_train_reg, cv=5)
    gb_cv = cross_val_score(gb_reg, X_train_sc, y_train_reg, cv=5)
    
    metrics = {
        'rf_reg': {'r2': r2_score(y_test_reg, rf_pred), 
                   'rmse': np.sqrt(mean_squared_error(y_test_reg, rf_pred)),
                   'mae': mean_absolute_error(y_test_reg, rf_pred),
                   'cv_mean': rf_cv.mean(), 'cv_std': rf_cv.std()},
        'gb_reg': {'r2': r2_score(y_test_reg, gb_pred),
                   'rmse': np.sqrt(mean_squared_error(y_test_reg, gb_pred)),
                   'mae': mean_absolute_error(y_test_reg, gb_pred),
                   'cv_mean': gb_cv.mean(), 'cv_std': gb_cv.std()},
        'rf_clf': {'accuracy': accuracy_score(y_test_clf, rf_clf.predict(X_test_sc)),
                   'f1': f1_score(y_test_clf, rf_clf.predict(X_test_sc), average='weighted')},
        'gb_clf': {'accuracy': accuracy_score(y_test_clf, gb_clf.predict(X_test_sc)),
                   'f1': f1_score(y_test_clf, gb_clf.predict(X_test_sc), average='weighted')},
        'rf_anom': {'accuracy': accuracy_score(y_test_anom, rf_anom.predict(X_test_sc))},
        'gb_anom': {'accuracy': accuracy_score(y_test_anom, gb_anom.predict(X_test_sc))}
    }
    
    return {
        'models': {'rf_reg': rf_reg, 'gb_reg': gb_reg, 'rf_clf': rf_clf, 
                   'gb_clf': gb_clf, 'rf_anom': rf_anom, 'gb_anom': gb_anom},
        'scaler': scaler,
        'encoders': {'city': le_city, 'transport': le_transport, 'weather': le_weather,
                     'monsoon': le_monsoon, 'festival': le_festival, 'congestion': le_congestion},
        'features': features,
        'metrics': metrics,
        'feature_importance': pd.DataFrame({
            'Feature': features,
            'RF_Importance': rf_reg.feature_importances_,
            'GB_Importance': gb_reg.feature_importances_
        }).sort_values('RF_Importance', ascending=False),
        'test_data': {'y_test': y_test_reg, 'rf_pred': rf_pred, 'gb_pred': gb_pred},
        'cv_scores': {'rf': rf_cv, 'gb': gb_cv}
    }

def hybrid_predict(models, input_scaled):
    """Combine RF and GB predictions"""
    rf_pred = models['rf_reg'].predict(input_scaled)[0]
    gb_pred = models['gb_reg'].predict(input_scaled)[0]
    hybrid = rf_pred * 0.55 + gb_pred * 0.45
    return {'rf': rf_pred, 'gb': gb_pred, 'hybrid': hybrid, 'confidence': 1 - abs(rf_pred - gb_pred)}
