"""
================================================================================
ENHANCED ITS FRAMEWORK FOR SRI LANKA - PROFESSIONAL DASHBOARD
================================================================================
Master's Thesis - Management of Information Systems
Advanced Streamlit Dashboard with Real-Time Monitoring and Alerts
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================
st.set_page_config(
    page_title="MTDT System - Sri Lanka ITS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CUSTOM CSS - PROFESSIONAL STYLING
# ================================================================================
st.markdown("""
<style>
    /* Main Container */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Alert Box Styling */
    .alert-box {
        background: #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .alert-critical {
        border-left-color: #e74c3c;
        background: #34495e;
    }
    
    .alert-warning {
        border-left-color: #f39c12;
        background: #34495e;
    }
    
    .alert-info {
        border-left-color: #3498db;
        background: #34495e;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    /* Filter Panel */
    .filter-panel {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* Toggle Switch Styling */
    .stCheckbox {
        background: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        background: #27ae60;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Device Status */
    .device-online {
        color: #27ae60;
        font-weight: bold;
    }
    
    .device-offline {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# DATA GENERATION WITH ENHANCED FEATURES
# ================================================================================
@st.cache_data
def generate_enhanced_dataset():
    """Generate Sri Lanka ITS dataset with additional fields for mockups"""
    np.random.seed(42)
    n = 2000
    
    cities = ['Colombo', 'Kandy', 'Galle', 'Jaffna', 'Negombo', 
              'Kurunegala', 'Ratnapura', 'Anuradhapura', 'Trincomalee', 'Batticaloa']
    roads = ['Baseline Road', 'Galle Road', 'Kandy Road', 'Marine Drive', 
             'Bauddhaloka Mawatha', 'Dutugemunu Street', 'A1 Highway', 'A9 Highway']
    transport_modes = ['SLTB Bus', 'Private Bus', 'Sri Lanka Railways', 
                      'Three-Wheeler', 'Private Vehicle', 'Motorcycle']
    weather = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Thunderstorm']
    
    data = {
        'Timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 10080)) for _ in range(n)],
        'City': np.random.choice(cities, n),
        'Road': np.random.choice(roads, n),
        'Transport_Mode': np.random.choice(transport_modes, n),
        'Weather': np.random.choice(weather, n),
        'Rainfall_mm': np.random.exponential(5, n),
        'Temperature_C': np.random.normal(28, 4, n),
        'Humidity_Percent': np.random.uniform(60, 95, n),
        'Vehicle_Count': np.random.randint(50, 500, n),
        'Avg_Speed_kmph': np.random.normal(35, 15, n),
        'Has_Incident': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    
    # Feature Engineering
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Is_Rush_Hour'] = df['Hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Congestion Index
    base_congestion = (df['Vehicle_Count'] / 500) * 0.4
    weather_impact = df['Weather'].map({'Clear': 0, 'Cloudy': 0.05, 'Light Rain': 0.15, 
                                       'Heavy Rain': 0.25, 'Thunderstorm': 0.35})
    speed_impact = (1 - df['Avg_Speed_kmph'] / 80) * 0.3
    incident_impact = df['Has_Incident'] * 0.2
    df['Congestion_Index'] = (base_congestion + weather_impact + speed_impact + incident_impact).clip(0, 1)
    
    # Congestion Level
    df['Congestion_Level'] = pd.cut(df['Congestion_Index'], 
                                    bins=[0, 0.3, 0.6, 1.0],
                                    labels=['Low', 'Medium', 'High'])
    
    # Device IDs for IoT monitoring
    df['Device_ID'] = [f"T{np.random.randint(100, 999)}.{np.random.randint(100, 999)}" for _ in range(n)]
    df['Device_Status'] = np.random.choice(['Online', 'Offline'], n, p=[0.95, 0.05])
    df['Data_Throughput_KBs'] = np.random.uniform(5, 100, n)
    
    # Train/Bus data for synchronization
    df['Train_Route'] = np.random.choice(['6037.58', '8.048.75', '2.008.59'], n)
    df['Bus_Route'] = np.random.choice(['32B', '177', '138'], n)
    df['Platform_Gate'] = np.random.choice(['1:54.55', '6:01.08', '8:00.35'], n)
    
    return df

# ================================================================================
# ML MODELS
# ================================================================================
@st.cache_resource
def train_ml_models(df):
    """Train Random Forest and Gradient Boosting models"""
    
    # Prepare features
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in ['City', 'Road', 'Transport_Mode', 'Weather']:
        le = LabelEncoder()
        df_encoded[col + '_Encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    feature_cols = ['Hour', 'DayOfWeek', 'Is_Weekend', 'Is_Rush_Hour',
                   'City_Encoded', 'Transport_Mode_Encoded', 'Weather_Encoded',
                   'Rainfall_mm', 'Temperature_C', 'Humidity_Percent', 'Vehicle_Count']
    
    X = df_encoded[feature_cols]
    y = df_encoded['Congestion_Index']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Metrics
    metrics = {
        'rf': {
            'r2': r2_score(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'mae': mean_absolute_error(y_test, rf_pred)
        },
        'gb': {
            'r2': r2_score(y_test, gb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'mae': mean_absolute_error(y_test, gb_pred)
        }
    }
    
    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'rf_pred': rf_pred,
        'gb_pred': gb_pred
    }

# ================================================================================
# ALERT GENERATION
# ================================================================================
def generate_alerts(df):
    """Generate realistic alerts based on current data"""
    alerts = []
    
    # High congestion alerts
    high_cong = df[df['Congestion_Index'] > 0.7].sample(min(3, len(df[df['Congestion_Index'] > 0.7])))
    for _, row in high_cong.iterrows():
        alerts.append({
            'level': 'ALERT',
            'message': f"High congestion predicted on {row['Road']} at {row['Hour']}:00",
            'color': '#e74c3c'
        })
    
    # Weather warnings
    heavy_rain = df[df['Weather'] == 'Heavy Rain'].sample(min(2, len(df[df['Weather'] == 'Heavy Rain'])))
    for _, row in heavy_rain.iterrows():
        alerts.append({
            'level': 'WARNING',
            'message': f"Delay forecast {row['Road']} ({row['Hour']}:00 - {(row['Hour']+2)%24}:00)",
            'color': '#f39c12'
        })
    
    # Normal traffic
    alerts.append({
        'level': 'INFO',
        'message': f"Normal traffic expected Marine Drive Raod",
        'color': '#3498db'
    })
    
    return alerts

# ================================================================================
# PAGE 1: CONGESTION PREDICTION & ALERT INTERFACE
# ================================================================================
def page_congestion_prediction(df, ml_models):
    st.markdown("""
    <div class="dashboard-header">
        <h1>🚦 Congestion Prediction & Alert Interface</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Predicted Congestion Probability Over Time")
        
        # Generate prediction time series
        hours = list(range(24))
        predictions = [0.2 + 0.3 * np.sin(h/3) + np.random.uniform(-0.1, 0.1) for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=predictions,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#3498db', width=3),
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        
        fig.update_layout(
            xaxis_title="Time (Hours)",
            yaxis_title="Congestion Probability",
            height=350,
            template='plotly_white',
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 24])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Confidence
        confidence = ml_models['metrics']['rf']['r2'] * 100
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; text-align: center;">
            <strong>AI Model Confidence: {confidence:.0f}%</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Filters
        st.markdown("### Filters")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Filter by Road:", ["All Roads"] + df['Road'].unique().tolist())
        with c2:
            st.selectbox("Filter by Date/Time Range:", ["Last 24 Hours", "Last Week", "Last Month"])
        
        live_feed = st.checkbox("Live Data Feed", value=True)
    
    with col2:
        st.markdown("### Alert Log")
        
        alerts = generate_alerts(df)
        
        for alert in alerts:
            level_class = "alert-critical" if alert['level'] == 'ALERT' else \
                         "alert-warning" if alert['level'] == 'WARNING' else "alert-info"
            
            st.markdown(f"""
            <div class="alert-box {level_class}">
                <strong>{alert['level']}:</strong> {alert['message']}
            </div>
            """, unsafe_allow_html=True)

# ================================================================================
# PAGE 2: TRAIN-BUS SYNCHRONIZATION SCREEN
# ================================================================================
def page_train_bus_sync(df):
    st.markdown("""
    <div class="dashboard-header">
        <h1>🚆 Train-Bus Synchronization Screen</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1.5, 1])
    
    with col1:
        st.markdown("### Scheduled Connections")
        
        # Sample schedule data
        schedule_data = {
            'Train Arrival Time': ['1:34.313', '1:38.313', '1:31.315', '1:95.314'],
            'Bus Departure': ['8:037.58', '8.048.59', '2.107.50', '2.013.58'],
            'Connecting Departure': ['6:037.58', '8.048.59', '2.107.50', '2.013.58'],
            'Bus Route #': ['6037.58', '8.048.59', '2.107.50', '2.013.58'],
            'Platform/Gate': ['1:54.55', '6:01.08', '8:00.35', '8:00.08']
        }
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True, height=200)
        
        # Additional rows
        more_data = {
            'Train Arrival Time': ['3:07.313', '8:73.314', '8:96.313', '8:88.218'],
            'Bus Departure': ['8.972.58', '6.038.48', '6.097.48', '8.902.59'],
            'Connecting Departure': ['8.972.58', '6.038.48', '6.097.48', '8.902.59'],
            'Bus Route #': ['8.972.58', '6.038.48', '6.097.48', '8.902.59'],
            'Platform/Gate': ['6:08.5B', '7:04.85', '7:05.65', '6:59.04']
        }
        
        more_df = pd.DataFrame(more_data)
        st.dataframe(more_df, use_container_width=True, height=200)
    
    with col2:
        st.markdown("### Delay Visualization (Minutes)")
        
        # Delay comparison
        delay_data = {
            'Day': ['Normal Day', 'Rainy Day'],
            'Average': [40, 90],
            'Peak': [15, 60]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Average', x=delay_data['Day'], y=delay_data['Average'], 
                            marker_color='#34495e'))
        fig.add_trace(go.Bar(name='Peak', x=delay_data['Day'], y=delay_data['Peak'], 
                            marker_color='#95a5a6'))
        
        fig.update_layout(
            barmode='group',
            height=300,
            template='plotly_white',
            yaxis_title="Delay (Minutes)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### Alert!")
        st.markdown("""
        <div class="alert-box alert-critical">
            <strong>Mismatched Schedule Detected:</strong><br>
            Train 1450 (8:10) & Bus 32B (8:05 AM)
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Resolve", use_container_width=True)
        st.button("Auto-Adjust", use_container_width=True)
    
    # Filters
    st.markdown("### Filters")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Filter by Road:", ["All Roads"])
    with c2:
        st.selectbox("Filter by Date/Time Range:", ["Last 24 Hours"])
    
    st.checkbox("Live Data Feed", value=True)

# ================================================================================
# PAGE 3: WEATHER-DRIVEN ROUTE OPTIMIZATION
# ================================================================================
def page_route_optimization(df):
    st.markdown("""
    <div class="dashboard-header">
        <h1>🌧️ Weather-Driven Route Optimization Panel</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Weather Feed")
        
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Rainfall:", value="15 mm", key="rainfall_1")
            st.text_input("Humidity:", value="85%", key="humidity_1")
        with c2:
            st.text_input("Temperature:", value="28°C", key="temp_1")
            st.text_input("Wind Speed:", value="12 km/h", key="wind_1")
        
        st.markdown("**— Bus Route —**")
        
        st.markdown("### Route Suggestions")
        st.markdown("**Driver Route:**")
        st.selectbox("Filter bo Date/Time Range:", ["Route A (Baseline Rd)", "Route B (Galle Rd)"])
        
        st.markdown("**Select Start:**")
        st.selectbox("Filter Destination:", ["Colombo Fort", "Mount Lavinia", "Dehiwala"])
    
    with col2:
        st.markdown("### Travel Time Comparison (Mins)")
        
        # Travel time comparison
        routes = ['Route A\n(Baseline Rd)', 'Route B\n(Galle Rd)']
        normal_times = [35, 10]
        rainy_times = [105, 60]
        
        fig = go.Figure()
        
        x_pos = [0, 1]
        colors_normal = ['#34495e', '#34495e']
        colors_rainy = ['#e74c3c', '#95a5a6']
        
        fig.add_trace(go.Bar(
            name='Normal Day',
            x=['Route A', 'Route B'],
            y=normal_times,
            marker_color=colors_normal,
            text=normal_times,
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Rainy Day',
            x=['Route A', 'Route B'],
            y=rainy_times,
            marker_color=colors_rainy,
            text=rainy_times,
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            height=350,
            template='plotly_white',
            yaxis_title="Travel Time (Minutes)",
            yaxis=dict(range=[0, 120]),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        simulate = st.checkbox("Simulate Rainy Conditions", value=False)
        st.button("Run Optimization", use_container_width=True)

# ================================================================================
# PAGE 4: REAL-TIME TRAFFIC PATTERN VISUALIZATION
# ================================================================================
def page_traffic_visualization(df):
    st.markdown("""
    <div class="dashboard-header">
        <h1>📍 Real-Time Traffic Pattern Visualization</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### City Traffic Map")
        
        # Create a scatter map visualization
        city_coords = {
            'Colombo': (6.9271, 79.8612),
            'Kandy': (7.2906, 80.6337),
            'Galle': (6.0535, 80.2210),
            'Jaffna': (9.6615, 80.0255),
            'Negombo': (7.2008, 79.8358)
        }
        
        map_data = []
        for city, (lat, lon) in city_coords.items():
            city_data = df[df['City'] == city]
            avg_congestion = city_data['Congestion_Index'].mean()
            map_data.append({
                'City': city,
                'Lat': lat,
                'Lon': lon,
                'Congestion': avg_congestion,
                'Size': avg_congestion * 50
            })
        
        map_df = pd.DataFrame(map_data)
        
        fig = px.scatter_mapbox(
            map_df,
            lat='Lat',
            lon='Lon',
            size='Size',
            color='Congestion',
            hover_name='City',
            hover_data={'Congestion': ':.2f', 'Lat': False, 'Lon': False, 'Size': False},
            color_continuous_scale='Reds',
            zoom=7,
            height=450
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Average Speed vs Time")
        
        hourly_speed = df.groupby('Hour')['Avg_Speed_kmph'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_speed.index,
            y=hourly_speed.values,
            mode='lines',
            line=dict(color='#2c3e50', width=2),
            fill='tozeroy',
            fillcolor='rgba(44, 62, 80, 0.2)'
        ))
        
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Speed (km/h)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Filters
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Filter by Road:", ["All Roads"] + df['Road'].unique().tolist())
    with c2:
        st.selectbox("Filter by Date/Time Range:", ["Last Hour", "Last 24 Hours"])
    
    st.checkbox("Live Data Feed", value=True)

# ================================================================================
# PAGE 5: CITY-WIDE SIMULATION (DIGITAL TWIN)
# ================================================================================
def page_digital_twin(df):
    st.markdown("""
    <div class="dashboard-header">
        <h1>🗺️ City-Wide Simulation View (Digital Twin Map)</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.markdown("### Digital Twin - Colombo Metropolitan Area")
        
        # Create enhanced map with multiple layers
        fig = go.Figure()
        
        # Add city boundary
        fig.add_trace(go.Scattermapbox(
            lat=[6.85, 6.85, 7.0, 7.0, 6.85],
            lon=[79.8, 80.0, 80.0, 79.8, 79.8],
            mode='lines',
            line=dict(width=2, color='#2c3e50'),
            name='City Boundary'
        ))
        
        # Add traffic density points
        np.random.seed(42)
        n_points = 30
        lats = np.random.uniform(6.85, 7.0, n_points)
        lons = np.random.uniform(79.8, 80.0, n_points)
        densities = np.random.uniform(0.3, 0.9, n_points)
        
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=15,
                color=densities,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Traffic<br>Density")
            ),
            text=[f"Density: {d:.2f}" for d in densities],
            name='Traffic Density'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=6.927, lon=79.861),
                zoom=11
            ),
            height=500,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline slider
        st.slider("Timeline Slider", 0, 24, 12, format="%d:00")
    
    with col2:
        st.markdown("### Legend")
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px;">
            <p>🚌 <strong>Buses</strong></p>
            <p>🚂 <strong>Trains</strong></p>
            <p>📊 <strong>Traffic Density</strong></p>
            <p>🌦️ <strong>Weather</strong></p>
            <hr>
            <p>🌡️ <strong>Emission Levels</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Filters
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Filter by Road:", ["All Roads"])
    with c2:
        st.selectbox("Filter by Date/Time Range:", ["Current", "Last Hour"])

# ================================================================================
# PAGE 6: ALERT MANAGEMENT & NOTIFICATION PANEL
# ================================================================================
def page_alert_management(df):
    st.markdown("""
    <div class="dashboard-header">
        <h1>🔔 Alert Management & Notification Panel</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Alert Log")
        
        # Tabs for alert levels
        tab1, tab2, tab3 = st.tabs(["Critical", "Moderate", "Low"])
        
        with tab1:
            st.markdown("""
            <div class="alert-box alert-critical">
                <strong>● Critical:</strong> Train 1450 delayed 15 min ctuo signal fault signal fault
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>● Moderate:</strong> Bus 32B rerouted bsue to-duo roadwork
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div class="alert-box alert-info">
                <strong>● Low:</strong> High traffic density predicted Baseline Rd Baseline Rd
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action buttons
        c1, c2, c3 = st.columns(3)
        with c2:
            st.button("Achowlegee", use_container_width=True)
        with c3:
            st.button("Resolve", use_container_width=True)
        
        st.checkbox("Integrate with Email/SMS Notifications", value=False)
    
    with col2:
        st.markdown("### System Status")
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px;">
            <h4>Active Alerts: 3</h4>
            <p><strong class="device-online">● Critical: 1</strong></p>
            <p><strong style="color: #f39c12;">● Moderate: 1</strong></p>
            <p><strong style="color: #3498db;">● Low: 1</strong></p>
            <hr>
            <p><strong>Last Update:</strong> 2 mins ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Filters
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Filter by Road:", ["All Roads"])
    with c2:
        st.selectbox("Filter by Date/Time Range:", ["Last Hour"])
    
    st.checkbox("Live Data Feed", value=True)

# ================================================================================
# PAGE 7: SENSOR AND IOT DEVICE STATUS
# ================================================================================
def page_iot_devices(df):
    st.markdown("""
    <div class="dashboard-header">
        <h1>📡 Sensor and IOT Device Status Page</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1.5, 1])
    
    with col1:
        st.markdown("### Device List")
        
        device_data = df[['Device_ID', 'Device_Status', 'Bus_Route', 'Platform_Gate']].head(10).copy()
        device_data.columns = ['Device ID', 'Status (Online)', 'Bus Rout (Neilbrs)', 'Platform/Gate']
        
        # Style the status
        def color_status(val):
            color = '#27ae60' if val == 'Online' else '#e74c3c'
            return f'color: {color}; font-weight: bold'
        
        styled_df = device_data.style.applymap(color_status, subset=['Status (Online)'])
        st.dataframe(styled_df, use_container_width=True, height=350)
    
    with col2:
        st.markdown("### Data Throughput Rate (KB/s)")
        
        # Time series of data throughput
        hours = list(range(24))
        throughput = [50 + 30 * np.sin(h/2) + np.random.uniform(-5, 5) for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=throughput,
            mode='lines',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Throughput (KB/s)",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### System Alerts")
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>WARNING:</strong> Missing data from Sensor T-101 (Galle Road)
        </div>
        <div class="alert-box alert-critical">
            <strong>ALERT:</strong> Device D-345 offline for 15 minutes (Not: At Systems Normal)
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Resolve", use_container_width=True)
        st.button("Auto-Adjust", use_container_width=True)
    
    # Filters
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Filter by Road:", ["All Roads"])
    with c2:
        st.selectbox("Filter by Date/Time Range:", ["Last Hour"])
    
    st.checkbox("Live Data Feed", value=True)

# ================================================================================
# PAGE 8: DATA ANALYTICS & REPORTS
# ================================================================================
def page_analytics(df, ml_models):
    st.markdown("""
    <div class="dashboard-header">
        <h1>📊 Data Analytics & Reports Page</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Correlation Matrix")
        
        # Create correlation matrix
        numeric_cols = ['Rainfall_mm', 'Temperature_C', 'Humidity_Percent', 
                       'Vehicle_Count', 'Avg_Speed_kmph', 'Congestion_Index']
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Rainfall', 'Temp', 'Humidity', 'Vehicles', 'Speed', 'Congestion'],
            y=['Rainfall', 'Temp', 'Humidity', 'Vehicles', 'Speed', 'Congestion'],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Rainfall Matrix",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Regression Model Outputs")
        
        rf_metrics = ml_models['metrics']['rf']
        gb_metrics = ml_models['metrics']['gb']
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4>Model: Random Forest</h4>
            <p><strong>Mean Absolute Error (MAE):</strong> {rf_metrics['mae']:.2f}</p>
            <p><strong>Root a Smurale Error (RMSE):</strong> {rf_metrics['rmse']:.2f}</p>
            <p><strong>R-squred (R²):</strong> {rf_metrics['r2']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Export Options")
        c1, c2 = st.columns(2)
        with c1:
            st.button("Generate PDF Report", use_container_width=True)
        with c2:
            st.button("Download Excel Data", use_container_width=True)
    
    # Filters
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Filter by Road:", ["All Roads"])
    with c2:
        st.selectbox("Filter by Date/Time Range:", ["Last Month"])
    
    st.checkbox("Live Data Feed", value=False)

# ================================================================================
# MAIN APPLICATION
# ================================================================================
def main():
    # Load data
    with st.spinner("🔄 Loading Enhanced ITS Dataset..."):
        df = generate_enhanced_dataset()
    
    # Train models
    with st.spinner("🤖 Training ML Models..."):
        ml_models = train_ml_models(df)
    
    # Sidebar Navigation
    st.sidebar.title("🚦 MTDT System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", [
        "🚦 Congestion Prediction",
        "🚆 Train-Bus Sync",
        "🌧️ Route Optimization",
        "📍 Traffic Visualization",
        "🗺️ Digital Twin Map",
        "🔔 Alert Management",
        "📡 IOT Device Status",
        "📊 Analytics & Reports"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.success("● All Systems Online")
    st.sidebar.info(f"Active Devices: {df['Device_Status'].value_counts().get('Online', 0)}")
    st.sidebar.warning(f"Alerts: {len(generate_alerts(df))}")
    
    # Page Routing
    if page == "🚦 Congestion Prediction":
        page_congestion_prediction(df, ml_models)
    elif page == "🚆 Train-Bus Sync":
        page_train_bus_sync(df)
    elif page == "🌧️ Route Optimization":
        page_route_optimization(df)
    elif page == "📍 Traffic Visualization":
        page_traffic_visualization(df)
    elif page == "🗺️ Digital Twin Map":
        page_digital_twin(df)
    elif page == "🔔 Alert Management":
        page_alert_management(df)
    elif page == "📡 IOT Device Status":
        page_iot_devices(df)
    elif page == "📊 Analytics & Reports":
        page_analytics(df, ml_models)

if __name__ == "__main__":
    main()
