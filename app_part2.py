# ================================================================================
# MAIN APPLICATION & PAGES
# ================================================================================
def main():
    st.markdown("""
    <div class="main-header">
        🚦 FRAMEWORK FOR INTEGRATING ITS CONCEPTS FOR SRI LANKA<br>
        <span style="font-size: 1rem;">Master's Thesis - Management of Information Systems</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔄 Loading Sri Lanka ITS Dataset..."):
        df = generate_sri_lanka_dataset()
    
    with st.spinner("🤖 Training ML Models..."):
        ml = train_ml_models(df)
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.radio("Select Section", [
        "🏠 Executive Overview",
        "📊 Traffic Analysis",
        "🚌 Multimodal Transport",
        "🌧️ Weather Impact",
        "🤖 ML Performance",
        "🔮 Hybrid Predictions",
        "🛣️ Route Optimization",
        "📈 Future Forecasting",
        "🔍 Data Explorer",
        "📋 Decision Framework",
        "🔗 Data Interoperability",
        "📝 Usability Evaluation"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.write(f"Records: {len(df):,}")
    st.sidebar.write(f"Cities: {df['City'].nunique()}")
    st.sidebar.write(f"RF R²: {ml['metrics']['rf_reg']['r2']:.4f}")
    st.sidebar.write(f"GB R²: {ml['metrics']['gb_reg']['r2']:.4f}")
    
    # Page routing
    if page == "🏠 Executive Overview":
        show_overview(df, ml)
    elif page == "📊 Traffic Analysis":
        show_traffic(df)
    elif page == "🚌 Multimodal Transport":
        show_multimodal(df)
    elif page == "🌧️ Weather Impact":
        show_weather(df)
    elif page == "🤖 ML Performance":
        show_ml_performance(df, ml)
    elif page == "🔮 Hybrid Predictions":
        show_hybrid(df, ml)
    elif page == "🛣️ Route Optimization":
        show_route(df, ml)
    elif page == "📈 Future Forecasting":
        show_forecast(df, ml)
    elif page == "🔍 Data Explorer":
        show_explorer(df)
    elif page == "📋 Decision Framework":
        show_decisions(df, ml)
    elif page == "🔗 Data Interoperability":
        show_interoperability()
    elif page == "📝 Usability Evaluation":
        show_usability()

# OVERVIEW PAGE
def show_overview(df, ml):
    st.markdown('<p class="sub-header">🏠 Executive Overview</p>', unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Congestion", f"{df['Congestion_Index'].mean():.3f}")
    c3.metric("Avg Speed", f"{df['Avg_Speed_kmph'].mean():.1f} km/h")
    c4.metric("Incidents", f"{df['Has_Incident'].sum()}")
    c5.metric("Hybrid R²", f"{(ml['metrics']['rf_reg']['r2']+ml['metrics']['gb_reg']['r2'])/2:.4f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        city_cong = df.groupby('City')['Congestion_Index'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(city_cong, x='City', y='Congestion_Index', title='🏙️ Congestion by City',
                     color='Congestion_Index', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        hourly = df.groupby('Hour')['Congestion_Index'].mean().reset_index()
        fig = px.line(hourly, x='Hour', y='Congestion_Index', title='⏰ Hourly Pattern', markers=True)
        fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.1)
        fig.add_vrect(x0=17, x1=19, fillcolor="red", opacity=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        mode_dist = df['Transport_Mode'].value_counts().reset_index()
        mode_dist.columns = ['Mode', 'Count']
        fig = px.pie(mode_dist, values='Count', names='Mode', title='🚌 Transport Modes', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        weather_cong = df.groupby('Weather_Condition')['Congestion_Index'].mean().reset_index()
        fig = px.bar(weather_cong, x='Weather_Condition', y='Congestion_Index', 
                     title='🌤️ Weather Impact', color='Congestion_Index', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic Map
    st.markdown("### 🗺️ Geographic Heatmap")
    map_data = df.groupby(['City', 'Latitude', 'Longitude'])['Congestion_Index'].mean().reset_index()
    fig = px.scatter_mapbox(map_data, lat='Latitude', lon='Longitude', size='Congestion_Index',
                            color='Congestion_Index', hover_name='City', color_continuous_scale='RdYlGn_r',
                            mapbox_style='open-street-map', zoom=6, center={'lat': 7.8731, 'lon': 80.7718})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# TRAFFIC ANALYSIS PAGE
def show_traffic(df):
    st.markdown('<p class="sub-header">📊 Traffic Congestion Analysis</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cities = st.multiselect("Cities", df['City'].unique(), default=list(df['City'].unique()))
    with col2:
        modes = st.multiselect("Transport", df['Transport_Mode'].unique(), default=list(df['Transport_Mode'].unique()))
    with col3:
        levels = st.multiselect("Congestion Level", df['Congestion_Level'].unique(), default=list(df['Congestion_Level'].unique()))
    
    filtered = df[(df['City'].isin(cities)) & (df['Transport_Mode'].isin(modes)) & (df['Congestion_Level'].isin(levels))]
    st.write(f"**Filtered:** {len(filtered):,} records")
    
    col1, col2 = st.columns(2)
    with col1:
        cong_dist = filtered['Congestion_Level'].value_counts().reset_index()
        cong_dist.columns = ['Level', 'Count']
        colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c', 'Severe': '#8e44ad'}
        fig = px.pie(cong_dist, values='Count', names='Level', title='🚦 Congestion Distribution',
                     color='Level', color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        city_level = filtered.groupby(['City', 'Congestion_Level']).size().reset_index(name='Count')
        fig = px.bar(city_level, x='City', y='Count', color='Congestion_Level',
                     title='📊 Levels by City', color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekly heatmap
    st.markdown("### 📅 Weekly Pattern")
    weekly = filtered.groupby(['Day_Name', 'Hour'])['Congestion_Index'].mean().reset_index()
    weekly_pivot = weekly.pivot(index='Hour', columns='Day_Name', values='Congestion_Index')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pivot = weekly_pivot[[d for d in day_order if d in weekly_pivot.columns]]
    fig = px.imshow(weekly_pivot, color_continuous_scale='RdYlGn_r', title='Weekly Congestion Heatmap')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# MULTIMODAL TRANSPORT PAGE
def show_multimodal(df):
    st.markdown('<p class="sub-header">🚌 Multimodal Transport Coordination</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        mode_stats = df.groupby('Transport_Mode').agg({
            'Passenger_Count': 'sum', 'Delay_Minutes': 'mean', 'Avg_Speed_kmph': 'mean'
        }).reset_index()
        fig = px.bar(mode_stats, x='Transport_Mode', y='Passenger_Count', 
                     title='👥 Passengers by Mode', color='Passenger_Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(mode_stats, x='Transport_Mode', y='Delay_Minutes',
                     title='⏱️ Avg Delay by Mode', color='Delay_Minutes', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sankey diagram
    st.markdown("### 🔄 Passenger Flow (Sankey)")
    city_mode = df.groupby(['City', 'Transport_Mode'])['Passenger_Count'].sum().reset_index()
    top_flows = city_mode.nlargest(15, 'Passenger_Count')
    
    cities_list = top_flows['City'].unique().tolist()
    modes_list = top_flows['Transport_Mode'].unique().tolist()
    all_nodes = cities_list + modes_list
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, label=all_nodes, color=px.colors.qualitative.Set3[:len(all_nodes)]),
        link=dict(
            source=[all_nodes.index(c) for c in top_flows['City']],
            target=[all_nodes.index(m) for m in top_flows['Transport_Mode']],
            value=top_flows['Passenger_Count'].tolist()
        )
    )])
    fig.update_layout(title='City to Transport Mode Flow', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Public vs Private
    cat_hourly = df.groupby(['Transport_Category', 'Hour'])['Passenger_Count'].sum().reset_index()
    fig = px.line(cat_hourly, x='Hour', y='Passenger_Count', color='Transport_Category',
                  title='📊 Hourly Passenger Volume', markers=True)
    st.plotly_chart(fig, use_container_width=True)

# WEATHER IMPACT PAGE
def show_weather(df):
    st.markdown('<p class="sub-header">🌧️ Weather Impact Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        weather_filter = st.multiselect("Weather", df['Weather_Condition'].unique(), 
                                        default=list(df['Weather_Condition'].unique()))
    with col2:
        monsoon_filter = st.multiselect("Monsoon", df['Monsoon_Season'].unique(),
                                        default=list(df['Monsoon_Season'].unique()))
    
    filtered = df[(df['Weather_Condition'].isin(weather_filter)) & (df['Monsoon_Season'].isin(monsoon_filter))]
    
    col1, col2 = st.columns(2)
    with col1:
        weather_cong = filtered.groupby('Weather_Condition').agg({
            'Congestion_Index': 'mean', 'Delay_Minutes': 'mean'
        }).reset_index()
        fig = px.bar(weather_cong, x='Weather_Condition', y=['Congestion_Index', 'Delay_Minutes'],
                     title='🌤️ Weather Impact', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(filtered, x='Rainfall_mm', y='Congestion_Index', color='Weather_Condition',
                         title='🌧️ Rainfall vs Congestion', trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
    
    # Monsoon analysis
    monsoon_stats = filtered.groupby('Monsoon_Season').agg({
        'Congestion_Index': 'mean', 'Rainfall_mm': 'mean', 'Delay_Minutes': 'mean'
    }).reset_index()
    fig = px.bar(monsoon_stats, x='Monsoon_Season', y=['Congestion_Index', 'Rainfall_mm'],
                 title='🌊 Monsoon Season Analysis', barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### 📊 Correlation Matrix")
    vars_corr = ['Rainfall_mm', 'Temperature_C', 'Humidity_Percent', 'Congestion_Index', 
                 'Avg_Speed_kmph', 'Delay_Minutes', 'Air_Quality_Index']
    corr = filtered[vars_corr].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title='Variable Correlations')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ML PERFORMANCE PAGE
def show_ml_performance(df, ml):
    st.markdown('<p class="sub-header">🤖 ML Model Performance</p>', unsafe_allow_html=True)
    
    st.markdown("### 📈 Regression: Congestion Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌲 Random Forest")
        st.metric("R² Score", f"{ml['metrics']['rf_reg']['r2']:.4f}")
        st.metric("RMSE", f"{ml['metrics']['rf_reg']['rmse']:.4f}")
        st.metric("MAE", f"{ml['metrics']['rf_reg']['mae']:.4f}")
        st.metric("CV Score", f"{ml['metrics']['rf_reg']['cv_mean']:.4f} ± {ml['metrics']['rf_reg']['cv_std']:.4f}")
    
    with col2:
        st.markdown("#### 🚀 Gradient Boosting")
        st.metric("R² Score", f"{ml['metrics']['gb_reg']['r2']:.4f}")
        st.metric("RMSE", f"{ml['metrics']['gb_reg']['rmse']:.4f}")
        st.metric("MAE", f"{ml['metrics']['gb_reg']['mae']:.4f}")
        st.metric("CV Score", f"{ml['metrics']['gb_reg']['cv_mean']:.4f} ± {ml['metrics']['gb_reg']['cv_std']:.4f}")
    
    # Actual vs Predicted
    st.markdown("### 📊 Actual vs Predicted")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(x=ml['test_data']['y_test'], y=ml['test_data']['rf_pred'],
                         title='Random Forest', labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(x=ml['test_data']['y_test'], y=ml['test_data']['gb_pred'],
                         title='Gradient Boosting', labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        fi = ml['feature_importance'].sort_values('RF_Importance')
        fig = px.bar(fi, x='RF_Importance', y='Feature', orientation='h',
                     title='RF Importance', color='RF_Importance', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fi = ml['feature_importance'].sort_values('GB_Importance')
        fig = px.bar(fi, x='GB_Importance', y='Feature', orientation='h',
                     title='GB Importance', color='GB_Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation
    st.markdown("### 📦 Cross-Validation")
    cv_df = pd.DataFrame({
        'Model': ['RF']*5 + ['GB']*5,
        'Fold': list(range(1,6))*2,
        'Score': list(ml['cv_scores']['rf']) + list(ml['cv_scores']['gb'])
    })
    fig = px.box(cv_df, x='Model', y='Score', color='Model', title='5-Fold CV Scores')
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification metrics
    st.markdown("### 📋 Classification Performance")
    clf_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'F1-Score'],
        'Random Forest': [ml['metrics']['rf_clf']['accuracy'], ml['metrics']['rf_clf']['f1']],
        'Gradient Boosting': [ml['metrics']['gb_clf']['accuracy'], ml['metrics']['gb_clf']['f1']]
    })
    fig = px.bar(clf_metrics, x='Metric', y=['Random Forest', 'Gradient Boosting'], barmode='group',
                 title='Classification Metrics')
    st.plotly_chart(fig, use_container_width=True)
