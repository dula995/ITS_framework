# 🚦 Framework for Integrating Intelligent Transportation Systems for Sri Lanka

## Master's Thesis - Management of Information Systems

---

## 📋 Overview

This **Streamlit dashboard** provides a comprehensive Intelligent Transportation System (ITS) framework for Sri Lanka with:

- **2000+ synthetic records** based on realistic Sri Lankan patterns
- **Machine Learning models** (Random Forest & Gradient Boosting)
- **Hybrid prediction system** combining both algorithms
- **Interactive visualizations** with filtering and sorting
- **Decision-making framework** for transportation authorities
- **Usability evaluation** with heuristic analysis

---

## 🎯 Features

### 1. Sri Lanka Native Dataset
- **10 Major Cities**: Colombo, Kandy, Galle, Jaffna, Negombo, Kurunegala, Ratnapura, Anuradhapura, Trincomalee, Batticaloa
- **6 Transport Modes**: SLTB Bus, Private Bus, Sri Lanka Railways, Three-Wheeler, Private Vehicle, Motorcycle
- **Weather Patterns**: Clear, Cloudy, Light Rain, Heavy Rain, Thunderstorm
- **Monsoon Seasons**: Southwest, Northeast, Inter-Monsoon, Dry
- **Festivals**: Sinhala Tamil New Year, Vesak, Poson, Esala Perahera, Deepavali, Christmas

### 2. Machine Learning Models
| Model | Task | Metrics |
|-------|------|---------|
| Random Forest Regressor | Congestion prediction | R², RMSE, MAE |
| Gradient Boosting Regressor | Congestion prediction | R², RMSE, MAE |
| Random Forest Classifier | Level classification | Accuracy, F1 |
| Gradient Boosting Classifier | Level classification | Accuracy, F1 |
| Anomaly Detectors | Unusual congestion | Precision, Recall |

### 3. Dashboard Pages
1. **Executive Overview** - KPIs, geographic heatmap, trends
2. **Traffic Analysis** - Filtering, congestion patterns, weekly heatmap
3. **Multimodal Transport** - Passenger flow, Sankey diagram, mode analysis
4. **Weather Impact** - Monsoon analysis, correlation matrix
5. **ML Performance** - Actual vs predicted, feature importance, cross-validation
6. **Hybrid Predictions** - Interactive prediction with recommendations
7. **Route Optimization** - Weather-driven routing suggestions
8. **Future Forecasting** - Scenario-based predictions
9. **Data Explorer** - Filter, sort, download data
10. **Decision Framework** - Action matrices for authorities
11. **Data Interoperability** - API specs, standards compliance
12. **Usability Evaluation** - Nielsen's heuristics, user test results

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.8+
- pip package manager

### Steps

```bash
# 1. Clone/Extract the project
cd its_thesis_final

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py

# 4. Open browser
# http://localhost:8501
```

---

## 📁 Project Structure

```
its_thesis_final/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── USABILITY_REPORT.md    # Detailed usability analysis
```

---

## 📊 User Interactions

| Feature | Type | Description |
|---------|------|-------------|
| City Filter | Multi-select | Filter by Sri Lankan cities |
| Transport Filter | Multi-select | Filter by transport modes |
| Date/Time | Sliders | Select time periods |
| Weather Filter | Multi-select | Filter by weather conditions |
| Prediction Input | Sliders/Selects | Configure prediction parameters |
| Sort | Dropdown + Radio | Sort data columns |
| Download | Button | Export filtered data as CSV |

---

## 🤖 Hybrid Prediction System

The dashboard combines Random Forest and Gradient Boosting predictions:

```
Hybrid = (RF × 0.55) + (GB × 0.45)
```

**Features used:**
- Hour, Day of Week, Month
- Weekend/Rush Hour indicators
- City, Transport Mode, Weather
- Monsoon Season, Festival Period
- Rainfall, Temperature, Humidity
- Vehicle Count

---

## 📈 Model Performance Summary

| Metric | Random Forest | Gradient Boosting |
|--------|--------------|-------------------|
| R² Score | ~0.85+ | ~0.83+ |
| RMSE | ~0.07 | ~0.08 |
| Classification Accuracy | ~85%+ | ~83%+ |
| Cross-Validation | Stable | Stable |

---

## 🎨 Visualizations Included

1. **Bar Charts**: Congestion by city, weather impact
2. **Line Charts**: Hourly patterns, forecasts
3. **Pie Charts**: Transport mode distribution
4. **Heatmaps**: Weekly patterns, correlations
5. **Scatter Plots**: Actual vs predicted, rainfall impact
6. **Geographic Maps**: Hub activity heatmap
7. **Sankey Diagrams**: Passenger flow
8. **Box Plots**: Cross-validation scores

---

## 📝 Usability Score

**Overall: 8.4/10** (System Usability Scale: 78/100 - Good)

| Heuristic | Score |
|-----------|-------|
| System Status Visibility | 9/10 |
| Real World Match | 9/10 |
| Consistency | 9/10 |
| Error Prevention | 8/10 |
| Recognition vs Recall | 9/10 |
| Flexibility | 8/10 |

---

## 🇱🇰 Sri Lanka Context

This framework is specifically designed for Sri Lanka's transportation challenges:

- **Monsoon patterns** affecting road conditions
- **Festival traffic** during cultural celebrations
- **Multi-modal integration** (SLTB, Railways, private transport)
- **Urban congestion** especially in Colombo metropolitan area
- **Weather-driven routing** for tropical conditions

---

## 📚 Academic Use

Suitable for:
- Master's thesis in Management Information Systems
- ITS research and development
- Smart city planning projects
- Transportation policy analysis

---

## 👤 Author

**MIS Master's Student**  
**University**: Sri Lanka  
**Year**: 2025

---

## 📄 License

Academic use only. For thesis and research purposes.
