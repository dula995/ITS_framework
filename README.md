# 🚦 ITS Dashboard - Sri Lanka (Streamlit)
## Master's Thesis - Management Information Systems

---

## 📋 Overview

This Streamlit dashboard provides interactive visualization and analysis for the **Intelligent Transportation Systems (ITS) Framework for Sri Lanka**. It includes:

- **Filtering**: Date, City, Mode, Weather, Time Period, Congestion Level
- **Sorting**: Multiple columns with ascending/descending order
- **User Input**: Traffic prediction simulator
- **Data Export**: Download filtered data as CSV
- **Interactive Visualizations**: Maps, charts, heatmaps

---

## 🚀 How to Run

### Step 1: Install Python
Make sure you have Python 3.8+ installed.

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Step 3: Prepare Files

Ensure these files are in the **SAME FOLDER**:
```
📁 streamlit_its_dashboard/
   ├── app.py
   ├── requirements.txt
   ├── sri_lanka_its_synthetic_dataset_v2.csv
   └── USABILITY_EVALUATION_REPORT.md
```

### Step 4: Run the Dashboard

Open terminal/command prompt and navigate to the folder:

```bash
cd path/to/streamlit_its_dashboard
```

Then run:
```bash
streamlit run app.py
```

### Step 5: View Dashboard

The dashboard will automatically open in your browser at:
```
http://localhost:8501
```

---

## 🎛️ Dashboard Features

### Sidebar Filters
| Filter | Description |
|--------|-------------|
| 📅 Date Range | Select start and end dates |
| 🏙️ Origin City | Filter by departure city |
| 🏙️ Destination City | Filter by arrival city |
| 🚗 Transport Mode | Select Bus, Train, Car, etc. |
| ⏰ Time Period | Morning, Afternoon, Evening, Night |
| 🚦 Congestion Range | Slider 0-100 |
| 🌧️ Weather | Clear, Light Rain, Heavy Rain |
| ⚡ Rush Hour | All / Rush Hour Only / Non-Rush |
| 📆 Day Type | All / Weekdays / Weekends |

### Sorting Options
- Sort by: Congestion, Delay, Travel Time, Distance, Passengers
- Order: Ascending / Descending

### Tabs
1. **📈 Overview** - Key metrics and patterns
2. **🗺️ Geographic Map** - Interactive Sri Lanka map
3. **🚦 Congestion Analysis** - Detailed congestion insights
4. **🌧️ Weather Impact** - Weather vs traffic analysis
5. **🚂 Multimodal Analysis** - Transport mode comparison
6. **📋 Data Explorer** - Raw data with custom columns

### Prediction Simulator
Enter parameters to predict congestion:
- Origin & Destination City
- Hour of Day
- Transport Mode
- Expected Rainfall
- Day of Week

---

## 📊 Screenshots

### Main Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│  🚦 INTELLIGENT TRANSPORTATION SYSTEMS DASHBOARD            │
│     Sri Lanka - Master's Thesis MIS                         │
├─────────────────────────────────────────────────────────────┤
│  📁 Total    │  🚦 Avg      │  ⏱️ Avg     │  👥 Total      │
│  Records     │  Congestion  │  Delay      │  Passengers    │
│  5,000       │  47.2        │  89.5 min   │  125,430       │
├─────────────────────────────────────────────────────────────┤
│  [📈 Overview] [🗺️ Map] [🚦 Congestion] [🌧️ Weather]      │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  Hourly Chart   │  │  Distribution   │                  │
│  │                 │  │                 │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `sri_lanka_its_synthetic_dataset_v2.csv` | ITS dataset (5,000 records) |
| `USABILITY_EVALUATION_REPORT.md` | Heuristic analysis document |
| `README.md` | This file |

---

## ⚠️ Troubleshooting

### Error: "ModuleNotFoundError"
```bash
pip install streamlit pandas numpy plotly
```

### Error: "FileNotFoundError"
Make sure the CSV file is in the same folder as `app.py`

### Dashboard not opening
Try manually opening: `http://localhost:8501`

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

---

## 📧 Contact

For questions about this dashboard:
- **Project:** Framework for Integrating ITS Concepts for Sri Lanka
- **Program:** Master's in Management Information Systems

---

**Version:** 1.0 | **Date:** January 2026
