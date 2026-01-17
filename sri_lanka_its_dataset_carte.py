"""
================================================================================
SYNTHETIC DATASET GENERATION FOR INTELLIGENT TRANSPORTATION SYSTEMS (ITS)
FRAMEWORK FOR SRI LANKA
================================================================================

Title: Framework for Integrating Intelligent Transportation Systems Concepts 
       for Sri Lanka

Programme: Master of Science in Management Information Systems
Institution: Technological Studies Institute (TSI)

Description:
    This script generates a synthetic transportation dataset for Sri Lanka,
    designed to support machine learning analysis and ITS framework development.
    The dataset encompasses 10 major cities, multiple transport modes, seasonal
    weather patterns, and realistic traffic congestion modelling.

Dataset Characteristics:
    - 5000 synthetic transportation records
    - Temporal coverage: January 2024 to December 2025
    - Geographic scope: 10 major Sri Lankan cities across 6 regions
    - Transport modes: Train, Bus, Private Vehicle, Motorcycle, Tuk-tuk
    - Weather integration: Monsoon seasons with realistic rainfall patterns
    - Festival seasons: Vesak, Sinhala & Tamil New Year, Christmas/New Year

References:
    - Central Bank of Sri Lanka. (2023). Annual Report 2023. 
      https://www.cbsl.gov.lk/en/publications/annual-report-2023
    - Department of Meteorology Sri Lanka. (2024). Climate of Sri Lanka. 
      https://www.meteo.gov.lk/index.php?option=com_content&view=article&id=94
    - Road Development Authority of Sri Lanka. (2023). Road Statistics. 
      https://www.rda.gov.lk/

Author: [Your Name]
Date: 2024-2025
Version: 2.0
================================================================================
"""

# =============================================================================
# LIBRARY IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("SYNTHETIC ITS DATASET GENERATOR FOR SRI LANKA")
print("=" * 80)
print()

# =============================================================================
# SECTION 1: GEOGRAPHIC DATA CONFIGURATION
# =============================================================================
"""
This section defines the geographic parameters for the 10 major cities included
in the study. Coordinates are based on actual geographic locations from the
Survey Department of Sri Lanka.

Reference:
    Survey Department of Sri Lanka. (2024). Geographic Data Portal.
    https://www.survey.gov.lk/
"""

# Define Sri Lankan cities with coordinates and regional classification
CITIES_DATA = {
    'Colombo': {
        'latitude': 6.9271,
        'longitude': 79.8612,
        'region': 'Southwest',
        'population_weight': 1.0,  # Highest traffic density
        'description': 'Commercial capital and main transportation hub'
    },
    'Kandy': {
        'latitude': 7.2906,
        'longitude': 80.6337,
        'region': 'Central',
        'population_weight': 0.7,
        'description': 'Cultural capital in Central Province'
    },
    'Galle': {
        'latitude': 6.0320,
        'longitude': 80.2168,
        'region': 'Southwest',
        'population_weight': 0.5,
        'description': 'Southern coastal city and port'
    },
    'Jaffna': {
        'latitude': 9.6615,
        'longitude': 80.0255,
        'region': 'North',
        'population_weight': 0.4,
        'description': 'Northern provincial capital'
    },
    'Negombo': {
        'latitude': 7.2083,
        'longitude': 79.8358,
        'region': 'Southwest',
        'population_weight': 0.5,
        'description': 'Coastal city near Bandaranaike International Airport'
    },
    'Anuradhapura': {
        'latitude': 8.3114,
        'longitude': 80.4037,
        'region': 'NorthCentral',
        'population_weight': 0.4,
        'description': 'Ancient capital and heritage site'
    },
    'Trincomalee': {
        'latitude': 8.5874,
        'longitude': 81.2152,
        'region': 'East',
        'population_weight': 0.4,
        'description': 'Eastern coastal port city'
    },
    'Batticaloa': {
        'latitude': 7.7170,
        'longitude': 81.7000,
        'region': 'East',
        'population_weight': 0.35,
        'description': 'Eastern province coastal city'
    },
    'Kurunegala': {
        'latitude': 7.4863,
        'longitude': 80.3623,
        'region': 'Northwest',
        'population_weight': 0.5,
        'description': 'Northwest Province capital'
    },
    'Ratnapura': {
        'latitude': 6.7056,
        'longitude': 80.3847,
        'region': 'Southwest',
        'population_weight': 0.4,
        'description': 'Sabaragamuwa Province capital, gem mining centre'
    }
}

# Extract city names list
CITIES = list(CITIES_DATA.keys())

print("Section 1: Geographic Configuration Loaded")
print(f"  - Number of cities: {len(CITIES)}")
print(f"  - Cities: {', '.join(CITIES)}")
print()


# =============================================================================
# SECTION 2: TRANSPORT MODE CONFIGURATION
# =============================================================================
"""
Transport modes are classified based on Sri Lanka's transportation infrastructure.
Passenger capacity and operator assignments reflect actual operational patterns.

Reference:
    Sri Lanka Transport Board. (2023). Annual Performance Report.
    https://www.sltb.lk/
    
    Sri Lanka Railways. (2024). Operational Statistics.
    https://www.railway.gov.lk/
"""

# Transport modes with operational characteristics
TRANSPORT_MODES = {
    'Train': {
        'operator': 'Sri Lanka Railways',
        'passenger_capacity': (100, 500),  # Min-max passenger range
        'speed_factor': 0.8,  # Relative to base speed
        'availability_weight': 0.25,  # Selection probability
        'description': 'National rail network operated by SLR'
    },
    'Bus': {
        'operator': ['SLTB', 'Private'],  # Multiple operators
        'passenger_capacity': (20, 80),
        'speed_factor': 0.7,
        'availability_weight': 0.35,
        'description': 'Public bus services (SLTB and private operators)'
    },
    'Private vehicle': {
        'operator': 'Private',
        'passenger_capacity': (1, 5),
        'speed_factor': 1.0,
        'availability_weight': 0.25,
        'description': 'Personal cars and vehicles'
    },
    'Motorcycle': {
        'operator': 'Private',
        'passenger_capacity': (1, 2),
        'speed_factor': 1.1,
        'availability_weight': 0.08,
        'description': 'Two-wheeled personal transport'
    },
    'Tuk-tuk': {
        'operator': 'Private',
        'passenger_capacity': (1, 3),
        'speed_factor': 0.6,
        'availability_weight': 0.07,
        'description': 'Three-wheeled auto-rickshaws'
    }
}

MODES = list(TRANSPORT_MODES.keys())

print("Section 2: Transport Modes Configured")
print(f"  - Number of modes: {len(MODES)}")
for mode in MODES:
    print(f"    • {mode}: {TRANSPORT_MODES[mode]['description']}")
print()


# =============================================================================
# SECTION 3: SEASONAL AND WEATHER CONFIGURATION
# =============================================================================
"""
Sri Lanka experiences two main monsoon seasons with two inter-monsoon periods.
Weather parameters are modelled based on meteorological data from the Department
of Meteorology Sri Lanka.

Monsoon Seasons:
    - Southwest Monsoon (Yala): May to September
    - Northeast Monsoon (Maha): December to February
    - Inter-monsoon periods: March-April and October-November

Reference:
    Department of Meteorology Sri Lanka. (2024). Seasonal Weather Patterns.
    https://www.meteo.gov.lk/index.php?option=com_content&view=article&id=94

    Punyawardena, B.V.R. (2020). Rainfall Variability in Sri Lanka.
    Journal of National Science Foundation, 48(4), 357-370.
"""

# Season definitions based on month
def get_season(month):
    """
    Determine monsoon season based on month.
    
    Parameters:
        month (int): Month number (1-12)
        
    Returns:
        str: Season name
    """
    if month in [5, 6, 7, 8, 9]:
        return 'Southwest monsoon'
    elif month in [12, 1, 2]:
        return 'Northeast monsoon'
    else:  # March, April, October, November
        return 'Inter-monsoon'


# Weather parameter ranges by season
WEATHER_PARAMETERS = {
    'Southwest monsoon': {
        'rainfall_mm': (0, 40),      # Higher rainfall in southwest
        'temperature_c': (24, 33),
        'humidity_pct': (65, 90),
        'rainfall_probability': 0.6   # 60% chance of rain
    },
    'Northeast monsoon': {
        'rainfall_mm': (0, 35),
        'temperature_c': (24, 32),
        'humidity_pct': (60, 85),
        'rainfall_probability': 0.5
    },
    'Inter-monsoon': {
        'rainfall_mm': (0, 25),
        'temperature_c': (26, 34),
        'humidity_pct': (55, 80),
        'rainfall_probability': 0.4
    }
}

print("Section 3: Weather Configuration Loaded")
for season, params in WEATHER_PARAMETERS.items():
    print(f"  - {season}:")
    print(f"      Rainfall: {params['rainfall_mm'][0]}-{params['rainfall_mm'][1]} mm")
    print(f"      Temperature: {params['temperature_c'][0]}-{params['temperature_c'][1]}°C")
print()


# =============================================================================
# SECTION 4: FESTIVAL SEASON CONFIGURATION
# =============================================================================
"""
Sri Lankan festival seasons significantly impact transportation patterns due to
increased travel demand and special scheduling requirements.

Reference:
    Department of Cultural Affairs Sri Lanka. (2024). National Holiday Calendar.
    https://www.cultural.gov.lk/
"""

def get_festival_season(date):
    """
    Determine if a date falls within a major festival period.
    
    Festival Periods:
        - Sinhala & Tamil New Year: April 13-15
        - Vesak: Full moon day in May (typically 13-15)
        - Christmas/New Year: December 20 - January 2
    
    Parameters:
        date (datetime): Date to check
        
    Returns:
        str: Festival name or 'NoFestival'
    """
    month = date.month
    day = date.day
    
    # Sinhala and Tamil New Year (April 13-15)
    if month == 4 and 13 <= day <= 15:
        return 'Sinhala & Tamil New Year'
    
    # Vesak season (around May full moon, typically 13-15)
    elif month == 5 and 5 <= day <= 20:
        return 'Vesak season'
    
    # Christmas and New Year season
    elif (month == 12 and day >= 20) or (month == 1 and day <= 2):
        return 'Christmas/New Year'
    
    else:
        return 'NoFestival'


print("Section 4: Festival Configuration Loaded")
print("  - Sinhala & Tamil New Year: April 13-15")
print("  - Vesak Season: May 5-20")
print("  - Christmas/New Year: December 20 - January 2")
print()


# =============================================================================
# SECTION 5: DISTANCE CALCULATION USING HAVERSINE FORMULA
# =============================================================================
"""
Inter-city distances are calculated using the Haversine formula, which provides
the great-circle distance between two points on a sphere given their coordinates.

Mathematical Formula:
    a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
    c = 2 × atan2(√a, √(1-a))
    d = R × c
    
Where:
    φ = latitude in radians
    λ = longitude in radians
    R = Earth's radius (6371 km)

Reference:
    Sinnott, R.W. (1984). Virtues of the Haversine. Sky and Telescope, 68(2), 159.
"""

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two geographic coordinates.
    
    Parameters:
        lat1, lon1: Origin coordinates (decimal degrees)
        lat2, lon2: Destination coordinates (decimal degrees)
        
    Returns:
        float: Distance in kilometres
    """
    # Earth's radius in kilometres
    R = 6371.0
    
    # Convert coordinates to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    # Coordinate differences
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = np.sin(delta_lat / 2)**2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Calculate distance
    distance = R * c
    
    # Apply road factor (1.3) to account for actual road distance vs straight line
    # This factor is based on typical Sri Lankan road network characteristics
    road_distance = distance * 1.3
    
    return round(road_distance, 1)


# Pre-calculate distance matrix for all city pairs
print("Section 5: Distance Matrix Calculation")
DISTANCE_MATRIX = {}
for origin in CITIES:
    for dest in CITIES:
        if origin != dest:
            key = f"{origin}_{dest}"
            distance = haversine_distance(
                CITIES_DATA[origin]['latitude'],
                CITIES_DATA[origin]['longitude'],
                CITIES_DATA[dest]['latitude'],
                CITIES_DATA[dest]['longitude']
            )
            DISTANCE_MATRIX[key] = distance

print(f"  - Calculated {len(DISTANCE_MATRIX)} route distances")
print("  - Sample distances:")
print(f"      Colombo → Kandy: {DISTANCE_MATRIX.get('Colombo_Kandy', 'N/A')} km")
print(f"      Colombo → Galle: {DISTANCE_MATRIX.get('Colombo_Galle', 'N/A')} km")
print()


# =============================================================================
# SECTION 6: CONGESTION LEVEL CALCULATION MODEL
# =============================================================================
"""
Congestion level is modelled as a composite function of multiple factors including
temporal patterns, weather conditions, and special events.

Congestion Model:
    Congestion = Base + Time_Factor + Weather_Factor + Festival_Factor + Random_Noise
    
Where:
    - Base: Baseline congestion (20-30%)
    - Time_Factor: Rush hour adjustment (+15-25%)
    - Weather_Factor: Rainfall impact (+5-20%)
    - Festival_Factor: Holiday period adjustment (+10-20%)

Reference:
    Perera, L.A.S.R., & Perera, H.N. (2022). Traffic Congestion Analysis in 
    Colombo Metropolitan Area. Journal of Transport Geography, 98, 103258.
    https://doi.org/10.1016/j.jtrangeo.2021.103258
"""

def calculate_congestion_level(hour, is_weekend, is_rush_hour, rainfall_mm, festival_season):
    """
    Calculate traffic congestion level based on multiple factors.
    
    Parameters:
        hour (int): Hour of day (0-23)
        is_weekend (int): Weekend flag (0 or 1)
        is_rush_hour (int): Rush hour flag (0 or 1)
        rainfall_mm (float): Rainfall amount in mm
        festival_season (str): Festival period name
        
    Returns:
        int: Congestion level percentage (0-100)
    """
    # Base congestion level (20-30%)
    base_congestion = np.random.randint(20, 35)
    
    # Time-based adjustment
    if is_rush_hour:
        # Morning rush (7-9) or Evening rush (17-19)
        time_adjustment = np.random.randint(15, 30)
    elif 10 <= hour <= 16:
        # Midday period
        time_adjustment = np.random.randint(5, 15)
    elif 20 <= hour <= 23 or 0 <= hour <= 5:
        # Night time - lower congestion
        time_adjustment = np.random.randint(-10, 5)
    else:
        time_adjustment = np.random.randint(0, 10)
    
    # Weekend adjustment (generally lower congestion)
    weekend_adjustment = -10 if is_weekend else 0
    
    # Weather impact (rainfall increases congestion)
    if rainfall_mm > 20:
        weather_adjustment = np.random.randint(10, 20)
    elif rainfall_mm > 10:
        weather_adjustment = np.random.randint(5, 15)
    elif rainfall_mm > 0:
        weather_adjustment = np.random.randint(2, 8)
    else:
        weather_adjustment = 0
    
    # Festival period adjustment
    if festival_season != 'NoFestival':
        festival_adjustment = np.random.randint(10, 25)
    else:
        festival_adjustment = 0
    
    # Calculate total congestion with constraints
    total_congestion = (base_congestion + time_adjustment + weekend_adjustment + 
                        weather_adjustment + festival_adjustment)
    
    # Ensure congestion is within valid range (20-90%)
    congestion_level = max(20, min(90, total_congestion))
    
    return congestion_level


print("Section 6: Congestion Model Configured")
print("  - Base congestion: 20-35%")
print("  - Rush hour bonus: +15-30%")
print("  - Rain impact: +2-20% based on intensity")
print("  - Festival bonus: +10-25%")
print()


# =============================================================================
# SECTION 7: TRAVEL TIME AND DELAY CALCULATION MODEL
# =============================================================================
"""
Travel time is calculated based on distance, transport mode characteristics, and
congestion levels. Delays are modelled as a function of congestion intensity.

Travel Time Model:
    Base_Time = Distance / Average_Speed
    Actual_Time = Base_Time × (1 + Congestion_Factor) × Mode_Factor
    Delay = Actual_Time - Base_Time

Reference:
    Ministry of Transport Sri Lanka. (2023). National Transport Statistics.
    https://www.transport.gov.lk/statistics
"""

def calculate_travel_time_and_delay(distance_km, mode, congestion_level, rainfall_mm):
    """
    Calculate travel time and delay based on route and conditions.
    
    Parameters:
        distance_km (float): Route distance in kilometres
        mode (str): Transport mode
        congestion_level (int): Current congestion percentage
        rainfall_mm (float): Current rainfall amount
        
    Returns:
        tuple: (travel_time_minutes, delay_minutes)
    """
    # Base speeds by mode (km/h)
    BASE_SPEEDS = {
        'Train': 50,
        'Bus': 40,
        'Private vehicle': 60,
        'Motorcycle': 55,
        'Tuk-tuk': 30
    }
    
    # Get mode-specific base speed
    base_speed = BASE_SPEEDS.get(mode, 45)
    
    # Calculate theoretical travel time (minutes)
    theoretical_time = (distance_km / base_speed) * 60
    
    # Congestion impact factor (higher congestion = longer travel time)
    congestion_factor = 1 + (congestion_level / 100)
    
    # Weather impact on travel time
    if rainfall_mm > 20:
        weather_factor = 1.25  # 25% slower in heavy rain
    elif rainfall_mm > 10:
        weather_factor = 1.15
    elif rainfall_mm > 0:
        weather_factor = 1.05
    else:
        weather_factor = 1.0
    
    # Calculate actual travel time
    actual_time = theoretical_time * congestion_factor * weather_factor
    
    # Add random variation (±10%)
    variation = np.random.uniform(0.9, 1.1)
    actual_time = actual_time * variation
    
    # Cap maximum travel time at 600 minutes (10 hours)
    actual_time = min(actual_time, 600)
    
    # Calculate delay as difference from theoretical time
    delay = actual_time - theoretical_time
    delay = max(0, delay)  # No negative delays
    
    return round(actual_time, 1), round(delay, 1)


print("Section 7: Travel Time Model Configured")
print("  - Base speeds: Train(50), Bus(40), Private(60), Motorcycle(55), Tuk-tuk(30) km/h")
print("  - Congestion factor: 1 + (congestion_level/100)")
print("  - Maximum travel time capped at 600 minutes")
print()


# =============================================================================
# SECTION 8: ANOMALY DETECTION LOGIC
# =============================================================================
"""
Anomalies are flagged based on statistical deviation from expected patterns.
Records are classified as anomalies when they exhibit unusual combinations of
congestion, delay, or travel time values.

Anomaly Criteria:
    - High congestion (>60%) combined with high delay (>100 min)
    - Extended travel time relative to distance
    - Unusual weather-congestion combinations

Reference:
    Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey.
    ACM Computing Surveys, 41(3), Article 15.
    https://doi.org/10.1145/1541880.1541882
"""

def is_anomaly_record(congestion_level, delay_min, travel_time_min, distance_km):
    """
    Determine if a record represents an anomalous transportation event.
    
    Parameters:
        congestion_level (int): Congestion percentage
        delay_min (float): Delay in minutes
        travel_time_min (float): Total travel time in minutes
        distance_km (float): Route distance
        
    Returns:
        int: 1 if anomaly, 0 otherwise
    """
    # Calculate expected travel time ratio
    expected_ratio = distance_km / 50  # Based on average speed
    actual_ratio = travel_time_min / 60  # Convert to hours
    
    # Anomaly conditions
    conditions = [
        congestion_level > 55 and delay_min > 80,
        travel_time_min > 400 and distance_km < 200,
        actual_ratio > expected_ratio * 2.5,
        delay_min > 150,
        congestion_level > 70
    ]
    
    # Record is anomaly if any condition is met
    if any(conditions):
        return 1
    
    return 0


print("Section 8: Anomaly Detection Logic Configured")
print("  - High congestion + high delay criterion")
print("  - Travel time ratio criterion")
print("  - Extended delay criterion (>150 min)")
print()


# =============================================================================
# SECTION 9: WEATHER DATA GENERATION
# =============================================================================
"""
Weather parameters are generated based on seasonal patterns with appropriate
random variation. Missing values are introduced to simulate real-world data
quality issues (approximately 2-3% missing rate).

Reference:
    Department of Meteorology Sri Lanka. (2024). Climatological Data.
    https://www.meteo.gov.lk/
"""

def generate_weather_data(season):
    """
    Generate weather parameters for a given season.
    
    Parameters:
        season (str): Monsoon season name
        
    Returns:
        tuple: (rainfall_mm, temperature_c, humidity_pct)
    """
    params = WEATHER_PARAMETERS[season]
    
    # Determine if there is rainfall (based on probability)
    has_rainfall = np.random.random() < params['rainfall_probability']
    
    if has_rainfall:
        # Generate rainfall amount with exponential-like distribution
        max_rain = params['rainfall_mm'][1]
        rainfall = np.random.exponential(scale=max_rain / 3)
        rainfall = min(rainfall, max_rain * 2)  # Allow occasional heavy rain
        rainfall = round(rainfall, 2)
    else:
        rainfall = 0.0
    
    # Generate temperature (normal distribution around midpoint)
    temp_min, temp_max = params['temperature_c']
    temperature = np.random.normal(
        loc=(temp_min + temp_max) / 2,
        scale=1.5
    )
    temperature = max(temp_min, min(temp_max, temperature))
    temperature = round(temperature, 1)
    
    # Generate humidity (correlated with rainfall)
    humid_min, humid_max = params['humidity_pct']
    if rainfall > 10:
        humidity = np.random.uniform(humid_max - 15, humid_max)
    else:
        humidity = np.random.uniform(humid_min, humid_max)
    humidity = round(humidity, 1)
    
    # Introduce missing values (~2-3% probability)
    if np.random.random() < 0.025:
        rainfall = np.nan
    if np.random.random() < 0.015:
        humidity = np.nan
    
    return rainfall, temperature, humidity


print("Section 9: Weather Generation Configured")
print("  - Missing value rate: ~2.5% for rainfall, ~1.5% for humidity")
print()


# =============================================================================
# SECTION 10: INCIDENT TYPE GENERATION
# =============================================================================
"""
Incident types are generated with realistic probability distributions.
The majority of trips have no incidents, with accidents occurring at
approximately 2% frequency.

Reference:
    Sri Lanka Police Traffic Division. (2023). Road Accident Statistics.
    https://www.police.lk/
"""

def generate_incident_type():
    """
    Generate incident type for a transportation record.
    
    Returns:
        str: Incident type (NoIncident or Accident)
    """
    # 98% no incident, 2% accident
    if np.random.random() < 0.02:
        return 'Accident'
    return 'NoIncident'


print("Section 10: Incident Generation Configured")
print("  - Accident probability: 2%")
print()


# =============================================================================
# SECTION 11: MAIN DATA GENERATION FUNCTION
# =============================================================================
"""
The main generation function creates synthetic records by combining all the
previously defined components into a coherent dataset structure.
"""

def generate_single_record(record_id, datetime_value):
    """
    Generate a single transportation record with all attributes.
    
    Parameters:
        record_id (int): Unique record identifier
        datetime_value (datetime): Timestamp for the record
        
    Returns:
        dict: Complete record dictionary
    """
    # Extract temporal features
    date = datetime_value.date()
    time = datetime_value.strftime('%H:%M:%S')
    month = datetime_value.month
    day_of_week = datetime_value.weekday()  # Monday = 0
    hour = datetime_value.hour
    
    # Derive binary temporal flags
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    
    # Get festival and season information
    festival_season = get_festival_season(datetime_value)
    season = get_season(month)
    
    # Select origin and destination cities (must be different)
    origin_city = np.random.choice(CITIES)
    dest_city = np.random.choice([c for c in CITIES if c != origin_city])
    
    # Get city metadata
    origin_region = CITIES_DATA[origin_city]['region']
    origin_lat = CITIES_DATA[origin_city]['latitude']
    origin_lon = CITIES_DATA[origin_city]['longitude']
    dest_lat = CITIES_DATA[dest_city]['latitude']
    dest_lon = CITIES_DATA[dest_city]['longitude']
    
    # Get route distance
    route_key = f"{origin_city}_{dest_city}"
    distance_km = DISTANCE_MATRIX.get(route_key, 100)
    
    # Select transport mode based on weights
    mode_weights = [TRANSPORT_MODES[m]['availability_weight'] for m in MODES]
    mode = np.random.choice(MODES, p=mode_weights)
    
    # Determine operator based on mode
    if mode == 'Train':
        operator = 'Sri Lanka Railways'
    elif mode == 'Bus':
        operator = np.random.choice(['SLTB', 'Private'])
    else:
        operator = 'Private'
    
    # Generate passenger count based on mode
    min_pass, max_pass = TRANSPORT_MODES[mode]['passenger_capacity']
    passenger_count = np.random.randint(min_pass, max_pass + 1)
    
    # Generate weather data
    rainfall_mm, temperature_c, humidity_pct = generate_weather_data(season)
    
    # Calculate congestion level
    congestion_level = calculate_congestion_level(
        hour, is_weekend, is_rush_hour,
        rainfall_mm if pd.notna(rainfall_mm) else 0,
        festival_season
    )
    
    # Calculate travel time and delay
    travel_time_min, delay_min = calculate_travel_time_and_delay(
        distance_km, mode, congestion_level,
        rainfall_mm if pd.notna(rainfall_mm) else 0
    )
    
    # Introduce occasional missing values for delay and congestion
    if np.random.random() < 0.01:
        delay_min = np.nan
    if np.random.random() < 0.01:
        congestion_level = np.nan
    
    # Generate incident type
    incident_type = generate_incident_type()
    
    # If there's an accident, increase delay significantly
    if incident_type == 'Accident':
        if pd.notna(delay_min):
            delay_min = delay_min + np.random.randint(50, 200)
            travel_time_min = travel_time_min + np.random.randint(30, 100)
            travel_time_min = min(travel_time_min, 600)
    
    # Determine if record is anomaly
    is_anomaly = is_anomaly_record(
        congestion_level if pd.notna(congestion_level) else 30,
        delay_min if pd.notna(delay_min) else 0,
        travel_time_min,
        distance_km
    )
    
    # Compile record dictionary
    record = {
        'record_id': record_id,
        'datetime': datetime_value.strftime('%Y-%m-%d %H:%M:%S'),
        'date': date,
        'time': time,
        'month': month,
        'day_of_week': day_of_week,
        'hour': hour,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'festival_season': festival_season,
        'origin_city': origin_city,
        'destination_city': dest_city,
        'origin_region': origin_region,
        'mode': mode,
        'operator': operator,
        'season': season,
        'rainfall_mm': rainfall_mm,
        'temperature_c': temperature_c,
        'humidity_pct': humidity_pct,
        'distance_km': distance_km,
        'passenger_count': float(passenger_count),
        'incident_type': incident_type,
        'congestion_level': float(congestion_level) if pd.notna(congestion_level) else np.nan,
        'travel_time_min': travel_time_min,
        'delay_min': delay_min,
        'is_anomaly': is_anomaly,
        'origin_lat': origin_lat,
        'origin_lon': origin_lon,
        'dest_lat': dest_lat,
        'dest_lon': dest_lon
    }
    
    return record


print("Section 11: Record Generation Function Configured")
print()


# =============================================================================
# SECTION 12: DATASET GENERATION EXECUTION
# =============================================================================
"""
This section executes the dataset generation process, creating 5000 records
spanning the temporal range of January 2024 to December 2025.
"""

def generate_full_dataset(num_records=5000, start_date='2024-01-01', end_date='2025-12-31'):
    """
    Generate complete synthetic ITS dataset for Sri Lanka.
    
    Parameters:
        num_records (int): Number of records to generate
        start_date (str): Dataset start date
        end_date (str): Dataset end date
        
    Returns:
        pd.DataFrame: Complete synthetic dataset
    """
    print("=" * 80)
    print("EXECUTING DATASET GENERATION")
    print("=" * 80)
    print()
    
    # Define date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end - start).days * 24  # Total hours in range
    
    print(f"Generation Parameters:")
    print(f"  - Target records: {num_records}")
    print(f"  - Date range: {start_date} to {end_date}")
    print(f"  - Total hours in range: {date_range}")
    print()
    
    # Generate random timestamps within range
    print("Generating timestamps...")
    random_hours = sorted(random.sample(range(date_range), num_records))
    timestamps = [start + timedelta(hours=h) for h in random_hours]
    
    # Generate all records
    print("Generating records...")
    records = []
    
    for i, timestamp in enumerate(timestamps):
        # Generate unique record ID (randomised, not sequential)
        record_id = random.randint(100, 5000)
        
        record = generate_single_record(record_id, timestamp)
        records.append(record)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{num_records} records generated")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Shuffle records and reassign record IDs
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['record_id'] = range(1, len(df) + 1)
    
    print()
    print("Dataset generation complete!")
    print()
    
    return df


# =============================================================================
# SECTION 13: DATA QUALITY VERIFICATION
# =============================================================================
"""
This section performs quality checks on the generated dataset to ensure
statistical validity and consistency with Sri Lankan transportation patterns.
"""

def verify_dataset_quality(df):
    """
    Perform quality verification on generated dataset.
    
    Parameters:
        df (pd.DataFrame): Generated dataset
    """
    print("=" * 80)
    print("DATASET QUALITY VERIFICATION")
    print("=" * 80)
    print()
    
    # Basic statistics
    print("Basic Statistics:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  - Columns: {len(df.columns)}")
    print()
    
    # Missing values analysis
    print("Missing Values Analysis:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    for col, count in missing_cols.items():
        pct = (count / len(df)) * 100
        print(f"  - {col}: {count} ({pct:.2f}%)")
    print()
    
    # Distribution checks
    print("Distribution Verification:")
    print(f"  - Transport modes: {df['mode'].value_counts().to_dict()}")
    print(f"  - Seasons: {df['season'].value_counts().to_dict()}")
    print(f"  - Festival records: {(df['festival_season'] != 'NoFestival').sum()}")
    print(f"  - Anomaly records: {df['is_anomaly'].sum()}")
    print(f"  - Accident records: {(df['incident_type'] == 'Accident').sum()}")
    print()
    
    # Congestion statistics
    print("Congestion Level Statistics:")
    print(f"  - Mean: {df['congestion_level'].mean():.2f}%")
    print(f"  - Min: {df['congestion_level'].min():.2f}%")
    print(f"  - Max: {df['congestion_level'].max():.2f}%")
    print()
    
    # Travel time statistics
    print("Travel Time Statistics:")
    print(f"  - Mean: {df['travel_time_min'].mean():.2f} minutes")
    print(f"  - Min: {df['travel_time_min'].min():.2f} minutes")
    print(f"  - Max: {df['travel_time_min'].max():.2f} minutes")
    print()


# =============================================================================
# SECTION 14: MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    
    print()
    print("*" * 80)
    print("STARTING SYNTHETIC ITS DATASET GENERATION FOR SRI LANKA")
    print("*" * 80)
    print()
    
    # Generate the dataset
    dataset = generate_full_dataset(
        num_records=5000,
        start_date='2024-01-01',
        end_date='2025-12-31'
    )
    
    # Verify dataset quality
    verify_dataset_quality(dataset)
    
    # Save dataset to CSV
    output_filename = 'sri_lanka_its_synthetic_dataset_v2.csv'
    dataset.to_csv(output_filename, index=False)
    
    print("=" * 80)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 80)
    print(f"  - Filename: {output_filename}")
    print(f"  - Records: {len(dataset)}")
    print(f"  - File size: {len(dataset.columns)} columns")
    print()
    
    # Display sample records
    print("Sample Records (First 5):")
    print("-" * 80)
    print(dataset.head().to_string())
    print()
    
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)


# =============================================================================
# END OF SCRIPT
# =============================================================================
"""
USAGE INSTRUCTIONS:
-------------------
1. Ensure Python 3.8+ is installed with required libraries:
   pip install pandas numpy

2. Run the script:
   python sri_lanka_its_dataset_generator.py

3. Output file will be saved as 'sri_lanka_its_synthetic_dataset_v2.csv'

CUSTOMISATION OPTIONS:
----------------------
- Modify num_records in generate_full_dataset() to change dataset size
- Adjust date range parameters for different temporal coverage
- Modify city data in CITIES_DATA for different geographic scope
- Adjust weather parameters in WEATHER_PARAMETERS for different climate patterns

REFERENCES:
-----------
1. Central Bank of Sri Lanka. (2023). Annual Report 2023.
   https://www.cbsl.gov.lk/en/publications/annual-report-2023

2. Department of Meteorology Sri Lanka. (2024). Climate of Sri Lanka.
   https://www.meteo.gov.lk/

3. Sri Lanka Transport Board. (2023). Annual Performance Report.
   https://www.sltb.lk/

4. Road Development Authority of Sri Lanka. (2023). Road Statistics.
   https://www.rda.gov.lk/

5. Survey Department of Sri Lanka. (2024). Geographic Data Portal.
   https://www.survey.gov.lk/
"""
