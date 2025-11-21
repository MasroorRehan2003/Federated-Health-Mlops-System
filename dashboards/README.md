# Dashboards

This directory contains interactive dashboards for health risk monitoring.

## Dashboards Available

### 1. Health Authorities Dashboard
**File:** `health_authorities_dashboard.py`

**Purpose:** Public health monitoring dashboard for health authorities and officials.

**Features:**
- 📊 Regional comparison across all hospitals
- 🚨 Real-time alerts and notifications
- 📈 Risk trends over time
- 🌍 Multi-region visualization
- 🌡️ Environmental factor monitoring
- 💓 Health metrics tracking
- 📋 Detailed data tables

**Usage:**
```bash
streamlit run dashboards/health_authorities_dashboard.py
```

### 2. Citizens Dashboard
**File:** `citizens_dashboard.py`

**Purpose:** Personal health risk monitoring dashboard for individual citizens.

**Features:**
- 🎯 Current health status
- 🚨 Personal health alerts
- 📈 Individual health trends
- 🌡️ Environmental conditions
- 💓 Personal health metrics
- 💡 Health recommendations

**Usage:**
```bash
streamlit run dashboards/citizens_dashboard.py
```

## Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - streamlit>=1.28.0
   - plotly>=5.17.0
   - seaborn>=0.12.0
   - pandas
   - numpy

2. **Generate Data:**
   Before running the dashboards, ensure you have generated the processed data:
   ```bash
   python src/data_ingestion/main.py
   ```

   This will create the necessary files in `data/processed/`:
   - `hospital_A_merged_labeled.csv`
   - `hospital_B_merged_labeled.csv`
   - `hospital_C_merged_labeled.csv`
   - `all_nodes_merged_labeled.csv`

## Running the Dashboards

### Option 1: Run from Project Root
```bash
# Health Authorities Dashboard
streamlit run dashboards/health_authorities_dashboard.py

# Citizens Dashboard
streamlit run dashboards/citizens_dashboard.py
```

### Option 2: Run from Dashboards Directory
```bash
cd dashboards

# Health Authorities Dashboard
streamlit run health_authorities_dashboard.py

# Citizens Dashboard
streamlit run citizens_dashboard.py
```

## Dashboard Features

### Health Authorities Dashboard

**Key Metrics:**
- Total records analyzed
- High risk percentage
- Average AQI (Air Quality Index)
- Average PM2.5 levels

**Alerts:**
- High risk detection alerts
- Air quality alerts
- PM2.5 alerts
- Temperature alerts

**Visualizations:**
- Regional comparison charts
- Risk trends over time
- Environmental factor plots
- Health metrics charts
- Interactive data tables

**Filters:**
- Select specific regions
- Choose date range
- Real-time updates

### Citizens Dashboard

**Current Status:**
- Current risk level (High/Medium/Low)
- Air quality status
- Temperature
- PM2.5 levels

**Personal Alerts:**
- Region-specific health alerts
- Personalized recommendations
- Risk notifications

**Trends:**
- Personal health risk trend
- Air quality trend
- Temperature and humidity
- PM2.5 and AQI trends
- Health metrics (heart rate, SpO2)

**Recommendations:**
- Air quality recommendations
- Activity suggestions
- Health precautions

## Data Structure

The dashboards expect data files in the following format:

**Required Columns:**
- `timestamp`: DateTime column
- `location`: Region identifier (hospital_A, hospital_B, hospital_C)
- `health_risk`: Binary or continuous risk value (0-1)
- `aqi`: Air Quality Index
- `pm25`: PM2.5 level (μg/m³)
- `temperature`: Temperature (°C)
- `humidity`: Humidity (%)
- `heart_rate_mean`: Average heart rate (bpm)
- `spo2_mean`: Average SpO2 (%)

**Optional Columns:**
- `pm10`: PM10 level
- `co2`: CO2 level
- `wind_speed`: Wind speed
- `pressure`: Atmospheric pressure
- `precipitation`: Precipitation
- `steps_total`: Total steps
- `active_users_count`: Active user count

## Troubleshooting

### Issue: "No data found" Error
**Solution:** Run the data pipeline first:
```bash
python src/data_ingestion/main.py
```

### Issue: ModuleNotFoundError
**Solution:** Install all required dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Dashboard Shows Empty Charts
**Solution:** 
1. Check if data files exist in `data/processed/`
2. Verify data files contain required columns
3. Check date range filters in sidebar

### Issue: Streamlit Not Found
**Solution:** Install Streamlit:
```bash
pip install streamlit
```

## Customization

### Changing Color Schemes
Edit the CSS in the dashboard files to customize colors:
```python
st.markdown("""
    <style>
    .main-header {
        color: #your-color;
    }
    </style>
""", unsafe_allow_html=True)
```

### Adding New Metrics
1. Add metric to data pipeline output
2. Update `dashboards/utils.py` if needed
3. Add visualization in dashboard file

### Modifying Alerts
Edit the `get_alerts()` function in `dashboards/utils.py` to customize alert thresholds.

## Support

For issues or questions:
1. Check the main project README
2. Review data pipeline documentation
3. Check data file format requirements

## License

Same as the main project (MIT License)

