# Phase 1: Data Pipeline - Documentation

## Overview

Phase 1 implements a complete data ingestion and processing pipeline that generates, merges, and labels health-related data from multiple sources.

## Components

### 1. Data Generators

#### 1.1 Wearable Sensor Data Generator (`wearable_generator.py`)
- **Purpose**: Generates realistic wearable health device data
- **Features**:
  - Heart rate (varies by time of day)
  - Steps (accumulated throughout the day)
  - SpO2 (oxygen saturation)
  - Sleep status and quality
- **Output**: CSV files per node (e.g., `hospital_A.csv`)
- **Parameters**:
  - Number of users per node
  - Date range
  - Baseline health metrics per user

#### 1.2 Air Quality Sensor Data Generator (`air_quality_generator.py`)
- **Purpose**: Generates city-level environmental data
- **Features**:
  - PM2.5 (fine particulate matter)
  - PM10 (coarse particulate matter)
  - AQI (Air Quality Index)
  - CO2 levels
- **Output**: CSV files per node (e.g., `hospital_A_air_quality.csv`)
- **Characteristics**:
  - Varies by time of day (rush hours have higher pollution)
  - Varies by day of week (weekends have less traffic)
  - City-specific baselines (different pollution levels per node)

#### 1.3 Weather Data Generator (`weather_generator.py`)
- **Purpose**: Generates hourly weather data
- **Features**:
  - Temperature (with daily and seasonal cycles)
  - Humidity (inversely correlated with temperature)
  - Wind speed
  - Atmospheric pressure
  - Precipitation
- **Output**: CSV files per node (e.g., `hospital_A_weather.csv`)
- **Characteristics**:
  - Realistic daily temperature cycles
  - City-specific climate baselines

### 2. Data Merger (`data_merger.py`)

- **Purpose**: Combines all data streams into unified datasets
- **Process**:
  1. Aggregates wearable data to hourly level (mean, std, max, sum)
  2. Merges with air quality data (on timestamp)
  3. Merges with weather data (on timestamp)
  4. Adds derived features (time-based, categories)
- **Output**: 
  - Per-node merged files: `{node}_merged.csv`
  - Combined dataset: `all_nodes_merged.csv`

### 3. Label Generator (`label_generator.py`)

- **Purpose**: Creates health risk labels based on multiple factors
- **Risk Calculation**:
  - **Air Quality (25%)**: High AQI increases risk
  - **Heart Rate (15%)**: Elevated heart rate indicates stress
  - **SpO2 (20%)**: Low oxygen saturation is critical
  - **PM2.5 (15%)**: Fine particulate matter is harmful
  - **Weather (10%)**: Extreme temperatures and humidity
  - **Combined Effects (15%)**: Multiple factors together
- **Output Types**:
  - **Binary**: 0 (Low Risk) or 1 (High Risk)
  - **Continuous**: 0-1 risk score
- **Output**: `{node}_merged_labeled.csv`

## Data Flow

```
Raw Data Generation
├── Wearable Data (per user, hourly)
├── Air Quality Data (city-level, hourly)
└── Weather Data (city-level, hourly)
    ↓
Data Merging
├── Aggregate wearable data to hourly
├── Join with air quality (timestamp)
└── Join with weather (timestamp)
    ↓
Feature Engineering
├── Time-based features (hour, day_of_week, month)
├── Categorical features (AQI category, temp category)
└── Risk indicators (high_heart_rate, low_spo2, high_pollution)
    ↓
Label Generation
├── Calculate risk score (weighted combination)
└── Create binary/continuous labels
    ↓
Final Dataset
└── Labeled, merged data ready for modeling
```

## Usage

### Command Line

```bash
# Run complete pipeline
python src/data_ingestion/main.py

# Or import and use programmatically
from data_ingestion.main import run_data_pipeline

run_data_pipeline(
    nodes=['hospital_A', 'hospital_B', 'hospital_C'],
    num_users_per_node=100,
    days=30,
    generate_labels=True
)
```

### Individual Components

```python
# Generate wearable data only
from data_ingestion.wearable_generator import WearableDataGenerator
generator = WearableDataGenerator(seed=42)
generator.generate_node_data('hospital_A', num_users=100, days=30)

# Generate air quality data only
from data_ingestion.air_quality_generator import AirQualityGenerator
aq_gen = AirQualityGenerator(seed=42)
aq_gen.generate_node_data('hospital_A', days=30)

# Generate weather data only
from data_ingestion.weather_generator import WeatherDataGenerator
weather_gen = WeatherDataGenerator(seed=42)
weather_gen.generate_node_data('hospital_A', days=30)

# Merge data
from data_ingestion.data_merger import DataMerger
merger = DataMerger()
merger.merge_data_streams('hospital_A')

# Generate labels
from data_ingestion.label_generator import LabelGenerator
labeler = LabelGenerator()
labeler.add_labels_to_data('data/processed/hospital_A_merged.csv', method='binary')
```

## Output Files

### Raw Data (`data/raw/`)
- `{node}.csv` - Wearable sensor data
- `{node}_air_quality.csv` - Air quality sensor data
- `{node}_weather.csv` - Weather data

### Processed Data (`data/processed/`)
- `{node}_merged.csv` - Merged data without labels
- `{node}_merged_labeled.csv` - Merged data with binary labels
- `all_nodes_merged.csv` - Combined data from all nodes
- `all_nodes_merged_labeled.csv` - Combined labeled data

## Data Schema

### Merged Dataset Columns

**Wearable Features**:
- `heart_rate_mean`, `heart_rate_std`, `heart_rate_max`
- `steps_total`, `cumulative_steps_max`
- `spo2_mean`, `spo2_min`
- `sleeping_users_count`, `sleep_quality_mean`
- `active_users_count`

**Air Quality Features**:
- `pm25`, `pm10`, `aqi`, `co2`

**Weather Features**:
- `temperature`, `humidity`, `wind_speed`, `pressure`
- `precipitation`, `rainfall`

**Derived Features**:
- `hour`, `day_of_week`, `is_weekend`, `month`
- `aqi_category`, `temp_category`
- `high_heart_rate`, `low_spo2`, `high_pollution`

**Labels**:
- `health_risk` (binary: 0/1 or continuous: 0-1)
- `risk_category` (Low/Medium/High)

## Testing

Run the test script to verify the pipeline:

```bash
python test_data_pipeline.py
```

This will:
1. Generate data for all nodes (with smaller sample size)
2. Verify all files are created
3. Report success/failure

## Exploratory Data Analysis

Use the Jupyter notebook for detailed analysis:

```bash
jupyter notebook notebooks/01_data_pipeline_eda.ipynb
```

The notebook includes:
- Data generation
- Raw data exploration
- Merged data analysis
- Visualizations (time series, correlations)
- Label distribution analysis
- Feature engineering insights
- Cross-node comparisons

## Key Design Decisions

1. **Hourly Aggregation**: Wearable data is aggregated to hourly level to match air quality and weather data frequency
2. **Weighted Risk Calculation**: Multiple factors contribute to risk score with different weights based on medical significance
3. **City-Specific Baselines**: Different nodes have different pollution and climate baselines to simulate real-world diversity
4. **Reproducibility**: All generators use fixed random seeds for reproducible results
5. **Scalability**: Pipeline can handle multiple nodes and varying numbers of users

## Next Steps

After completing Phase 1, the labeled datasets are ready for:
- **Phase 2**: Federated Learning model training
- **Phase 3**: Model evaluation and comparison
- **Phase 4**: MLOps pipeline integration
- **Phase 5**: Deployment and monitoring

