# End-to-End MLOps System for Health Risk Prediction

A comprehensive MLOps system that collects data from multiple sources (wearable health devices, air-quality sensors, weather data) and uses AI models with Federated Learning to predict health risks in real-time.

## Project Structure

```
.
├── data/                      # Data storage
│   ├── raw/                   # Raw data from sources
│   ├── processed/             # Processed data
│   └── models/                # Trained models
├── src/
│   ├── data_ingestion/        # Data collection from multiple sources
│   ├── federated_learning/    # Federated learning implementation
│   ├── models/                # AI model definitions
│   ├── mlops/                 # MLOps pipeline components
│   ├── monitoring/            # Model monitoring and drift detection
│   └── api/                   # API endpoints
├── notebooks/                 # Jupyter notebooks for EDA and experiments
├── dashboards/                # Dashboard applications
├── docker/                    # Docker configurations
├── k8s/                       # Kubernetes manifests
├── ci_cd/                     # CI/CD pipeline configurations
└── docs/                      # Documentation and project paper

```

## Features

- **Multi-Source Data Ingestion**: Collects data from wearables, IoT sensors, and weather APIs
- **Federated Learning**: Train models across distributed nodes without centralizing data
- **Data Drift Detection**: Monitor and detect changes in data distribution
- **MLOps Pipeline**: Automated CI/CD for ML models
- **Real-time Monitoring**: Track model performance and system health
- **Interactive Dashboards**: Visualize insights for health authorities and citizens

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Kubernetes (optional, for production)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "MLOPS Project"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Start all services with Docker Compose
docker-compose up -d

# Or run individual components

# 1. Generate data
python src/data_ingestion/main.py

# 2. Run federated learning (requires multiple terminals or Docker)
python src/federated_learning/fl_server.py  # Terminal 1
python src/federated_learning/fl_client.py --node-name hospital_A  # Terminal 2
python src/federated_learning/fl_client.py --node-name hospital_B  # Terminal 3
python src/federated_learning/fl_client.py --node-name hospital_C  # Terminal 4

# 3. Run dashboards
streamlit run dashboards/health_authorities_dashboard.py  # Health authorities dashboard
streamlit run dashboards/citizens_dashboard.py  # Citizens dashboard
```

## Components

### 1. Data Ingestion System
Simulates data collection from:
- Wearable health devices (heart rate, steps, sleep)
- Air quality sensors (PM2.5, PM10, CO2)
- Weather data (temperature, humidity, pressure)

### 2. Federated Learning
Implements federated learning using Flower framework:
- Central server for aggregation
- Multiple client nodes (hospitals, cities)
- Privacy-preserving model training

### 3. MLOps Pipeline
- Experiment tracking with MLflow
- Model versioning and registry
- Automated retraining pipelines
- CI/CD integration

### 4. Dashboards
- **Health Authorities Dashboard** (`dashboards/health_authorities_dashboard.py`)
  - Regional risk comparison
  - Real-time alerts and notifications
  - Risk trends over time
  - Environmental factor monitoring
  - Multi-region visualization
  
- **Citizens Dashboard** (`dashboards/citizens_dashboard.py`)
  - Personal health status
  - Individual health alerts
  - Personal health trends
  - Health recommendations
  - Environmental conditions

## Documentation

See `docs/` directory for:
- Project paper
- API documentation
- Deployment guides
- Evaluation reports

## License

MIT License

