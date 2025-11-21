# Phase 2: Federated Learning - Documentation

## Overview

Phase 2 implements a federated learning system that trains health risk prediction models across multiple distributed nodes (hospitals/cities) without centralizing sensitive data.

## Architecture

### Components

1. **Federated Learning Server** (`fl_server.py`)
   - Coordinates training across multiple nodes
   - Aggregates model parameters using FedAvg (Federated Averaging)
   - Tracks experiments with MLflow
   - Manages training rounds

2. **Federated Learning Clients** (`fl_client.py`)
   - Represents individual nodes (hospitals/cities)
   - Trains models on local data
   - Participates in federated aggregation
   - Evaluates model performance locally

3. **Health Risk Model** (`health_risk_model.py`)
   - Supports multiple model types (Random Forest, Gradient Boosting, Logistic Regression)
   - Handles feature preparation and scaling
   - Provides model serialization for federated learning
   - Evaluates model performance

## How Federated Learning Works

1. **Initialization**: Server initializes a global model
2. **Round Start**: Server sends current global model parameters to selected clients
3. **Local Training**: Each client trains the model on its local data
4. **Parameter Upload**: Clients send updated parameters back to server
5. **Aggregation**: Server aggregates parameters (weighted average based on data size)
6. **Evaluation**: Clients evaluate the aggregated model on local test data
7. **Repeat**: Steps 2-6 repeat for specified number of rounds

## Usage

### Manual Setup (Testing)

**⚠️ IMPORTANT: Start the SERVER first, then start the CLIENTS in separate terminals!**

#### Step 1: Start the Server (Terminal 1)

```bash
python src/federated_learning/fl_server.py --rounds 10
```

Wait for the server to start and display "Starting Flower server..." before proceeding.

#### Step 2: Start Clients (Terminals 2, 3, 4)

**Terminal 2:**
```bash
python src/federated_learning/fl_client.py --node-name hospital_A
```

**Terminal 3:**
```bash
python src/federated_learning/fl_client.py --node-name hospital_B
```

**Terminal 4:**
```bash
python src/federated_learning/fl_client.py --node-name hospital_C
```

#### Alternative: Use the Helper Script

```bash
python run_federated_learning.py
```

This script will:
- Check if data files exist
- Provide step-by-step instructions
- Optionally start the server for you

### Automated Testing

```bash
# Test model training
python test_federated_learning.py

# Test federated setup
python src/federated_learning/main.py
```

## Model Types

### Random Forest (Default)
- **Pros**: Handles non-linear relationships, feature importance
- **Cons**: Less interpretable, larger model size
- **Federated Learning Note**: Tree models can't be directly averaged like linear models. Each client trains its own trees, and feature importances are aggregated as a proxy. This is a limitation but still provides benefits from local training.

### Gradient Boosting
- **Pros**: High accuracy, handles complex patterns
- **Cons**: Longer training time, more hyperparameters
- **Federated Learning Note**: Similar to Random Forest - trees are trained locally on each client.

### Logistic Regression (Recommended for FedAvg)
- **Pros**: Fast, interpretable, small model size, **true parameter averaging** in federated learning
- **Cons**: Assumes linear relationships
- **Use Case**: Best choice for federated learning as parameters can be directly averaged across clients

## Federated Learning Strategy

### FedAvg (Federated Averaging)
- Standard federated learning algorithm
- Aggregates model parameters based on number of training samples
- Formula: `θ_global = Σ(n_i * θ_i) / Σ(n_i)`
  - Where `n_i` is the number of samples at client `i`
  - And `θ_i` is the model parameters from client `i`

### Configuration
- **fraction_fit**: Fraction of clients used for training (default: 1.0 = all clients)
- **fraction_evaluate**: Fraction of clients used for evaluation (default: 1.0 = all clients)
- **min_fit_clients**: Minimum number of clients for training (default: 2)
- **min_evaluate_clients**: Minimum number of clients for evaluation (default: 2)

## Experiment Tracking

### MLflow Integration
- Tracks metrics per round (accuracy, precision, recall, F1, ROC-AUC)
- Logs aggregated metrics from all clients
- Stores model artifacts
- Enables experiment comparison

### Metrics Tracked
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (if binary classification)

## Data Privacy

### Key Features
- **No Data Sharing**: Raw data never leaves the local node
- **Parameter Only**: Only model parameters are shared
- **Differential Privacy**: Can be extended with differential privacy mechanisms
- **Secure Aggregation**: Parameters are aggregated securely

## Model Serialization

Models are saved in the following format:
- **Location**: `data/models/`
- **Format**: Pickle (`.pkl`)
- **Contents**: Model, scaler, model type, feature names

## Performance Considerations

### Training Time
- Depends on:
  - Number of rounds
  - Number of clients
  - Model type
  - Data size per client
  - Network latency

### Communication Cost
- Only model parameters are transmitted
- For Random Forest: Tree structures (can be large)
- For Logistic Regression: Coefficients (small)
- For Gradient Boosting: Tree structures (can be large)

### Scalability
- Can handle multiple nodes simultaneously
- Server can handle concurrent client connections
- Flower framework handles connection management

## Troubleshooting

### Common Issues

1. **Connection Refused / UNAVAILABLE Error**
   - **Cause**: Server is not running or not ready
   - **Solution**: 
     - Start the server FIRST: `python src/federated_learning/fl_server.py`
     - Wait for server to display "Starting Flower server..." before starting clients
     - Verify server address matches (default: `localhost:8080`)
   
2. **"Trying to connect an http1.x server" Error**
   - **Cause**: Server is not running or wrong port
   - **Solution**: Make sure the Flower server is running, not another service on port 8080

3. **Data Not Found**
   - **Cause**: Phase 1 not run or data files missing
   - **Solution**: 
     - Run Phase 1: `python src/data_ingestion/main.py`
     - Verify data files exist in `data/processed/`
     - Check file paths in client configuration

4. **Model Type Mismatch**
   - **Cause**: Server and clients using different model types
   - **Solution**: All clients must use the same model type as the server

5. **Deprecation Warnings**
   - **Cause**: Flower API changes
   - **Solution**: The code handles both old and new APIs automatically. Warnings are safe to ignore.

6. **Import Errors**
   - **Cause**: Missing dependencies or wrong Python path
   - **Solution**: 
     - Install dependencies: `pip install -r requirements.txt`
     - Check Python path includes `src/` directory

## Next Steps

After completing Phase 2:
- **Phase 3**: Model evaluation and comparison
- **Phase 4**: MLOps pipeline integration
- **Phase 5**: Data drift detection
- **Phase 6**: Deployment and monitoring

## References

- Flower Framework: https://flower.dev/
- FedAvg Paper: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
- MLflow Documentation: https://mlflow.org/docs/latest/index.html

