"""
Test script for Federated Learning
Tests the federated learning setup with a simple scenario
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.health_risk_model import HealthRiskModel


def test_model_training():
    """Test model training on a single node"""
    print("Testing Model Training...")
    print("=" * 80)
    
    # Load data
    data_path = "data/processed/hospital_A_merged_labeled.csv"
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("  Please run Phase 1 first: python src/data_ingestion/main.py")
        return False
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded data: {len(df)} records")
    
    # Initialize model
    model = HealthRiskModel(model_type='random_forest', n_estimators=50)
    
    # Prepare features
    X, y = model.prepare_features(df)
    print(f"✓ Prepared features: {X.shape}")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Split data: Train={len(X_train)}, Test={len(X_test)}")
    
    # Train model
    print("\nTraining model...")
    model.train(X_train, y_train, scale_features=True)
    print("✓ Model trained")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test, scale_features=True)
    
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test model saving/loading
    model_path = "data/models/test_model.pkl"
    model.save(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    loaded_model = HealthRiskModel.load(model_path)
    print("✓ Model loaded successfully")
    
    # Verify loaded model works
    metrics_loaded = loaded_model.evaluate(X_test, y_test, scale_features=True)
    print("\nLoaded Model Metrics:")
    for key, value in metrics_loaded.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ MODEL TRAINING TEST PASSED")
    print("=" * 80)
    
    return True


def test_federated_setup():
    """Test federated learning setup (without actually running FL)"""
    print("\nTesting Federated Learning Setup...")
    print("=" * 80)
    
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    all_exist = True
    
    for node in nodes:
        data_path = f"data/processed/{node}_merged_labeled.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✓ {node}: {len(df)} records")
        else:
            print(f"✗ {node}: Data file not found")
            all_exist = False
    
    if all_exist:
        print("\n" + "=" * 80)
        print("✓ FEDERATED LEARNING SETUP TEST PASSED")
        print("=" * 80)
        print("\nTo run federated learning:")
        print("1. Start server: python src/federated_learning/fl_server.py")
        print("2. Start clients: python src/federated_learning/fl_client.py --node-name <node>")
    else:
        print("\n✗ Some data files are missing. Please run Phase 1 first.")
    
    return all_exist


if __name__ == "__main__":
    print("=" * 80)
    print("FEDERATED LEARNING TESTS")
    print("=" * 80)
    
    success1 = test_model_training()
    success2 = test_federated_setup()
    
    if success1 and success2:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        sys.exit(1)

