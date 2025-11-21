"""
Federated Learning Client
Represents a node (hospital/city) in the federated learning system
"""

import flwr as fl
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.health_risk_model import HealthRiskModel


class FederatedClient(fl.client.NumPyClient):
    """Federated Learning Client for a single node"""
    
    def __init__(
        self,
        node_name: str,
        data_path: str,
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.node_name = node_name
        self.data_path = data_path
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize model
        self.model = HealthRiskModel(model_type=model_type, random_state=random_state)
        
        # Load and prepare data
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()
        
        # Local initialization training
        self.model.train(self.X_train, self.y_train, scale_features=True)
        
        print(f"✓ Client {node_name} initialized")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
    
    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Feature mapping is consistent across all nodes
        feature_mapping = self._get_feature_mapping()
        
        X, y = self.model.prepare_features(df, feature_mapping=feature_mapping)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def _get_feature_mapping(self):
        nodes = ["hospital_A", "hospital_B", "hospital_C"]
        dfs = []

        for node in nodes:
            p = f"data/processed/{node}_merged_labeled.csv"
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))

        if not dfs:
            return {
                "aqi_category": ['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy', 'Hazardous'],
                "temp_category": ['Cold', 'Moderate', 'Warm', 'Hot']
            }

        combined = pd.concat(dfs, ignore_index=True)
        mapping = {}
        for col in ["aqi_category", "temp_category"]:
            if col in combined.columns:
                mapping[col] = sorted(combined[col].dropna().unique().tolist())
        return mapping

    # -------------------- Flower Interface --------------------

    def get_parameters(self, config):
        params = self.model.get_model_parameters()

        if self.model_type == "logistic_regression":
            return [params["coef_"].flatten(), params["intercept_"]]

        model = self.model.model
        if hasattr(model, "feature_importances_"):
            if hasattr(self.model.scaler, "mean_"):
                return [model.feature_importances_, self.model.scaler.mean_, self.model.scaler.scale_]
            return [model.feature_importances_]

        n_features = self.X_train.shape[1]
        return [np.ones(n_features) / n_features]

    def set_parameters(self, parameters):
        if self.model_type == "logistic_regression":
            coef = np.array(parameters[0]).reshape(1, -1)
            intercept = np.array(parameters[1])
            self.model.set_model_parameters({"coef_": coef, "intercept_": intercept})
            return

        # Tree models – scaler only
        if len(parameters) >= 3:
            try:
                self.model.scaler.mean_ = np.array(parameters[1])
                self.model.scaler.scale_ = np.array(parameters[2])
            except:
                pass

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train(self.X_train, self.y_train, scale_features=True)

        updated_params = self.get_parameters(config)
        metrics = self.model.evaluate(self.X_train, self.y_train, scale_features=True)

        return updated_params, len(self.X_train), {
            "node_name": self.node_name,
            "train_accuracy": metrics["accuracy"],
            "train_f1": metrics["f1"],
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        metrics = self.model.evaluate(self.X_test, self.y_test, scale_features=True)
        loss = 1 - metrics["accuracy"]

        return loss, len(self.X_test), {
            "node_name": self.node_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }

# ---------------------- CLIENT STARTER ----------------------

def start_client(
    node_name: str,
    server_address: str = "fl-server:8080",   # FIXED for Docker
    data_path: Optional[str] = None,
    model_type: str = "random_forest",
):
    # Docker override
    server_address = os.getenv("SERVER_ADDRESS", server_address)

    if data_path is None:
        data_path = f"data/processed/{node_name}_merged_labeled.csv"

    print("=" * 80)
    print(f"FEDERATED LEARNING CLIENT: {node_name}")
    print("=" * 80)
    print(f"Server address: {server_address}")
    print(f"Data path: {data_path}")
    print(f"Model type: {model_type}")
    print("=" * 80)

    if not os.path.exists(data_path):
        print(f"✗ Error: Data not found: {data_path}")
        return

    client = FederatedClient(
        node_name=node_name,
        data_path=data_path,
        model_type=model_type
    )

    try:
        print("\nConnecting to Flower server...")
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
    except Exception as e:
        print(f"✗ Error connecting to server: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--server-address", default="fl-server:8080")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-type", default="random_forest")

    args = parser.parse_args()

    start_client(
        node_name=args.node_name,
        server_address=args.server_address,
        data_path=args.data_path,
        model_type=args.model_type
    )
