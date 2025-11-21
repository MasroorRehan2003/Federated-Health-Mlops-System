"""
Federated Learning Server
Coordinates federated training across multiple nodes using Flower framework
"""

import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
import os
from collections import OrderedDict
import mlflow
import mlflow.sklearn
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.health_risk_model import HealthRiskModel


class FederatedServer(fl.server.strategy.FedAvg):
    """Federated Learning Server using FedAvg strategy"""
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "federated_learning"
    ):
        self.model_type = model_type
        self.global_model = None
        self.round_num = 0
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=self.get_fit_config,
            on_evaluate_config_fn=self.get_evaluate_config,
            evaluate_metrics_aggregation_fn=self.aggregate_evaluate_metrics
        )
    
    def get_fit_config(self, server_round: int) -> Dict:
        self.round_num = server_round
        return {"server_round": server_round, "learning_rate": 0.1}
    
    def get_evaluate_config(self, server_round: int) -> Dict:
        return {"server_round": server_round}
    
    def aggregate_evaluate_metrics(
        self,
        metrics: List[Tuple[int, Dict]]
    ) -> Dict:
        total_samples = sum([num_samples for num_samples, _ in metrics])
        aggregated_metrics = {}
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            weighted_sum = sum([
                num_samples * m.get(metric_name, 0)
                for num_samples, m in metrics
            ])
            aggregated_metrics[metric_name] = (
                weighted_sum / total_samples if total_samples > 0 else 0
            )
        
        aggregated_metrics['total_samples'] = total_samples
        aggregated_metrics['num_clients'] = len(metrics)
        
        if self.mlflow_tracking_uri:
            with mlflow.start_run(run_name=f"round_{self.round_num}"):
                mlflow.log_metrics(aggregated_metrics, step=self.round_num)
        
        print(f"\n[Round {self.round_num}] Aggregated Metrics:")
        for key, value in aggregated_metrics.items():
            if key not in ['total_samples', 'num_clients']:
                print(f"  {key}: {value:.4f}")
        
        return aggregated_metrics


def check_port_available(host: str, port: int) -> bool:
    """Docker-safe port check"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
            return True
    except OSError:
        return False


def start_server(
    server_address: str = "0.0.0.0:8080",   # FIXED
    num_rounds: int = 10,
    model_type: str = 'random_forest',
    mlflow_tracking_uri: Optional[str] = None
):
    if ':' in server_address:
        host, port_str = server_address.rsplit(':', 1)
        port = int(port_str)
    else:
        host = server_address
        port = 8080
    
    if not check_port_available(host, port):
        print(f"\n✗ Error: Port {port} is already in use!")
        return
    
    print("=" * 80)
    print("FEDERATED LEARNING SERVER")
    print("=" * 80)
    print(f"Server address: {server_address}")
    print(f"Model type: {model_type}")
    print(f"Number of rounds: {num_rounds}")
    print(f"MLflow tracking: {mlflow_tracking_uri or 'Disabled'}")
    print("=" * 80)
    
    strategy = FederatedServer(
        model_type=model_type,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name="federated_health_risk_prediction"
    )
    
    try:
        print(f"\nStarting server on {server_address}...")
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )
    except RuntimeError as e:
        print(f"\n✗ Error binding to {server_address}")
        print(f"  {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server address")  # FIXED
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression'])
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        model_type=args.model_type,
        mlflow_tracking_uri=args.mlflow_uri
    )
