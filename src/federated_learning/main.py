"""
Federated Learning Main Script
Orchestrates federated learning training across multiple nodes
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def run_federated_learning(
    nodes: List[str] = None,
    num_rounds: int = 10,
    model_type: str = 'random_forest',
    server_address: str = "localhost:8080",
    mlflow_tracking_uri: Optional[str] = None,
    run_in_background: bool = False
):
    """
    Run federated learning training
    
    Args:
        nodes: List of node names (default: ['hospital_A', 'hospital_B', 'hospital_C'])
        num_rounds: Number of federated learning rounds
        model_type: Type of model to train
        server_address: Server address
        mlflow_tracking_uri: MLflow tracking URI
        run_in_background: Whether to run clients in background (for testing)
    """
    if nodes is None:
        nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    
    print("=" * 80)
    print("PHASE 2: FEDERATED LEARNING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Nodes: {nodes}")
    print(f"  Model type: {model_type}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Server address: {server_address}")
    print(f"  MLflow tracking: {mlflow_tracking_uri or 'Disabled'}")
    print("\n" + "-" * 80)
    
    # Verify data files exist
    print("\n[1/3] Verifying data files...")
    missing_files = []
    for node in nodes:
        data_path = f"data/processed/{node}_merged_labeled.csv"
        if not os.path.exists(data_path):
            missing_files.append(data_path)
            print(f"  ✗ Missing: {data_path}")
        else:
            print(f"  ✓ Found: {data_path}")
    
    if missing_files:
        print(f"\n✗ Error: Missing data files. Please run Phase 1 first.")
        return
    
    # Instructions for running
    print("\n[2/3] Starting Federated Learning...")
    print("\nTo run federated learning, you need to:")
    print("1. Start the server in one terminal:")
    print(f"   python src/federated_learning/fl_server.py --address {server_address} --rounds {num_rounds} --model-type {model_type}")
    if mlflow_tracking_uri:
        print(f"   --mlflow-uri {mlflow_tracking_uri}")
    
    print("\n2. Start each client in separate terminals:")
    for node in nodes:
        print(f"   python src/federated_learning/fl_client.py --node-name {node} --server-address {server_address} --model-type {model_type}")
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING SETUP COMPLETE")
    print("=" * 80)
    print("\nNote: For automated testing, see test_federated_learning.py")
    print("      For production, use Docker Compose or Kubernetes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Main")
    parser.add_argument("--nodes", nargs='+', default=['hospital_A', 'hospital_B', 'hospital_C'],
                       help="List of node names")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       help="Model type")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    run_federated_learning(
        nodes=args.nodes,
        num_rounds=args.rounds,
        model_type=args.model_type,
        server_address=args.server_address,
        mlflow_tracking_uri=args.mlflow_uri
    )

