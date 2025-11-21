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
        """
        Initialize federated learning client
        
        Args:
            node_name: Name of the node (e.g., 'hospital_A')
            data_path: Path to labeled data CSV file
            model_type: Type of model to train
            test_size: Fraction of data to use for testing
            random_state: Random seed
        """
        self.node_name = node_name
        self.data_path = data_path
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize model first (needed for feature consistency)
        self.model = HealthRiskModel(model_type=model_type, random_state=random_state)
        
        # Load and prepare data (this will use consistent feature mapping)
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()
        
        # Train initial model
        self.model.train(self.X_train, self.y_train, scale_features=True)
        
        print(f"✓ Client {node_name} initialized")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Prepare features - use consistent feature mapping
        # First, load all nodes to determine all possible categorical values
        feature_mapping = self._get_feature_mapping()
        
        # Use the model's feature names if already set (for consistency across rounds)
        if self.model.feature_names is not None:
            # Use existing feature names to ensure consistency
            X, y = self.model.prepare_features(df, feature_mapping=feature_mapping)
        else:
            # First time: prepare features and store feature names
            X, y = self.model.prepare_features(df, feature_mapping=feature_mapping)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def _get_feature_mapping(self) -> dict:
        """Get consistent feature mapping across all nodes"""
        # Load all node data to determine all possible categorical values
        nodes = ['hospital_A', 'hospital_B', 'hospital_C']
        all_dfs = []
        
        for node in nodes:
            node_path = f"data/processed/{node}_merged_labeled.csv"
            if os.path.exists(node_path):
                df = pd.read_csv(node_path)
                all_dfs.append(df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Determine all unique values for categorical columns
            feature_mapping = {}
            categorical_cols = ['aqi_category', 'temp_category']
            
            for col in categorical_cols:
                if col in combined_df.columns:
                    feature_mapping[col] = sorted(combined_df[col].dropna().unique().tolist())
            
            return feature_mapping
        
        # Default mapping if files don't exist
        return {
            'aqi_category': ['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy', 'Hazardous'],
            'temp_category': ['Cold', 'Moderate', 'Warm', 'Hot']
        }
    
    def get_parameters(self, config: Dict) -> list:
        """Return current model parameters"""
        params = self.model.get_model_parameters()
        
        if self.model_type == 'logistic_regression':
            # Return as numpy arrays for Flower
            return [
                params['coef_'].flatten(),
                params['intercept_']
            ]
        else:
            # For tree-based models, we use a simplified approach:
            # Return feature importances as a proxy for model parameters
            # This allows Flower to aggregate "parameters" across clients
            # Note: Tree models can't be truly federated this way, but this allows the framework to work
            model = self.model.model
            
            if hasattr(model, 'feature_importances_'):
                # Return feature importances as the "parameters"
                importances = model.feature_importances_
                # Also include scaler parameters if available
                if hasattr(self.model.scaler, 'mean_') and self.model.scaler.mean_ is not None:
                    return [
                        importances,
                        self.model.scaler.mean_,
                        self.model.scaler.scale_
                    ]
                return [importances]
            else:
                # Fallback: return dummy parameters
                n_features = self.X_train.shape[1]
                return [np.ones(n_features) / n_features]  # Uniform importance
    
    def set_parameters(self, parameters: list) -> None:
        """Set model parameters from server"""
        if self.model_type == 'logistic_regression':
            # Reconstruct parameters
            coef_flat = parameters[0]
            intercept_ = parameters[1]
            
            # Handle numpy array conversion
            if not isinstance(coef_flat, np.ndarray):
                coef_flat = np.array(coef_flat)
            if not isinstance(intercept_, np.ndarray):
                intercept_ = np.array(intercept_)
            
            # Reshape coefficients
            n_features = len(coef_flat)
            coef_ = coef_flat.reshape(1, -1)
            
            params = {
                'coef_': coef_,
                'intercept_': intercept_
            }
            self.model.set_model_parameters(params)
        else:
            # For tree-based models, we can't directly set aggregated tree structures
            # Instead, we accept the aggregated feature importances but don't modify the model
            # The model will be retrained on local data in the fit() method
            # This is a limitation of federating tree-based models
            
            # Handle scaler parameters if provided
            if len(parameters) >= 3:
                try:
                    # Parameters include: importances, scaler mean, scaler scale
                    if hasattr(self.model.scaler, 'mean_'):
                        self.model.scaler.mean_ = np.array(parameters[1])
                        self.model.scaler.scale_ = np.array(parameters[2])
                except Exception as e:
                    print(f"Warning: Could not update scaler parameters: {e}")
            
            # Note: We don't update the tree model itself here
            # Tree models will be retrained on local data in fit()
            # The "parameters" from get_parameters are just feature importances for aggregation
            pass
    
    def fit(self, parameters: list, config: Dict) -> Tuple[list, int, Dict]:
        """Train model on local data"""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Train on local data
        self.model.train(self.X_train, self.y_train, scale_features=True)
        
        # Get updated parameters
        updated_params = self.get_parameters(config)
        
        # Return parameters, number of samples, and metrics
        num_samples = len(self.X_train)
        
        # Calculate training metrics
        train_metrics = self.model.evaluate(self.X_train, self.y_train, scale_features=True)
        
        return updated_params, num_samples, {
            "node_name": self.node_name,
            "train_accuracy": train_metrics['accuracy'],
            "train_f1": train_metrics['f1']
        }
    
    def evaluate(self, parameters: list, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local test data"""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Evaluate on test data
        metrics = self.model.evaluate(self.X_test, self.y_test, scale_features=True)
        
        # Return loss (1 - accuracy), number of samples, and metrics
        loss = 1.0 - metrics['accuracy']
        num_samples = len(self.X_test)
        
        return loss, num_samples, {
            "node_name": self.node_name,
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "roc_auc": metrics['roc_auc']
        }


def start_client(
    node_name: str,
    server_address: str = "localhost:8080",
    data_path: Optional[str] = None,
    model_type: str = 'random_forest'
):
    """
    Start a federated learning client
    
    Args:
        node_name: Name of the node
        server_address: Server address (host:port)
        data_path: Path to labeled data (defaults to data/processed/{node_name}_merged_labeled.csv)
        model_type: Type of model to train
    """
    if data_path is None:
        data_path = f"data/processed/{node_name}_merged_labeled.csv"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"✗ Error: Data file not found: {data_path}")
        print("  Please run Phase 1 first: python src/data_ingestion/main.py")
        return
    
    print("=" * 80)
    print(f"FEDERATED LEARNING CLIENT: {node_name}")
    print("=" * 80)
    print(f"Server address: {server_address}")
    print(f"Data path: {data_path}")
    print(f"Model type: {model_type}")
    print("=" * 80)
    print("\n⚠️  Make sure the server is running before starting the client!")
    print(f"   Start server with: python src/federated_learning/fl_server.py")
    print("=" * 80)
    
    # Create client
    client = FederatedClient(
        node_name=node_name,
        data_path=data_path,
        model_type=model_type
    )
    
    # Start client - try new API first, fallback to old
    try:
        # Try new Flower API (v1.5+)
        if hasattr(client, 'to_client'):
            print("\nConnecting to server using new Flower API...")
            fl.client.start_client(
                server_address=server_address,
                client=client.to_client()
            )
        else:
            # Use old API
            print("\nConnecting to server using legacy Flower API...")
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client
            )
    except Exception as e:
        print(f"\n✗ Error connecting to server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the server is running:")
        print("   python src/federated_learning/fl_server.py")
        print("2. Check the server address is correct")
        print("3. Check firewall/network settings")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--node-name", type=str, required=True, help="Node name")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data file")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       help="Model type")
    
    args = parser.parse_args()
    
    start_client(
        node_name=args.node_name,
        server_address=args.server_address,
        data_path=args.data_path,
        model_type=args.model_type
    )

