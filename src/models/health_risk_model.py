"""
Health Risk Prediction Model
Combines multiple data types (time series features) for health risk prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from typing import Tuple, Optional, Dict


class HealthRiskModel:
    """Health risk prediction model supporting federated learning"""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize the model
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                solver='lbfgs'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame, feature_mapping: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from dataframe
        
        Args:
            df: DataFrame with features and 'health_risk' label
            feature_mapping: Optional dict mapping categorical columns to all possible values
        
        Returns:
            Tuple of (X, y) arrays
        """
        # Select feature columns (exclude non-feature columns)
        exclude_cols = ['timestamp', 'location', 'health_risk', 'risk_category', 
                       'user_id', 'node', 'date']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical features with consistent encoding
        df_processed = df.copy()
        
        # Define all possible categorical values to ensure consistency
        if feature_mapping is None:
            feature_mapping = {
                'aqi_category': ['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy', 'Hazardous'],
                'temp_category': ['Cold', 'Moderate', 'Warm', 'Hot']
            }
        
        # One-hot encode categorical features with all possible values
        for col in feature_cols:
            if df_processed[col].dtype == 'object':
                if col in feature_mapping:
                    # Use predefined categories to ensure consistency
                    categories = feature_mapping[col]
                    for cat in categories:
                        df_processed[f'{col}_{cat}'] = (df_processed[col] == cat).astype(int)
                    df_processed = df_processed.drop(columns=[col])
                else:
                    # For other categorical columns, use get_dummies but ensure consistency
                    df_processed = pd.get_dummies(df_processed, columns=[col], prefix=col)
        
        # Update feature columns after encoding
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        
        # Sort columns to ensure consistent order
        feature_cols = sorted(feature_cols)
        
        # Ensure all expected features exist (add missing ones with zeros)
        if self.feature_names is not None:
            # Use existing feature names if available
            for feat in self.feature_names:
                if feat not in df_processed.columns:
                    df_processed[feat] = 0
            feature_cols = [f for f in self.feature_names if f in df_processed.columns]
        else:
            # Store feature names for future use
            self.feature_names = feature_cols
        
        # Select only the feature columns in the correct order
        X = df_processed[feature_cols].values
        y = df_processed['health_risk'].values if 'health_risk' in df_processed.columns else None
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, scale_features: bool = True):
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target labels
            scale_features: Whether to scale features
        """
        if scale_features:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X) if hasattr(self.scaler, 'mean_') else X
        
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            scale_features: Whether to scale features
        
        Returns:
            Predictions
        """
        if scale_features:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            scale_features: Whether to scale features
        
        Returns:
            Prediction probabilities
        """
        if scale_features:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, scale_features: bool = True) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X: Feature matrix
            y: True labels
            scale_features: Whether to scale features
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X, scale_features=scale_features)
        y_proba = self.predict_proba(X, scale_features=scale_features)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
        }
        
        # ROC AUC (handle case where only one class is present)
        try:
            if len(np.unique(y)) > 1:
                metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics['roc_auc'] = 0.0
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def get_model_parameters(self) -> Dict:
        """
        Get model parameters (for federated learning aggregation)
        
        Returns:
            Dictionary of model parameters
        """
        if self.model_type == 'logistic_regression':
            return {
                'coef_': self.model.coef_,
                'intercept_': self.model.intercept_,
                'scaler_mean_': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale_': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
            }
        elif self.model_type in ['random_forest', 'gradient_boosting']:
            # For tree-based models, we return the model itself
            return {
                'model': self.model,
                'scaler_mean_': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale_': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
            }
        else:
            return {}
    
    def set_model_parameters(self, params: Dict):
        """
        Set model parameters (for federated learning aggregation)
        
        Args:
            params: Dictionary of model parameters
        """
        if self.model_type == 'logistic_regression':
            if 'coef_' in params:
                self.model.coef_ = params['coef_']
            if 'intercept_' in params:
                self.model.intercept_ = params['intercept_']
        elif self.model_type in ['random_forest', 'gradient_boosting']:
            if 'model' in params:
                self.model = params['model']
        
        # Set scaler parameters
        if 'scaler_mean_' in params and params['scaler_mean_'] is not None:
            self.scaler.mean_ = params['scaler_mean_']
        if 'scaler_scale_' in params and params['scaler_scale_'] is not None:
            self.scaler.scale_ = params['scaler_scale_']
    
    def save(self, filepath: str):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load the model from disk
        
        Args:
            filepath: Path to load the model from
        
        Returns:
            Loaded HealthRiskModel instance
        """
        data = joblib.load(filepath)
        model = cls(model_type=data['model_type'])
        model.model = data['model']
        model.scaler = data['scaler']
        model.feature_names = data.get('feature_names')
        return model

