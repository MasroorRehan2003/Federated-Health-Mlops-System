"""
Dashboard utilities for data loading and processing
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List
from datetime import datetime, timedelta


def load_node_data(node_name: str) -> Optional[pd.DataFrame]:
    """Load labeled data for a specific node"""
    data_path = f"data/processed/{node_name}_merged_labeled.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


def load_all_nodes_data() -> pd.DataFrame:
    """Load combined data from all nodes"""
    data_path = "data/processed/all_nodes_merged_labeled.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    # If combined file doesn't exist, merge individual files
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    dfs = []
    for node in nodes:
        node_df = load_node_data(node)
        if node_df is not None:
            dfs.append(node_df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def get_node_summary(node_name: str) -> Dict:
    """Get summary statistics for a node"""
    df = load_node_data(node_name)
    if df is None or df.empty:
        return {}
    
    summary = {
        'node_name': node_name,
        'total_records': len(df),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'high_risk_percentage': (df['health_risk'].sum() / len(df)) * 100 if 'health_risk' in df.columns else 0,
        'avg_aqi': df['aqi'].mean() if 'aqi' in df.columns else 0,
        'avg_pm25': df['pm25'].mean() if 'pm25' in df.columns else 0,
        'avg_temperature': df['temperature'].mean() if 'temperature' in df.columns else 0,
        'avg_heart_rate': df['heart_rate_mean'].mean() if 'heart_rate_mean' in df.columns else 0,
    }
    return summary


def get_regional_comparison() -> pd.DataFrame:
    """Get comparison statistics across all regions"""
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    summaries = []
    
    for node in nodes:
        summary = get_node_summary(node)
        if summary:
            summaries.append(summary)
    
    if summaries:
        return pd.DataFrame(summaries)
    return pd.DataFrame()


def calculate_risk_trends(df: pd.DataFrame, node_name: Optional[str] = None) -> pd.DataFrame:
    """Calculate daily risk trends"""
    if df.empty:
        return pd.DataFrame()
    
    df['date'] = df['timestamp'].dt.date
    daily_stats = df.groupby('date').agg({
        'health_risk': ['sum', 'count', 'mean'],
        'aqi': 'mean',
        'pm25': 'mean',
        'temperature': 'mean',
        'heart_rate_mean': 'mean'
    }).reset_index()
    
    daily_stats.columns = ['date', 'high_risk_count', 'total_count', 'risk_percentage', 
                          'avg_aqi', 'avg_pm25', 'avg_temperature', 'avg_heart_rate']
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    daily_stats['risk_percentage'] = daily_stats['risk_percentage'] * 100
    
    if node_name:
        daily_stats['node'] = node_name
    
    return daily_stats


def get_alerts(df: pd.DataFrame, threshold_high_risk: float = 0.3) -> List[Dict]:
    """Generate alerts based on current conditions"""
    alerts = []
    
    if df.empty:
        return alerts
    
    # Get latest data point
    latest = df.iloc[-1] if not df.empty else None
    if latest is None:
        return alerts
    
    # Recent high risk rate (last 24 hours)
    recent_24h = df[df['timestamp'] >= (df['timestamp'].max() - pd.Timedelta(hours=24))]
    if not recent_24h.empty:
        recent_risk_rate = recent_24h['health_risk'].mean()
        
        if recent_risk_rate > threshold_high_risk:
            alerts.append({
                'type': 'high_risk',
                'severity': 'high' if recent_risk_rate > 0.5 else 'medium',
                'message': f'High health risk detected: {recent_risk_rate*100:.1f}% of recent cases show elevated risk',
                'location': latest.get('location', 'Unknown'),
                'timestamp': latest['timestamp']
            })
    
    # Air quality alerts
    if 'aqi' in latest and latest['aqi'] > 150:
        alerts.append({
            'type': 'air_quality',
            'severity': 'high' if latest['aqi'] > 200 else 'medium',
            'message': f'Poor air quality: AQI = {latest["aqi"]:.0f}',
            'location': latest.get('location', 'Unknown'),
            'timestamp': latest['timestamp']
        })
    
    # PM2.5 alerts
    if 'pm25' in latest and latest['pm25'] > 50:
        alerts.append({
            'type': 'pm25',
            'severity': 'high' if latest['pm25'] > 75 else 'medium',
            'message': f'High PM2.5: {latest["pm25"]:.1f} μg/m³',
            'location': latest.get('location', 'Unknown'),
            'timestamp': latest['timestamp']
        })
    
    # Temperature alerts
    if 'temperature' in latest:
        if latest['temperature'] > 35:
            alerts.append({
                'type': 'temperature',
                'severity': 'medium',
                'message': f'High temperature: {latest["temperature"]:.1f}°C',
                'location': latest.get('location', 'Unknown'),
                'timestamp': latest['timestamp']
            })
        elif latest['temperature'] < 5:
            alerts.append({
                'type': 'temperature',
                'severity': 'medium',
                'message': f'Low temperature: {latest["temperature"]:.1f}°C',
                'location': latest.get('location', 'Unknown'),
                'timestamp': latest['timestamp']
            })
    
    return alerts


def get_risk_category(risk_value: float) -> str:
    """Categorize risk value"""
    if risk_value >= 0.7:
        return "High Risk"
    elif risk_value >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


def format_timestamp(ts) -> str:
    """Format timestamp for display"""
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d %H:%M")
    return str(ts)

