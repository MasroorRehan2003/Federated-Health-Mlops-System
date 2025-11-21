"""
Air Quality Sensor Data Generator
Generates realistic air quality data (AQI, PM2.5, PM10) at city level
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict


class AirQualityGenerator:
    """Generate simulated air quality sensor data"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        # City-specific baseline air quality (simulating different pollution levels)
        self.city_baselines = {
            'hospital_A': {'pm25_base': 35, 'pm10_base': 50, 'aqi_base': 60},  # Moderate pollution
            'hospital_B': {'pm25_base': 25, 'pm10_base': 40, 'aqi_base': 45},  # Good air quality
            'hospital_C': {'pm25_base': 45, 'pm10_base': 65, 'aqi_base': 75},  # High pollution
        }
    
    def calculate_aqi(self, pm25: float, pm10: float) -> float:
        """
        Calculate Air Quality Index (AQI) from PM2.5 and PM10
        
        Simplified AQI calculation based on EPA standards
        """
        # AQI is typically the maximum of PM2.5 AQI and PM10 AQI
        # Simplified version: weighted average
        
        # PM2.5 AQI breakpoints (simplified)
        if pm25 <= 12:
            aqi_pm25 = (pm25 / 12) * 50
        elif pm25 <= 35.4:
            aqi_pm25 = 50 + ((pm25 - 12) / (35.4 - 12)) * 50
        elif pm25 <= 55.4:
            aqi_pm25 = 100 + ((pm25 - 35.4) / (55.4 - 35.4)) * 50
        elif pm25 <= 150.4:
            aqi_pm25 = 150 + ((pm25 - 55.4) / (150.4 - 55.4)) * 100
        else:
            aqi_pm25 = 250 + ((pm25 - 150.4) / (250.4 - 150.4)) * 150
        
        # PM10 AQI breakpoints (simplified)
        if pm10 <= 54:
            aqi_pm10 = (pm10 / 54) * 50
        elif pm10 <= 154:
            aqi_pm10 = 50 + ((pm10 - 54) / (154 - 54)) * 50
        elif pm10 <= 254:
            aqi_pm10 = 100 + ((pm10 - 154) / (254 - 154)) * 50
        elif pm10 <= 354:
            aqi_pm10 = 150 + ((pm10 - 254) / (354 - 254)) * 100
        else:
            aqi_pm10 = 250 + ((pm10 - 354) / (424 - 354)) * 150
        
        # AQI is the maximum of the two
        aqi = max(aqi_pm25, aqi_pm10)
        return min(500, max(0, aqi))  # Clamp between 0-500
    
    def generate_node_data(
        self,
        node_name: str,
        start_date: datetime = None,
        days: int = 30,
        output_dir: str = 'data/raw'
    ) -> str:
        """
        Generate air quality data for a node (city)
        
        Args:
            node_name: Name of the node (e.g., 'hospital_A')
            start_date: Start date (defaults to 30 days ago)
            days: Number of days to generate
            output_dir: Output directory for CSV files
        
        Returns:
            Path to saved CSV file
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        end_date = start_date + timedelta(days=days)
        
        # Get baseline for this city
        baseline = self.city_baselines.get(node_name, {'pm25_base': 30, 'pm10_base': 45, 'aqi_base': 50})
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        
        data = []
        
        for timestamp in timestamps:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Air quality varies by time of day and day of week
            # Higher pollution during rush hours (7-9 AM, 5-7 PM)
            # Lower pollution on weekends
            
            # Time of day factor
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                time_factor = 1.3
            elif 10 <= hour <= 16:  # Daytime
                time_factor = 1.1
            else:  # Night/early morning
                time_factor = 0.9
            
            # Day of week factor (weekends have less traffic)
            if day_of_week >= 5:  # Weekend
                day_factor = 0.85
            else:  # Weekday
                day_factor = 1.0
            
            # Add some random variation and trends
            trend = np.sin(2 * np.pi * timestamp.hour / 24) * 0.1  # Daily cycle
            random_variation = np.random.normal(0, 0.15)
            
            # Calculate PM2.5 and PM10
            pm25 = baseline['pm25_base'] * time_factor * day_factor * (1 + trend + random_variation)
            pm10 = baseline['pm10_base'] * time_factor * day_factor * (1 + trend + random_variation * 0.8)
            
            # Ensure positive values
            pm25 = max(0, pm25)
            pm10 = max(0, pm10)
            
            # Calculate AQI
            aqi = self.calculate_aqi(pm25, pm10)
            
            # CO2 levels (ppm) - correlated with air quality
            co2_base = 400  # Normal atmospheric CO2
            co2 = co2_base + (aqi / 10) * np.random.normal(1, 0.2)
            co2 = max(400, min(1000, co2))
            
            data.append({
                'timestamp': timestamp,
                'location': node_name,
                'pm25': round(pm25, 2),
                'pm10': round(pm10, 2),
                'aqi': round(aqi, 1),
                'co2': round(co2, 1)
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{node_name}_air_quality.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✓ Generated air quality data for {node_name}: {len(df)} records")
        print(f"  Average AQI: {df['aqi'].mean():.1f}, PM2.5: {df['pm25'].mean():.1f} μg/m³")
        
        return output_path


if __name__ == "__main__":
    generator = AirQualityGenerator(seed=42)
    
    # Generate data for three nodes
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    
    for node in nodes:
        generator.generate_node_data(
            node_name=node,
            days=30
        )

