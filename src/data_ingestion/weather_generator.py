"""
Weather Data Generator
Generates realistic weather data (temperature, humidity, wind speed, pressure)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict


class WeatherDataGenerator:
    """Generate simulated weather data"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        # City-specific climate baselines (simulating different geographic locations)
        self.city_climates = {
            'hospital_A': {
                'temp_base': 22,  # Moderate climate
                'temp_range': 8,
                'humidity_base': 65,
                'wind_base': 12
            },
            'hospital_B': {
                'temp_base': 18,  # Cooler climate
                'temp_range': 6,
                'humidity_base': 70,
                'wind_base': 15
            },
            'hospital_C': {
                'temp_base': 28,  # Warmer climate
                'temp_range': 10,
                'humidity_base': 60,
                'wind_base': 10
            },
        }
    
    def generate_node_data(
        self,
        node_name: str,
        start_date: datetime = None,
        days: int = 30,
        output_dir: str = 'data/raw'
    ) -> str:
        """
        Generate weather data for a node (city)
        
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
        
        # Get climate baseline for this city
        climate = self.city_climates.get(
            node_name,
            {'temp_base': 20, 'temp_range': 8, 'humidity_base': 65, 'wind_base': 12}
        )
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday
            
            # Temperature: varies by time of day and season
            # Daily cycle: lowest at 4 AM, highest at 2 PM
            daily_temp_variation = np.sin(2 * np.pi * (hour - 4) / 24) * climate['temp_range']
            
            # Seasonal variation (simplified)
            seasonal_variation = np.sin(2 * np.pi * day_of_year / 365) * 5
            
            # Random weather events
            random_temp = np.random.normal(0, 2)
            
            temperature = climate['temp_base'] + daily_temp_variation + seasonal_variation + random_temp
            
            # Humidity: inversely correlated with temperature, higher at night
            if 2 <= hour <= 6:  # Early morning (highest humidity)
                humidity_base = climate['humidity_base'] + 15
            elif 12 <= hour <= 16:  # Afternoon (lowest humidity)
                humidity_base = climate['humidity_base'] - 10
            else:
                humidity_base = climate['humidity_base']
            
            # Humidity decreases with temperature
            temp_factor = (temperature - climate['temp_base']) * -0.5
            humidity = humidity_base + temp_factor + np.random.normal(0, 5)
            humidity = max(20, min(100, humidity))
            
            # Wind speed: varies throughout the day, higher during day
            if 10 <= hour <= 18:
                wind_base = climate['wind_base'] * 1.3
            else:
                wind_base = climate['wind_base'] * 0.8
            
            wind_speed = wind_base + np.random.normal(0, 3)
            wind_speed = max(0, min(50, wind_speed))
            
            # Atmospheric pressure: varies with weather patterns
            # Normal range: 980-1050 hPa
            pressure_base = 1013.25  # Standard atmospheric pressure
            pressure = pressure_base + np.random.normal(0, 15)
            pressure = max(980, min(1050, pressure))
            
            # Precipitation (binary: 1 = raining, 0 = not raining)
            # Higher chance during certain hours and with high humidity
            rain_probability = 0.1
            if humidity > 80:
                rain_probability = 0.3
            if 14 <= hour <= 20:
                rain_probability *= 1.5
            
            precipitation = 1 if np.random.random() < rain_probability else 0
            
            # Rainfall amount (mm) if raining
            if precipitation == 1:
                rainfall = np.random.exponential(2.5)
                rainfall = min(50, rainfall)  # Cap at 50mm
            else:
                rainfall = 0.0
            
            data.append({
                'timestamp': timestamp,
                'location': node_name,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'wind_speed': round(wind_speed, 1),
                'pressure': round(pressure, 2),
                'precipitation': precipitation,
                'rainfall': round(rainfall, 2)
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{node_name}_weather.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✓ Generated weather data for {node_name}: {len(df)} records")
        print(f"  Avg Temp: {df['temperature'].mean():.1f}°C, Humidity: {df['humidity'].mean():.1f}%")
        
        return output_path


if __name__ == "__main__":
    generator = WeatherDataGenerator(seed=42)
    
    # Generate data for three nodes
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    
    for node in nodes:
        generator.generate_node_data(
            node_name=node,
            days=30
        )

