"""
Wearable Sensor Data Generator
Generates realistic wearable health device data (heart rate, steps, sleep, SpO2)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict


class WearableDataGenerator:
    """Generate simulated wearable health device data"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_user_data(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
        base_heart_rate: float = 70.0,
        base_steps: int = 8000,
        base_spo2: float = 98.0
    ) -> pd.DataFrame:
        """
        Generate hourly wearable data for a single user
        
        Args:
            user_id: Unique identifier for the user
            start_date: Start datetime
            end_date: End datetime
            base_heart_rate: Baseline heart rate (bpm)
            base_steps: Baseline daily steps
            base_spo2: Baseline SpO2 percentage
        
        Returns:
            DataFrame with hourly wearable metrics
        """
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        n_hours = len(timestamps)
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            
            # Heart Rate: varies by time of day (lower at night, higher during day)
            if 2 <= hour <= 6:  # Night (sleeping)
                hr_mean = base_heart_rate - 15
                hr_std = 5
            elif 6 < hour <= 10:  # Morning (waking up)
                hr_mean = base_heart_rate + 10
                hr_std = 8
            elif 10 < hour <= 18:  # Day (active)
                hr_mean = base_heart_rate + 15
                hr_std = 12
            else:  # Evening
                hr_mean = base_heart_rate
                hr_std = 8
            
            heart_rate = max(50, min(180, np.random.normal(hr_mean, hr_std)))
            
            # Steps: accumulated throughout the day, reset at midnight
            # Steps accumulate during waking hours
            if hour == 0:
                steps_per_hour = 0  # Midnight, reset
            elif 6 <= hour <= 22:
                steps_per_hour = np.random.poisson(base_steps / 16)  # Distribute over 16 waking hours
            else:
                steps_per_hour = np.random.poisson(10)  # Minimal steps at night
            
            # SpO2: generally stable, slight variations
            spo2 = max(90, min(100, np.random.normal(base_spo2, 1.5)))
            
            # Sleep: binary indicator (1 = sleeping, 0 = awake)
            # Assume sleep between 11 PM and 7 AM
            if hour >= 23 or hour < 7:
                sleep_status = 1
            else:
                sleep_status = 0
            
            # Sleep quality score (0-100) - only meaningful during sleep hours
            if sleep_status == 1:
                sleep_quality = np.random.normal(75, 15)
                sleep_quality = max(0, min(100, sleep_quality))
            else:
                sleep_quality = 0
            
            data.append({
                'timestamp': timestamp,
                'user_id': user_id,
                'heart_rate': round(heart_rate, 1),
                'steps': int(steps_per_hour),
                'spo2': round(spo2, 1),
                'sleep_status': sleep_status,
                'sleep_quality': round(sleep_quality, 1)
            })
        
        df = pd.DataFrame(data)
        
        # Calculate cumulative steps per day
        df['date'] = df['timestamp'].dt.date
        df['cumulative_steps'] = df.groupby('date')['steps'].cumsum()
        
        return df[['timestamp', 'user_id', 'heart_rate', 'steps', 'cumulative_steps', 
                   'spo2', 'sleep_status', 'sleep_quality']]
    
    def generate_node_data(
        self,
        node_name: str,
        num_users: int = 100,
        start_date: datetime = None,
        days: int = 30,
        output_dir: str = 'data/raw'
    ) -> str:
        """
        Generate wearable data for all users in a node (hospital/city)
        
        Args:
            node_name: Name of the node (e.g., 'hospital_A')
            num_users: Number of users in this node
            start_date: Start date (defaults to 30 days ago)
            days: Number of days to generate
            output_dir: Output directory for CSV files
        
        Returns:
            Path to saved CSV file
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        end_date = start_date + timedelta(days=days)
        
        # Generate data for all users
        all_data = []
        
        for user_idx in range(num_users):
            user_id = f"{node_name}_user_{user_idx+1:03d}"
            
            # Vary baseline parameters per user
            base_hr = np.random.normal(72, 8)
            base_steps = int(np.random.normal(8500, 2000))
            base_spo2 = np.random.normal(98, 1)
            
            user_df = self.generate_user_data(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                base_heart_rate=base_hr,
                base_steps=base_steps,
                base_spo2=base_spo2
            )
            
            all_data.append(user_df)
        
        # Combine all users
        node_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{node_name}.csv")
        node_df.to_csv(output_path, index=False)
        
        print(f"✓ Generated wearable data for {node_name}: {len(node_df)} records")
        print(f"  Users: {num_users}, Date range: {start_date.date()} to {end_date.date()}")
        
        return output_path


if __name__ == "__main__":
    generator = WearableDataGenerator(seed=42)
    
    # Generate data for three hospitals
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    
    for node in nodes:
        generator.generate_node_data(
            node_name=node,
            num_users=100,
            days=30
        )

