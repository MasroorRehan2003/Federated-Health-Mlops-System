"""
Data Merger
Merges wearable, air quality, and weather data streams
"""

import pandas as pd
import numpy as np
import os
from typing import List
from datetime import datetime


class DataMerger:
    """Merge multiple data streams into unified datasets"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir

    # ---------------------------------------------------------
    # LOAD ALL DATA STREAMS FOR A NODE
    # ---------------------------------------------------------
    def load_node_data(self, node_name: str) -> tuple:

        # Wearable ---------------------------------------------------------
        wearable_path = os.path.join(self.data_dir, f"{node_name}.csv")
        if not os.path.exists(wearable_path):
            raise FileNotFoundError(f"Wearable data missing: {wearable_path}")

        wearable_df = pd.read_csv(wearable_path)
        wearable_df["timestamp"] = pd.to_datetime(wearable_df["timestamp"])

        # Air Quality -------------------------------------------------------
        air_path = os.path.join(self.data_dir, f"{node_name}_air_quality.csv")
        if not os.path.exists(air_path):
            raise FileNotFoundError(f"Air quality data missing: {air_path}")

        air_df = pd.read_csv(air_path)
        air_df["timestamp"] = pd.to_datetime(air_df["timestamp"])

        # Weather -----------------------------------------------------------
        weather_path = os.path.join(self.data_dir, f"{node_name}_weather.csv")
        if not os.path.exists(weather_path):
            raise FileNotFoundError(f"Weather data missing: {weather_path}")

        weather_df = pd.read_csv(weather_path)
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

        return wearable_df, air_df, weather_df

    # ---------------------------------------------------------
    # MERGE THE 3 STREAMS INTO A UNIFIED HOURLY DATASET
    # ---------------------------------------------------------
    def merge_data_streams(self, node_name: str, output_dir: str = "data/processed") -> str:
        print(f"\nMerging data streams for {node_name}...")

        wearable_df, air_df, weather_df = self.load_node_data(node_name)

        # ---------------------------------------------------------
        # 1. Aggregate wearable data to hourly-level
        # ---------------------------------------------------------
        wearable_hourly = wearable_df.groupby("timestamp").agg({
            "heart_rate": ["mean", "std", "max"],
            "steps": "sum",
            "cumulative_steps": "max",
            "spo2": ["mean", "min"],
            "sleep_status": "sum",
            "sleep_quality": "mean",
            "user_id": "count"
        }).reset_index()

        wearable_hourly.columns = [
            "timestamp",
            "heart_rate_mean", "heart_rate_std", "heart_rate_max",
            "steps_total",
            "cumulative_steps_max",
            "spo2_mean", "spo2_min",
            "sleeping_users_count",
            "sleep_quality_mean",
            "active_users_count"
        ]

        # ---------------------------------------------------------
        # 2. Merge wearable + air quality
        # ---------------------------------------------------------
        merged_df = pd.merge(
            wearable_hourly,
            air_df,
            on="timestamp",
            how="inner"
        )

        # ---------------------------------------------------------
        # 3. Merge with weather data
        # ---------------------------------------------------------
        merged_df = pd.merge(
            merged_df,
            weather_df,
            on="timestamp",
            how="inner"
        )

        # ---------------------------------------------------------
        # 4. Force clean single location column
        # ---------------------------------------------------------
        merged_df["location"] = node_name
        merged_df = merged_df.drop(
            columns=[c for c in merged_df.columns if "location_" in c],
            errors="ignore"
        )

        # ---------------------------------------------------------
        # 5. Sort by timestamp
        # ---------------------------------------------------------
        merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

        # ---------------------------------------------------------
        # 6. Add derived engineered features
        # ---------------------------------------------------------
        merged_df = self._add_derived_features(merged_df)

        # ---------------------------------------------------------
        # 7. Save merged dataset
        # ---------------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{node_name}_merged.csv")
        merged_df.to_csv(out_path, index=False)

        print(f"✓ Merged {node_name} → {len(merged_df)} records")
        print(f"  From {merged_df['timestamp'].min()} → {merged_df['timestamp'].max()}")
        print(f"  Features: {merged_df.shape[1]}")

        return out_path

    # ---------------------------------------------------------
    # ADD FEATURE ENGINEERING
    # ---------------------------------------------------------
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # AQI category
        df["aqi_category"] = pd.cut(
            df["aqi"],
            bins=[0, 50, 100, 150, 200, 300, 500],
            labels=["Good", "Moderate", "Sensitive", "Unhealthy", "Very_Unhealthy", "Hazardous"]
        )

        # Temperature category
        df["temp_category"] = pd.cut(
            df["temperature"],
            bins=[-999, 10, 20, 30, 999],
            labels=["Cold", "Mild", "Warm", "Hot"]
        )

        # Preliminary health indicators
        df["high_heart_rate"] = (df["heart_rate_mean"] > 90).astype(int)
        df["low_spo2"] = (df["spo2_mean"] < 95).astype(int)
        df["high_pollution"] = (df["aqi"] > 100).astype(int)

        return df

    # ---------------------------------------------------------
    # MERGE ALL NODES INTO ONE GLOBAL FILE
    # ---------------------------------------------------------
    def merge_all_nodes(self, node_names: List[str], output_dir: str = "data/processed") -> str:
        all_data = []

        for node in node_names:
            merged_path = os.path.join(output_dir, f"{node}_merged.csv")

            if not os.path.exists(merged_path):
                self.merge_data_streams(node, output_dir)

            df = pd.read_csv(merged_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        out_path = os.path.join(output_dir, "all_nodes_merged.csv")
        combined_df.to_csv(out_path, index=False)

        print(f"\n✓ Combined all nodes into one dataset")
        print(f"  Total records: {len(combined_df)}")
        print(f"  Nodes: {node_names}")

        return out_path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    merger = DataMerger()

    nodes = ["hospital_A", "hospital_B", "hospital_C"]

    for node in nodes:
        merger.merge_data_streams(node)

    merger.merge_all_nodes(nodes)
