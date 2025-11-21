"""
Main Data Ingestion Pipeline
Orchestrates the entire data generation and processing pipeline
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_ingestion.wearable_generator import WearableDataGenerator
from data_ingestion.air_quality_generator import AirQualityGenerator
from data_ingestion.weather_generator import WeatherDataGenerator
from data_ingestion.data_merger import DataMerger
from data_ingestion.label_generator import LabelGenerator


def run_data_pipeline(
    nodes: list = None,
    num_users_per_node: int = 100,
    days: int = 30,
    start_date: datetime = None,
    generate_labels: bool = True
):
    """
    Run the complete data ingestion pipeline
    
    Args:
        nodes: List of node names (default: ['hospital_A', 'hospital_B', 'hospital_C'])
        num_users_per_node: Number of users per node
        days: Number of days of data to generate
        start_date: Start date (defaults to days ago from now)
        generate_labels: Whether to generate health risk labels
    """
    if nodes is None:
        nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    print("=" * 80)
    print("PHASE 1: DATA PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Nodes: {nodes}")
    print(f"  Users per node: {num_users_per_node}")
    print(f"  Days: {days}")
    print(f"  Start date: {start_date.date()}")
    print(f"  Generate labels: {generate_labels}")
    print("\n" + "-" * 80)
    
    # Step 1: Generate Wearable Data
    print("\n[1/5] Generating Wearable Sensor Data...")
    print("-" * 80)
    wearable_gen = WearableDataGenerator(seed=42)
    for node in nodes:
        wearable_gen.generate_node_data(
            node_name=node,
            num_users=num_users_per_node,
            start_date=start_date,
            days=days
        )
    
    # Step 2: Generate Air Quality Data
    print("\n[2/5] Generating Air Quality Sensor Data...")
    print("-" * 80)
    aq_gen = AirQualityGenerator(seed=42)
    for node in nodes:
        aq_gen.generate_node_data(
            node_name=node,
            start_date=start_date,
            days=days
        )
    
    # Step 3: Generate Weather Data
    print("\n[3/5] Generating Weather Data...")
    print("-" * 80)
    weather_gen = WeatherDataGenerator(seed=42)
    for node in nodes:
        weather_gen.generate_node_data(
            node_name=node,
            start_date=start_date,
            days=days
        )
    
    # Step 4: Merge Data Streams
    print("\n[4/5] Merging Data Streams...")
    print("-" * 80)
    merger = DataMerger()
    for node in nodes:
        merger.merge_data_streams(node)
    
    # Create combined dataset
    merger.merge_all_nodes(nodes)
    
    # Step 5: Generate Labels
    if generate_labels:
        print("\n[5/5] Generating Health Risk Labels...")
        print("-" * 80)
        labeler = LabelGenerator()
        
        # Generate binary labels
        for node in nodes:
            input_path = os.path.join('data/processed', f"{node}_merged.csv")
            if os.path.exists(input_path):
                labeler.add_labels_to_data(input_path, method='binary')
        
        # Generate labels for combined dataset
        combined_path = os.path.join('data/processed', "all_nodes_merged.csv")
        if os.path.exists(combined_path):
            labeler.add_labels_to_data(combined_path, method='binary')
            labeler.add_labels_to_data(combined_path, method='continuous')
    
    print("\n" + "=" * 80)
    print("✓ DATA PIPELINE COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  Raw data: data/raw/")
    print("    - {node}.csv (wearable data)")
    print("    - {node}_air_quality.csv")
    print("    - {node}_weather.csv")
    print("  Processed data: data/processed/")
    print("    - {node}_merged.csv")
    print("    - {node}_merged_labeled.csv")
    print("    - all_nodes_merged.csv")
    print("    - all_nodes_merged_labeled.csv")


if __name__ == "__main__":
    # Run the complete pipeline
    run_data_pipeline(
        nodes=['hospital_A', 'hospital_B', 'hospital_C'],
        num_users_per_node=100,
        days=30,
        generate_labels=True
    )

