"""
Test script for Phase 1: Data Pipeline
Run this to verify the data generation pipeline works correctly
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_ingestion.main import run_data_pipeline


def test_data_pipeline():
    """Test the complete data pipeline"""
    print("Testing Data Pipeline...")
    print("=" * 80)
    
    # Run pipeline with smaller dataset for testing
    run_data_pipeline(
        nodes=['hospital_A', 'hospital_B', 'hospital_C'],
        num_users_per_node=10,  # Smaller for testing
        days=7,  # One week for testing
        generate_labels=True
    )
    
    # Verify files were created
    print("\n" + "=" * 80)
    print("Verifying Generated Files...")
    print("=" * 80)
    
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    all_files_exist = True
    
    # Check raw data files
    print("\nRaw Data Files:")
    for node in nodes:
        files = [
            f'data/raw/{node}.csv',
            f'data/raw/{node}_air_quality.csv',
            f'data/raw/{node}_weather.csv'
        ]
        for file in files:
            exists = os.path.exists(file)
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
            if not exists:
                all_files_exist = False
    
    # Check processed data files
    print("\nProcessed Data Files:")
    for node in nodes:
        files = [
            f'data/processed/{node}_merged.csv',
            f'data/processed/{node}_merged_labeled.csv'
        ]
        for file in files:
            exists = os.path.exists(file)
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
            if not exists:
                all_files_exist = False
    
    # Check combined files
    combined_files = [
        'data/processed/all_nodes_merged.csv',
        'data/processed/all_nodes_merged_labeled.csv'
    ]
    print("\nCombined Data Files:")
    for file in combined_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_files_exist = False
    
    if all_files_exist:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - Data pipeline is working correctly!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ SOME FILES ARE MISSING - Please check the errors above")
        print("=" * 80)
    
    return all_files_exist


if __name__ == "__main__":
    success = test_data_pipeline()
    sys.exit(0 if success else 1)

