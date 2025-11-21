"""
Simplified script to run federated learning
This script helps coordinate server and clients
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_data_files():
    """Check if required data files exist"""
    nodes = ['hospital_A', 'hospital_B', 'hospital_C']
    missing = []
    
    for node in nodes:
        path = f"data/processed/{node}_merged_labeled.csv"
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("✗ Missing data files:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease run Phase 1 first:")
        print("  python src/data_ingestion/main.py")
        return False
    
    print("✓ All data files found")
    return True


def main():
    print("=" * 80)
    print("FEDERATED LEARNING - QUICK START")
    print("=" * 80)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("SETUP INSTRUCTIONS")
    print("=" * 80)
    print("\nTo run federated learning, you need multiple terminals:")
    print("\n1. TERMINAL 1 - Start the server:")
    print("   python src/federated_learning/fl_server.py --rounds 10")
    print("\n2. TERMINAL 2 - Start client 1:")
    print("   python src/federated_learning/fl_client.py --node-name hospital_A")
    print("\n3. TERMINAL 3 - Start client 2:")
    print("   python src/federated_learning/fl_client.py --node-name hospital_B")
    print("\n4. TERMINAL 4 - Start client 3:")
    print("   python src/federated_learning/fl_client.py --node-name hospital_C")
    print("\n" + "=" * 80)
    print("\n⚠️  IMPORTANT:")
    print("   - Start the SERVER first, then start the CLIENTS")
    print("   - Wait for server to be ready before starting clients")
    print("   - All clients must use the same model type as the server")
    print("=" * 80)
    
    # Ask if user wants to start server
    response = input("\nDo you want to start the server now? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nStarting server...")
        print("(Press Ctrl+C to stop)")
        print("=" * 80)
        try:
            subprocess.run([
                sys.executable,
                "src/federated_learning/fl_server.py",
                "--rounds", "10"
            ])
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
    else:
        print("\nYou can start the server manually with:")
        print("  python src/federated_learning/fl_server.py --rounds 10")


if __name__ == "__main__":
    main()

