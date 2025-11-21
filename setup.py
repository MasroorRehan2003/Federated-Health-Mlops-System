from setuptools import setup, find_packages

setup(
    name="mlops-health-prediction",
    version="1.0.0",
    description="End-to-End MLOps System for Health Risk Prediction",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "flwr>=1.5.0",
        "mlflow>=2.7.0",
        "streamlit>=1.25.0",
        "fastapi>=0.100.0",
    ],
)

