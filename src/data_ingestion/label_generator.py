"""
Label Generator
Creates health risk labels based on rules combining multiple data sources.
"""

import pandas as pd
import numpy as np
import os


class LabelGenerator:
    """Generate health risk labels from merged data"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    # -----------------------------------------------------
    #                RISK SCORE GENERATION
    # -----------------------------------------------------
    def calculate_risk_score(self, df: pd.DataFrame, method: str = "binary") -> pd.Series:
        """
        Calculate health risk score using environmental + wearable features.

        method = "continuous" → returns float between 0–1  
        method = "binary" → uses 70th percentile to mark top 30% as high-risk (1)
        """

        required = ["aqi", "heart_rate_mean", "spo2_mean", "pm25", "temperature", "humidity"]
        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"Missing required column '{col}' in merged dataset. "
                    f"These must exist: {required}"
                )

        # -----------------------------------------------------
        # AQI risk
        # -----------------------------------------------------
        aqi_risk = np.where(
            df["aqi"] > 150, 0.8,
            np.where(df["aqi"] > 100, 0.5,
            np.where(df["aqi"] > 50, 0.2, 0.0))
        )

        # -----------------------------------------------------
        # Heart rate risk
        # -----------------------------------------------------
        hr_norm = (df["heart_rate_mean"] - 60) / 50
        hr_norm = np.clip(hr_norm, 0, 1)

        hr_risk = np.where(
            df["heart_rate_mean"] > 110, 0.9,
            hr_norm * 0.6
        )

        # -----------------------------------------------------
        # SpO2 risk
        # -----------------------------------------------------
        spo2_risk = np.where(
            df["spo2_mean"] < 90, 1.0,
            np.where(df["spo2_mean"] < 94, 0.7,
            np.where(df["spo2_mean"] < 96, 0.35, 0.0))
        )

        # -----------------------------------------------------
        # PM2.5 risk
        # -----------------------------------------------------
        pm25_risk = np.where(
            df["pm25"] > 55, 0.7,
            np.where(df["pm25"] > 35, 0.4,
            np.where(df["pm25"] > 12, 0.15, 0.0))
        )

        # -----------------------------------------------------
        # Weather risk
        # -----------------------------------------------------
        temp_risk = np.where(
            (df["temperature"] > 35) | (df["temperature"] < 5), 0.3,
            np.where((df["temperature"] > 30) | (df["temperature"] < 10), 0.15, 0.0)
        )

        humidity_risk = np.where(
            df["humidity"] > 85, 0.2,
            np.where(df["humidity"] < 30, 0.1, 0.0)
        )

        weather_risk = np.clip(temp_risk + humidity_risk, 0, 1)

        # -----------------------------------------------------
        # Combined effect risk
        # -----------------------------------------------------
        combined_risk = (
            (df["aqi"] > 100).astype(int) * 0.3 +
            (df["heart_rate_mean"] > 90).astype(int) * 0.2 +
            (df["spo2_mean"] < 95).astype(int) * 0.3
        )
        combined_risk = np.clip(combined_risk, 0, 1)

        # -----------------------------------------------------
        # FINAL CONTINUOUS RISK SCORE
        # -----------------------------------------------------
        weights = [0.25, 0.15, 0.20, 0.15, 0.10, 0.15]  
        components = [aqi_risk, hr_risk, spo2_risk, pm25_risk, weather_risk, combined_risk]

        risk_score = sum(w * comp for w, comp in zip(weights, components))
        risk_score = np.clip(risk_score, 0, 1)

        if method == "continuous":
            return pd.Series(risk_score, name="health_risk")

        # -----------------------------------------------------
        # BINARY LABELING (TOP 30% = 1)
        # -----------------------------------------------------
        threshold = np.quantile(risk_score, 0.70)

        if np.allclose(risk_score, risk_score[0]):
            labels = np.zeros(len(risk_score), dtype=int)
            n_pos = max(1, int(0.3 * len(labels)))
            labels[:n_pos] = 1
            np.random.shuffle(labels)
            return pd.Series(labels, name="health_risk")

        labels = (risk_score >= threshold).astype(int)
        return pd.Series(labels, name="health_risk")

    # -----------------------------------------------------
    #                ADD LABELS + SAVE FILE
    # -----------------------------------------------------
    def add_labels_to_data(self, input_path: str, output_path: str = None, method: str = "binary") -> str:
        """Load merged CSV, add health_risk + risk_category, and save to new file."""
        
        df = pd.read_csv(input_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        print(f"\n=== Generating labels for {input_path} ===")
        print(f"Records: {len(df)} | Method: {method}")

        df["health_risk"] = self.calculate_risk_score(df, method=method)

        if method == "continuous":
            df["risk_category"] = pd.cut(
                df["health_risk"],
                bins=[0, 0.3, 0.6, 1],
                labels=["Low", "Medium", "High"]
            )
        else:
            df["risk_category"] = df["health_risk"].map({0: "Low", 1: "High"})

        # auto filename selection
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            if method == "binary":
                output_path = f"{base}_labeled.csv"
            else:
                output_path = f"{base}_labeled_continuous.csv"

        df.to_csv(output_path, index=False)

        if method == "binary":
            print("Label distribution:", dict(df["health_risk"].value_counts()))

        print(f"✓ Saved labeled data → {output_path}")
        return output_path


# ---------------------------------------------------------
#                   RUN LABEL GENERATION
# ---------------------------------------------------------
if __name__ == "__main__":
    labeler = LabelGenerator()

    nodes = ["hospital_A", "hospital_B", "hospital_C"]
    data_dir = "data/processed"

    for node in nodes:
        input_file = os.path.join(data_dir, f"{node}_merged.csv")
        if os.path.exists(input_file):
            labeler.add_labels_to_data(input_file, method="binary")
            labeler.add_labels_to_data(input_file, method="continuous")

    combined_file = os.path.join(data_dir, "all_nodes_merged.csv")
    if os.path.exists(combined_file):
        labeler.add_labels_to_data(combined_file, method="binary")
        labeler.add_labels_to_data(combined_file, method="continuous")
