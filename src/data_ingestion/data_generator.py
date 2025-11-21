"""
Perfect High-Quality Dataset Generator
Generates wearable, air-quality, and weather data
WITH REALISTIC ANOMALIES AND BALANCED HEALTH RISKS
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class PerfectDatasetGenerator:
    def __init__(self, users=100, days=30, seed=42):
        self.users = users
        self.days = days
        self.seed = seed
        np.random.seed(seed)

        self.locations = {
            "hospital_A": {"temp": 20, "hum": 65, "aqi": 90},
            "hospital_B": {"temp": 16, "hum": 70, "aqi": 70},
            "hospital_C": {"temp": 26, "hum": 60, "aqi": 110},
        }

    # ---------------------------------------------------------
    # 1. WEARABLE DATA (heart rate, SPO2, steps, sleep)
    # ---------------------------------------------------------
    def generate_wearable(self, node, outfolder="data/raw"):
        print(f"✓ Generating wearable for {node}...")

        start = datetime.now() - timedelta(days=self.days)
        timestamps = pd.date_range(start=start, periods=self.days * 24, freq="H")

        data = []
        for user in range(self.users):
            hr_base = np.random.randint(60, 85)
            spo2_base = np.random.randint(95, 99)

            for t in timestamps:

                # NORMAL BEHAVIOUR
                heart_rate = np.random.normal(hr_base, 6)
                spo2 = np.random.normal(spo2_base, 1)

                # --------------------------------------------------
                # 🔥 HEALTH ANOMALIES (create high-risk events)
                # --------------------------------------------------
                if np.random.rand() < 0.05:  # 5% anomaly hours
                    heart_rate += np.random.randint(20, 65)   # HR spike
                    spo2 -= np.random.randint(3, 10)          # sudden drop

                steps = np.random.randint(0, 250)

                data.append({
                    "timestamp": t,
                    "user_id": user,
                    "location": node,
                    "heart_rate": round(heart_rate, 1),
                    "spo2": round(spo2, 1),
                    "steps": steps,
                    "sleep_status": 1 if 0 <= t.hour <= 5 else 0,
                })

        df = pd.DataFrame(data)

        df["cumulative_steps"] = df.groupby("user_id")["steps"].cumsum()

        os.makedirs(outfolder, exist_ok=True)
        df.to_csv(f"{outfolder}/{node}.csv", index=False)

        print(f"✓ Wearable saved: {len(df)} rows")
        return df

    # ---------------------------------------------------------
    # 2. AIR QUALITY (AQI, PM2.5, PM10)
    # ---------------------------------------------------------
    def generate_air_quality(self, node, outfolder="data/raw"):
        print(f"✓ Generating air quality for {node}...")

        base = self.locations[node]
        start = datetime.now() - timedelta(days=self.days)
        timestamps = pd.date_range(start=start, periods=self.days * 24, freq="H")

        data = []

        for t in timestamps:
            # NORMAL AQI
            aqi = np.random.normal(base["aqi"], 15)
            pm25 = np.random.normal(aqi * 0.3, 4)
            pm10 = np.random.normal(aqi * 0.45, 6)

            # --------------------------------------------------
            # 🔥 'SMOG EVENTS' — increases risk
            # --------------------------------------------------
            if np.random.rand() < 0.06:
                aqi += np.random.randint(50, 110)
                pm25 += np.random.randint(20, 60)
                pm10 += np.random.randint(30, 70)

            data.append({
                "timestamp": t,
                "location": node,
                "aqi": round(aqi, 1),
                "pm25": round(pm25, 1),
                "pm10": round(pm10, 1),
            })

        df = pd.DataFrame(data)
        os.makedirs(outfolder, exist_ok=True)
        df.to_csv(f"{outfolder}/{node}_air_quality.csv", index=False)
        print(f"✓ Air quality saved: {len(df)} rows")
        return df

    # ---------------------------------------------------------
    # 3. WEATHER (temperature, humidity, wind)
    # ---------------------------------------------------------
    def generate_weather(self, node, outfolder="data/raw"):
        print(f"✓ Generating weather for {node}...")

        base = self.locations[node]
        start = datetime.now() - timedelta(days=self.days)
        timestamps = pd.date_range(start=start, periods=self.days * 24, freq="H")

        data = []

        for t in timestamps:
            temp = np.random.normal(base["temp"], 5)
            hum = np.random.normal(base["hum"], 7)
            wind = np.random.normal(10, 3)

            # --------------------------------------------------
            # 🔥 EXTREME WEATHER EVENTS (risk factors)
            # --------------------------------------------------
            if np.random.rand() < 0.05:
                temp += np.random.randint(10, 18)   # heatwave
                hum += np.random.randint(-30, 20)  # humidity shock

            data.append({
                "timestamp": t,
                "location": node,
                "temperature": round(temp, 1),
                "humidity": round(hum, 1),
                "wind_speed": round(wind, 1),
            })

        df = pd.DataFrame(data)
        os.makedirs(outfolder, exist_ok=True)
        df.to_csv(f"{outfolder}/{node}_weather.csv", index=False)
        print(f"✓ Weather saved: {len(df)} rows")
        return df

    # ---------------------------------------------------------
    # 4. RUN ALL FOR ALL NODES
    # ---------------------------------------------------------
    def run_all(self, nodes=None):
        if nodes is None:
            nodes = ["hospital_A", "hospital_B", "hospital_C"]

        print("\n=====================================================")
        print(" GENERATING PERFECT BALANCED DATASET ")
        print("=====================================================")

        for node in nodes:
            self.generate_wearable(node)
            self.generate_air_quality(node)
            self.generate_weather(node)

        print("\n✓ ALL DATA SOURCES GENERATED SUCCESSFULLY\n")


if __name__ == "__main__":
    gen = PerfectDatasetGenerator(users=120, days=30)
    gen.run_all()
