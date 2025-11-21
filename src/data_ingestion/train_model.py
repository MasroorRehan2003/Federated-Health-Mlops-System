import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

DATA_PATH = "data/processed/all_nodes_merged_labeled.csv"
MODEL_PATH = "data/models/final_model.pkl"

os.makedirs("data/models", exist_ok=True)

def train():
    df = pd.read_csv(DATA_PATH)

    y = df["health_risk"]
    X = df.drop(columns=["health_risk", "risk_category", "timestamp"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.4f}")

    pickle.dump(model, open(MODEL_PATH, "wb"))
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
