import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(csv_path):
    """
    Load Algerian Forest Fires dataset from CSV.
    Converts Classes to 0/1 and drops 'Region'.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Drop 'Region'
    if "Region" in df.columns:
        df = df.drop(columns=["Region"])

    # Convert Classes to binary
    df["Classes"] = df["Classes"].str.strip().str.lower()
    df["Classes"] = df["Classes"].map({"not fire": 0, "fire": 1})

    X = df.drop(columns=["Classes"]).values
    y = df["Classes"].values
    feature_names = df.drop(columns=["Classes"]).columns.tolist()
    target_names = ["not fire", "fire"]

    return X, y, feature_names, target_names

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
