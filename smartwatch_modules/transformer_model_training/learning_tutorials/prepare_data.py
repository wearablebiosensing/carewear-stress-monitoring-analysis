import pandas as pd

# Load your CSV
df = pd.read_csv("heart_rate_1_merged_labels.csv")

# Drop unnecessary columns (keep only what you need)
df = df[["HeartRate", "Timestamp", "activity"]]

# Drop rows where 'activity' is missing (if any)
df = df.dropna(subset=["activity"])
