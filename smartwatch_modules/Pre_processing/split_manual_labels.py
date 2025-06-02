import os
import pandas as pd

# Path to your labels.csv file
LABELS_CSV = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Timeline/labels.csv'  # Change as needed

# Output folder
OUTPUT_FOLDER = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Manual'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Activity mapping
activity_map = {
    'R1': 'rest1',
    'R3': 'rest3',
    "SBH":"stationary_Bike2",
    'SBL': 'stationary_Bike1',
    'PS': 'prepare speech',
    'R2': 'rest2',
    'GS': 'give speech',
    'MM': 'mental math'
}

# Helper: map activity label
def map_activity(label):
    for k, v in activity_map.items():
        if k in label:
            return v
    return label  # fallback: keep original if no match

# Read the CSV
df = pd.read_csv(LABELS_CSV)

# Check for required columns
required_cols = {'PID', 'activity', 'start', 'stop', 'notes'}
print(df.columns)
if not required_cols.issubset(df.columns):
    raise ValueError(f"Input file must have columns: {required_cols}")

# Map activities
df['activity'] = df['activity'].apply(map_activity)
print("df = \n",df)
# Group by participant and write separate files
for pid, group in df.groupby('PID'):
    # Ensure correct header order
    out_df = group[['activity', 'start', 'stop', 'notes']].copy()
    out_df = out_df.rename(columns={
    'activity': 'manual_labels_activity',
    'start': 'start_time',
    'stop': 'end_time',
    'notes': 'notes'
})

    filename = f"manual_task_timeline_{str(pid).zfill(2)}.csv"
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
