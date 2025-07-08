import pandas as pd

INPUT_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/merged_lables/"
OUT_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/merged_lables_resampled/"

def resample(df_hr,filename):
    # Convert Timestamp to datetime and create second-level floor
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp_floor'] = df['Timestamp'].dt.floor('S')

    # Generate complete 1-second timeline
    start_time = df['Timestamp_floor'].min()
    end_time = df['Timestamp_floor'].max()
    full_timeline = pd.date_range(start=start_time, end=end_time, freq='S')
    timeline_df = pd.DataFrame({'Timestamp_floor': full_timeline})

    # Merge with original data
    merged = pd.merge(timeline_df, df, on='Timestamp_floor', how='left')

    # Handle missing heart rates
    merged['HeartRate'] = merged['HeartRate'].fillna(-1)
    merged['Timestamp'] = merged['Timestamp'].fillna(merged['Timestamp_floor'])

    # Cleanup and final columns
    result = merged[['HeartRate', 'Timestamp', 'activity', 'Timestamp_pd', 'manual_labels_activity']]
    result = result.sort_values('Timestamp').reset_index(drop=True)

    # Save or use result
    result.to_csv(OUT_FOLDER +filename, index=False)
filename = 'heart_rate_26_merged_labels.csv'
# Load your CSV file (replace with actual file path)
df = pd.read_csv(INPUT_FOLDER + filename)
resample(df,filename)