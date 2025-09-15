import os
import glob
import pandas as pd

# ----------------- Folder Paths -----------------
input_folder = "/Volumes/CW_2024/merged_lables"
output_folder = "/Volumes/CW_2024/hr_chunks"
os.makedirs(output_folder, exist_ok=True)

# ----------------- Activity Mapping -----------------
activity_mapping = {
    'rest1': 1,
    'prepare speech': 2,
    'give speech': 3,
    'rest2': 4,
    'mental math': 5,
    'rest3': 6,
    'stationary_Bike1': 7,
    'stationary_Bike2': 8
}
files_processed = []
# ----------------- Processing -----------------
for file_path in glob.glob(os.path.join(input_folder, "*heart_rate*.csv")):
    file_name = os.path.basename(file_path)
    print(f"\nProcessing file: {file_name}")
    files_processed.append(file_name)
    try:
        df = pd.read_csv(file_path)

        if 'manual_labels_activity' not in df.columns:
            print(f"⚠️ File {file_name} missing 'manual_labels_activity', skipping.")
            continue

        # Map manual_labels_activity to activity_int
        df['activity_int'] = df['manual_labels_activity'].map(activity_mapping).fillna(-1).astype(int)

        unique_activities = df['activity_int'].unique()
        print(f"Unique activities found: {unique_activities}")

        # Write subset files for each activity (skip -1)
        for activity in unique_activities:
            if activity == -1:
                continue

            subset_df = df[df['activity_int'] == activity].copy()
            output_file = os.path.join(output_folder, f"activity_id_{activity}_{file_name}")
            subset_df.to_csv(output_file, index=False)
            print(f"✅ Saved activity {activity} subset to {output_file}")

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
print("files_processed===",files_processed)

# file_list = ['heart_rate_1_merged_labels.csv', 'heart_rate_10_merged_labels.csv', 'heart_rate_11_merged_labels.csv', 'heart_rate_12_merged_labels.csv', 'heart_rate_13_merged_labels.csv', 'heart_rate_14_merged_labels.csv', 'heart_rate_15_merged_labels.csv', 'heart_rate_16_merged_labels.csv', 'heart_rate_17_merged_labels.csv', 'heart_rate_18_merged_labels.csv', 'heart_rate_2_merged_labels.csv', 'heart_rate_20_merged_labels.csv', 'heart_rate_21_merged_labels.csv', 'heart_rate_22_merged_labels.csv', 'heart_rate_23_merged_labels.csv', 'heart_rate_24_merged_labels.csv', 'heart_rate_25_merged_labels.csv', 'heart_rate_26_merged_labels.csv', 'heart_rate_27_merged_labels.csv', 'heart_rate_3_merged_labels.csv', 'heart_rate_4_merged_labels.csv', 'heart_rate_5_merged_labels.csv', 'heart_rate_6_merged_labels.csv', 'heart_rate_7_merged_labels.csv', 'heart_rate_8_merged_labels.csv', 'heart_rate_9_merged_labels.csv']