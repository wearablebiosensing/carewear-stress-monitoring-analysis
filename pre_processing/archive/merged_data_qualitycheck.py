import os
import pandas as pd
import glob

# Path to your folder
folder_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/merged_lables/"

# List all files matching pattern
files = glob.glob(os.path.join(folder_path, "heart_rate*_merged_labels.csv"))

results = []

for file_path in files:
    # Extract filename
    filename = os.path.basename(file_path)
    
    # Extract PID as integer
    pid_str = filename.split("_")[2]  # '12' in heart_rate_12_merged_labels.csv
    pid = int(pid_str)
    
    # Get file size in MB (2 decimal places)
    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Row count
    number_rows = len(df)
    
    # Unique labels count from 'manual_labels_activity'
    labels_num = df['manual_labels_activity'].nunique()
    
    # Append to results
    results.append({
        "filename": filename,
        "pid": pid,
        "file_size_MB": file_size_mb,
        "number_rows": number_rows,
        "labels_num": labels_num
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by pid
results_df = results_df.sort_values(by="pid").reset_index(drop=True)

results_df.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/merged_labeles_quality/"+"watch_labels_quality_test.csv", index=False)
