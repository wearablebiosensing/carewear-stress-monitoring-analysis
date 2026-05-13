import pandas as pd
from datetime import datetime

# Load the labels file
# labels_file_path = "labels.txt"  # Replace with your actual path
# labels_df = pd.read_csv(labels_file_path)
labels_df = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/labels.txt")

# Function to validate timestamps
def validate_labels(labels_df):
    errors = []
    
    # Iterate through each row in the labels DataFrame
    for index, row in labels_df.iterrows():
        pid = row['PID']
        activity = row['activity']
        start_time = row['start']
        stop_time = row['stop']
        notes = row['notes']

        try:
            # Convert start and stop times to datetime objects
            start_dt = datetime.strptime(start_time, "%I:%M %p") if pd.notnull(start_time) else None
            stop_dt = datetime.strptime(stop_time, "%I:%M %p") if pd.notnull(stop_time) else None
            
            # Check for errors
            if start_dt is None or stop_dt is None:
                errors.append((index, pid, activity, "Missing start or stop time"))
            elif start_dt > stop_dt:
                errors.append((index, pid, activity, "Start time is later than stop time"))
        
        except ValueError:
            # Invalid timestamp format
            errors.append((index, pid, activity, "Invalid timestamp format"))

    return errors

# Validate labels and print errors
errors = validate_labels(labels_df)
if errors:
    print("Rows with errors:")
    for error in errors:
        print(f"Row {error[0]} | PID: {error[1]} | Activity: {error[2]} | Error: {error[3]}")
else:
    print("No errors found in labels file.")
