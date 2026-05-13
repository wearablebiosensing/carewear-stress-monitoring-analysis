import os
import pandas as pd

folder = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Belt'  # Change to your folder path
new_columns = ['Unnamed: 0','Belt_Activity_Labels', 'start_time', 'end_time']  # Edit as needed

for filename in os.listdir(folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder, filename)
        df = pd.read_csv(file_path)
        # print(df.columns)
        df.columns = new_columns  # Rename all columns at once
        df.to_csv(file_path, index=False)
        print(f"Renamed columns in {filename}")
