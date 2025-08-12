import pandas as pd
import glob
import os
from collections import defaultdict

# Input/Output folders
directory = '/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File'
OUT_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/merged_lables/"

# Get biopac CSV files
filepaths = glob.glob(os.path.join(directory, '*biopac*.csv'))
print("filepaths:", filepaths)

# Group files by participant
participants_files = defaultdict(list)
for f in filepaths:
    basename = os.path.basename(f)
    participant = basename.split('_')[0]  # e.g. 'P2'
    participants_files[participant].append(f)

# Process each participant
for participant, files in participants_files.items():

    # Check if output file already exists
    output_file = os.path.join(OUT_FOLDER, f'{participant}_biopac.csv')
    if os.path.exists(output_file):
        print(f"Skipping {participant} — file already exists at {output_file}")
        continue

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['filename'] = os.path.basename(f)
        dfs.append(df)

    # Concatenate
    participant_df = pd.concat(dfs, ignore_index=True)

    # Basic validation
    labels_len = len(participant_df["filename"].unique())
    if labels_len >= 7:
        print("participant id", participant, "labels len", labels_len, "==== PASSED ✅")
    else:
        print("participant id", participant, "labels len", labels_len, "==== FAIL")

    # Save to new file
    participant_df.to_csv(output_file, index=False)
    print(f'{participant}: {participant_df.shape[0]} rows concatenated → saved to {output_file}')
