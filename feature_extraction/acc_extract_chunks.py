import pandas as pd
import os
import glob
import pandas as pd
import numpy as np
input_folder = "/Volumes/CW_2024/merged_lables"
output_folder = "/Volumes/CW_2024/acc_chunks/"

for file_path in glob.glob(os.path.join(input_folder, "*heart_rate*.csv")):
    print("Processing:======", file_path.split("/")[-1])
    df = pd.read_csv(file_path)
    print("df.columns ===== ",df.columns)
    print("unique activitys === ",df["activity_int"].unique())
    for activity in df["activity_int"].unique():
        if activity != -1:
            print("activity id ====",activity)
            df[df["activity_int"]==activity].to_csv(output_folder+"activity_id_"+ str(activity) +"_"+ file_path.split("/")[-1])
        
