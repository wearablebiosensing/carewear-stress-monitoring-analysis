from heart_rate_check import detect_data_gaps
import pandas as pd
import numpy as np

df_hr = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/2025-Test/CareWear_V2Test/T3/06-26-25/heart_rate_06_26_2025.csv")
gaps_df_strict = detect_data_gaps(df_hr, time_col='Timestamp', threshold_sec=1.0)
print(gaps_df_strict)