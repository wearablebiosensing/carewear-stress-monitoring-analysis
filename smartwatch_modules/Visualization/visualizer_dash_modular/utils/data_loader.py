import os
import pandas as pd

def list_csv_files(folder, filter_str):
    return [f for f in os.listdir(folder) if f.endswith('.csv') and filter_str in f]

def extract_participant_id(file_name):
    import re
    match = re.search(r'_P(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')

def load_csv(folder, filename):
    return pd.read_csv(os.path.join(folder, filename))
