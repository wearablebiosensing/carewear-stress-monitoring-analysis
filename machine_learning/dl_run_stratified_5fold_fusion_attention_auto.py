"""
DeepFusionNet with Temporal Attention for Multimodal Stress Detection

SCIENTIFIC REFERENCES & ARCHITECTURAL JUSTIFICATION:

1. Base Architecture (DeepConvLSTM):
   Justifies using 1D-CNNs for spatial feature learning of raw IMU data, 
   followed by LSTMs for temporal dynamics.
   - Ordóñez, F. J., & Roggen, D. (2016). Deep Convolutional and LSTM Recurrent 
     Neural Networks for Multimodal Wearable Activity Recognition. Sensors, 16(1), 115.

2. Temporal Self-Attention Mechanism:
   Justifies adding an attention layer after the LSTM to dynamically weight 
   informative micro-tremors in the window, ignoring non-stressful stillness.
   - Zeng, M., et al. (2018). Understanding and Improving Recurrent Networks 
     for Human Activity Recognition by Continuous Attention. ISWC '18.

3. Multimodal Late Fusion (ACC + HR):
   Justifies building separate deep branches for physical (ACC) and physiological (HR)
   arousal before fusing them at the fully-connected layer.
   - Radu, V., et al. (2018). Multimodal Deep Learning for Activity and Context 
     Recognition. IMWUT.
   - Schmidt, P., et al. (2018). Introducing WESAD, a Multimodal Dataset for 
     Wearable Stress and Affect Detection. ICMI '18.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, medfilt
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, 
    recall_score, classification_report
)

def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Stress (0)', 'Stress (1)'], 
                yticklabels=['Non-Stress (0)', 'Stress (1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------------------------------------------------------
# 1. DEEP FUSION NETWORK ARCHITECTURE WITH ATTENTION
# ---------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    Self-attention layer for time-series.
    Dynamically weights the importance of different time steps in the LSTM output sequence.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, lstm_outputs):
        # lstm_outputs shape: (Batch, Seq_Len, Hidden_Size)
        attn_weights = self.attention(lstm_outputs)  # (Batch, Seq_Len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize weights over sequence length
        # Weighted sum of LSTM hidden states
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)  # (Batch, Hidden_Size)
        return context_vector, attn_weights

class DeepFusionNet(nn.Module):
    def __init__(self, num_classes=2, acc_seq_len=3000, hr_seq_len=60):
        super(DeepFusionNet, self).__init__()
        
        # --- ACCELEROMETER BRANCH ---
        self.acc_conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.acc_relu1 = nn.ReLU()
        self.acc_maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        self.acc_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.acc_relu2 = nn.ReLU()
        self.acc_maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        self.acc_conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.acc_relu3 = nn.ReLU()
        self.acc_maxpool3 = nn.MaxPool1d(kernel_size=2)
        
        self.acc_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        
        # Attention Mechanism for ACC micro-tremors
        self.acc_attention = TemporalAttention(hidden_size=128)
        
        # --- HEART RATE BRANCH ---
        # HR is 1 channel, 1 Hz (seq_len=60)
        self.hr_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.hr_relu1 = nn.ReLU()
        self.hr_maxpool1 = nn.MaxPool1d(kernel_size=2) # Pooling for 60-element sequence
        
        self.hr_lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        
        # --- FUSION LAYER ---
        # 128 (ACC LSTM) + 32 (HR LSTM) = 160
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 + 32, 64)
        self.fc_relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x_acc, x_hr):
        # x_acc shape: (Batch, 3, Seq_Len)
        # x_hr shape: (Batch, 1, Seq_Len)
        
        # Process ACC
        a = self.acc_conv1(x_acc)
        a = self.acc_relu1(a)
        a = self.acc_maxpool1(a)
        
        a = self.acc_conv2(a)
        a = self.acc_relu2(a)
        a = self.acc_maxpool2(a)
        
        a = self.acc_conv3(a)
        a = self.acc_relu3(a)
        a = self.acc_maxpool3(a)
        
        a = a.permute(0, 2, 1) # Reshape for LSTM
        out_acc, _ = self.acc_lstm(a)
        
        # Apply Temporal Attention instead of just taking the last time step
        last_acc, acc_attn_weights = self.acc_attention(out_acc) # (Batch, 128)
        
        # Process HR
        h = self.hr_conv1(x_hr)
        h = self.hr_relu1(h)
        h = self.hr_maxpool1(h)
        
        h = h.permute(0, 2, 1) # Reshape for LSTM
        out_hr, _ = self.hr_lstm(h)
        last_hr = out_hr[:, -1, :] # (Batch, 32)
        
        # Fuse
        fused = torch.cat((last_acc, last_hr), dim=1) # (Batch, 160)
        
        x = self.dropout(fused)
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.fc2(x)
        
        return x

class DeepHRNet(nn.Module):
    def __init__(self, num_classes=2, hr_seq_len=3000):
        super(DeepHRNet, self).__init__()
        # HR is 1 channel
        self.hr_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.hr_relu1 = nn.ReLU()
        self.hr_maxpool1 = nn.MaxPool1d(kernel_size=4) # Aggressive pooling for HR
        
        self.hr_lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32, 16)
        self.fc_relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x_hr):
        # x_hr shape: (Batch, 1, Seq_Len)
        h = self.hr_conv1(x_hr)
        h = self.hr_relu1(h)
        h = self.hr_maxpool1(h)
        
        h = h.permute(0, 2, 1) # Reshape for LSTM
        out_hr, _ = self.hr_lstm(h)
        last_hr = out_hr[:, -1, :] # (Batch, 32)
        
        x = self.dropout(last_hr)
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.fc2(x)
        
        return x

# ---------------------------------------------------------
# 2. DATASET DEFINITION & LOADING
# ---------------------------------------------------------
class FusionWearableDataset(Dataset):
    def __init__(self, X_acc, X_hr, y, groups):
        self.X_acc = torch.tensor(X_acc, dtype=torch.float32)
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_acc[idx], self.X_hr[idx], self.y[idx]

class HRWearableDataset(Dataset):
    def __init__(self, X_hr, y, groups):
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_hr[idx], self.y[idx]

def extract_metadata_from_filename(filename):
    pattern = r"activity_id_([\d\.-]+)_acc_(\d+)_merged_labels\.csv"
    match = re.search(pattern, filename)
    if match:
        act = float(match.group(1))
        part = int(match.group(2))
        return act, part
    return None, None

def apply_standard_acc_preprocessing(data_arr, fs=25):
    processed = np.zeros_like(data_arr)
    nyq = 0.5 * fs
    low = 0.3 / nyq
    high_freq = min(20.0, nyq - 0.5)
    high = high_freq / nyq
    b, a = butter(3, [low, high], btype='bandpass')
    
    for i in range(data_arr.shape[1]):
        med_filtered = medfilt(data_arr[:, i], kernel_size=3)
        processed[:, i] = filtfilt(b, a, med_filtered)
        
    return processed

def standardize_timestamp(df, prioritize_timestamp=False):
    # If prioritize_timestamp is True, it tries 'Timestamp' first (to avoid CareWear's corrupted Timestamp_pd)
    if prioritize_timestamp and 'Timestamp' in df.columns:
        return pd.to_datetime(df['Timestamp'], errors='coerce')
    
    if 'Timestamp_pd' in df.columns:
        return pd.to_datetime(df['Timestamp_pd'], errors='coerce')
    elif 'Timestamp' in df.columns:
        return pd.to_datetime(df['Timestamp'], errors='coerce')
    elif 'datetime' in df.columns:
        return pd.to_datetime(df['datetime'], errors='coerce')
    else:
        return None

def load_and_window_fusion_data(jobs, window_sec=60):
    print(f"\n[INFO] Scanning for ACC and HR chunks and building Fusion Dataset...")
    
    all_X_acc = []
    all_X_hr = []
    all_y = []
    all_groups = []
    
    for job in jobs:
        acc_dir = job['acc_dir']
        hr_dir = job['hr_dir']
        dataset_type = job['dataset_type']
        part_offset = job.get('part_offset', 0)
        
        print(f"\n  [INFO] Processing {dataset_type} dataset from {acc_dir}...")
        
        if dataset_type.lower() == "biopac":
            sampling_rate = 50
        else:
            sampling_rate = 50 # Standardize CareWear and GalaxyWatch to 50 Hz
            
        if dataset_type.lower() == "carewear":
            stress_mapping = {
                1: -1, 2: 1, 3: 1, 4: -1, 5: 1, 
                6: -1, 7: 0, 8: 0
            }
            print("  [INFO] Applied CareWear Stress Mapping (2,3,5 -> 1 | 7,8 -> 0)")
        else:
            # Mapping for GalaxyPPG dataset:
            # Stress (1): TSST (3,4), Sing a Song (7,8)
            # Non-Stress (0): Walking (16), Jogging (18), Running (20)
            stress_mapping = {
                1: -1, 2: -1, 3: 1, 4: 1, 5: -1, 
                6: -1, 7: 1, 8: 1, 9: -1, 10: -1,
                11: -1, 12: -1, 13: -1, 14: -1, 15: -1,
                16: 0, 17: -1, 18: 0, 19: -1, 20: 0
            }
            print("  [INFO] Applied GalaxyPPG Stress Mapping")
        
        window_size = int(window_sec * sampling_rate)
        overlap = int(window_size * 0.5)
        step_size = window_size - overlap
        
        acc_files = list(Path(acc_dir).rglob("activity_id_*_acc_*_merged_labels.csv"))
        print(f"    -> Found {len(acc_files)} potential ACC chunk files.")
        
        for fpath_acc in acc_files:
            act_id, part_id = extract_metadata_from_filename(fpath_acc.name)
            if act_id is None or part_id is None:
                continue
                
            target_label = stress_mapping.get(act_id, -1)
            if target_label == -1:
                continue
                
            # Check if activity_id is integer-like (1.0 vs 1) and participant ID is zero-padded (02 vs 2)
            hr_filename_1 = f"activity_id_{act_id}_hr_{part_id}_merged_labels.csv"
            hr_filename_2 = f"activity_id_{int(act_id)}_hr_{part_id}_merged_labels.csv"
            hr_filename_3 = f"activity_id_{act_id:.1f}_hr_{part_id}_merged_labels.csv"
            hr_filename_4 = f"activity_id_{act_id}_hr_{part_id:02d}_merged_labels.csv"
            hr_filename_5 = f"activity_id_{int(act_id)}_hr_{part_id:02d}_merged_labels.csv"
            hr_filename_6 = f"activity_id_{act_id:.1f}_hr_{part_id:02d}_merged_labels.csv"
            
            # Add GalaxyPPG HR chunk filename candidates (e.g., activity_id_10.0_P02_GalaxyWatch_HR_chunk.csv)
            hr_filename_7 = f"activity_id_{act_id}_P{part_id:02d}_GalaxyWatch_HR_chunk.csv"
            hr_filename_8 = f"activity_id_{int(act_id)}_P{part_id:02d}_GalaxyWatch_HR_chunk.csv"
            hr_filename_9 = f"activity_id_{act_id:.1f}_P{part_id:02d}_GalaxyWatch_HR_chunk.csv"
            hr_filename_10 = f"activity_id_{act_id}_P{part_id}_GalaxyWatch_HR_chunk.csv"
            hr_filename_11 = f"activity_id_{int(act_id)}_P{part_id}_GalaxyWatch_HR_chunk.csv"
            hr_filename_12 = f"activity_id_{act_id:.1f}_P{part_id}_GalaxyWatch_HR_chunk.csv"
            
            fpath_hr = None
            candidates = [
                hr_filename_1, hr_filename_2, hr_filename_3, hr_filename_4, hr_filename_5, hr_filename_6,
                hr_filename_7, hr_filename_8, hr_filename_9, hr_filename_10, hr_filename_11, hr_filename_12
            ]
            
            for cand in candidates:
                cand_path = Path(hr_dir) / cand
                if cand_path.exists():
                    fpath_hr = cand_path
                    break
                    
            if fpath_hr is None:
                print(f"  [WARNING] No matching HR file found for {fpath_acc.name}. Skipping.")
                continue
                
            try:
                # Load ACC
                df_acc = pd.read_csv(fpath_acc, low_memory=False)
                
                # Standardize column names for different datasets
                df_acc['Timestamp_pd'] = standardize_timestamp(df_acc)
                if df_acc['Timestamp_pd'] is None:
                    print(f"  [WARNING] Missing required timestamp column in ACC file {fpath_acc.name}. Skipping.")
                    continue
                    
                if not all(c in df_acc.columns for c in ['x', 'y', 'z']):
                    print(f"  [WARNING] Missing required columns in ACC file {fpath_acc.name}. Skipping.")
                    continue
                for col in ['x', 'y', 'z']:
                    df_acc[col] = pd.to_numeric(df_acc[col], errors='coerce')
                df_acc = df_acc.dropna(subset=['x', 'y', 'z', 'Timestamp_pd'])
                df_acc = df_acc.sort_values('Timestamp_pd')
                
                # Resample ACC strictly to 50 Hz
                if dataset_type.lower() != "biopac":
                    df_acc = df_acc[['Timestamp_pd', 'x', 'y', 'z']].set_index('Timestamp_pd')
                    df_acc = df_acc.resample("20ms").mean().interpolate(method='linear').reset_index()
                
                # Load HR
                df_hr = pd.read_csv(fpath_hr, low_memory=False)
                
                # Standardize column names for different datasets
                # Use prioritize_timestamp=True for CareWear HR
                is_carewear_hr = (dataset_type.lower() == "carewear")
                df_hr['Timestamp_pd'] = standardize_timestamp(df_hr, prioritize_timestamp=is_carewear_hr)
                
                if df_hr['Timestamp_pd'] is None:
                    print(f"  [WARNING] Missing required timestamp column in HR file {fpath_hr.name}. Skipping.")
                    continue
                if 'hr' in df_hr.columns and 'HeartRate' not in df_hr.columns:
                    df_hr.rename(columns={'hr': 'HeartRate'}, inplace=True)
                    
                if not all(c in df_hr.columns for c in ['HeartRate', 'Timestamp_pd']):
                    print(f"  [WARNING] Missing required columns in HR file {fpath_hr.name}. Skipping.")
                    continue
                df_hr['HeartRate'] = pd.to_numeric(df_hr['HeartRate'], errors='coerce')
                df_hr = df_hr.dropna(subset=['Timestamp_pd'])
                df_hr = df_hr.sort_values('Timestamp_pd')
                
                # Resample HR strictly to 1 Hz
                if dataset_type.lower() != "biopac":
                    df_hr = df_hr[['Timestamp_pd', 'HeartRate']].set_index('Timestamp_pd')
                    df_hr = df_hr.resample("1s").mean().interpolate(method='linear').reset_index()
                
                # Clean HR
                df_hr.loc[df_hr['HeartRate'] <= 0, 'HeartRate'] = np.nan
                df_hr.loc[df_hr['HeartRate'] > 300, 'HeartRate'] = np.nan
                # Interpolate missing values
                df_hr['HeartRate'] = df_hr['HeartRate'].ffill().bfill()
                
                # If completely empty or NaN, we can't use this chunk
                if df_hr['HeartRate'].isna().all():
                    continue
                    
                # Find Common Time Range (Overlap)
                start_time = max(df_acc['Timestamp_pd'].min(), df_hr['Timestamp_pd'].min())
                end_time = min(df_acc['Timestamp_pd'].max(), df_hr['Timestamp_pd'].max())
                
                df_acc = df_acc[(df_acc['Timestamp_pd'] >= start_time) & (df_acc['Timestamp_pd'] <= end_time)]
                df_hr = df_hr[(df_hr['Timestamp_pd'] >= start_time) & (df_hr['Timestamp_pd'] <= end_time)]
    
                acc_arr = df_acc[['x', 'y', 'z']].values
                hr_arr = df_hr[['HeartRate']].values
                
                hr_window_size = window_sec * 1 # HR is 1 Hz
                
                if len(acc_arr) < window_size or len(hr_arr) < hr_window_size:
                    continue
                
                # Preprocess ACC (Filtering is safe now because fs=50 is mathematically guaranteed by resampling)
                acc_arr = apply_standard_acc_preprocessing(acc_arr, fs=sampling_rate)
                
                # (Standardization is now done subject-wise after windowing)
                
                # Create Decoupled Sliding Windows
                acc_step_size = step_size
                hr_step_size = int(hr_window_size * 0.5)
                
                num_windows = (len(acc_arr) - window_size) // acc_step_size + 1
                
                for i in range(num_windows):
                    start_acc = i * acc_step_size
                    start_hr = i * hr_step_size
                    
                    window_acc = acc_arr[start_acc : start_acc + window_size]
                    window_hr = hr_arr[start_hr : start_hr + hr_window_size]
                    
                    if len(window_hr) < hr_window_size:
                        break
                    
                    # Transpose for PyTorch Conv1d
                    window_acc = window_acc.T
                    window_hr = window_hr.T
                    
                    all_X_acc.append(window_acc)
                    all_X_hr.append(window_hr)
                    all_y.append(target_label)
                    all_groups.append(part_id + part_offset)
                    
            except Exception as e:
                print(f"    [ERROR] Failed processing {fpath_acc.name}: {e}")
                
    X_acc_arr = np.array(all_X_acc)
    X_hr_arr = np.array(all_X_hr)
    y_arr = np.array(all_y)
    groups_arr = np.array(all_groups)
    
    print("\n[INFO] Applying Subject-Wise Normalization...")
    unique_groups = np.unique(groups_arr)
    for g in unique_groups:
        g_idx = (groups_arr == g)
        if not np.any(g_idx):
            continue
            
        # Standardize ACC for this participant
        acc_g = X_acc_arr[g_idx] # (N_g, Channels, Seq_Len)
        N_g, C, S = acc_g.shape
        acc_g_flat = acc_g.transpose(0, 2, 1).reshape(-1, C) # (N_g * Seq_Len, Channels)
        acc_scaler = StandardScaler()
        acc_g_scaled = acc_scaler.fit_transform(acc_g_flat)
        X_acc_arr[g_idx] = acc_g_scaled.reshape(N_g, S, C).transpose(0, 2, 1)
        
        # Standardize HR for this participant
        hr_g = X_hr_arr[g_idx] # (N_g, Channels, Seq_Len)
        N_g, C, S = hr_g.shape
        hr_g_flat = hr_g.transpose(0, 2, 1).reshape(-1, C)
        hr_scaler = StandardScaler()
        hr_g_scaled = hr_scaler.fit_transform(hr_g_flat)
        X_hr_arr[g_idx] = hr_g_scaled.reshape(N_g, S, C).transpose(0, 2, 1)
    
    print(f"\n[INFO] Dataset built successfully!")
    print(f"  -> Total Windows: {len(X_acc_arr)}")
    if len(X_acc_arr) > 0:
        print(f"  -> ACC Window Shape: {X_acc_arr.shape[1:]}")
        print(f"  -> HR Window Shape: {X_hr_arr.shape[1:]}")
    print(f"  -> Class Distribution: {Counter(y_arr)}")
    print(f"  -> Unique Participants: {len(np.unique(groups_arr))}")
    
    return X_acc_arr, X_hr_arr, y_arr, groups_arr

def extract_metadata_from_hr_filename(filename):
    patterns = [
        r"activity_id_([\d\.-]+)_biopac_(\d+)_merged_labels\.csv",
        r"activity_id_([\d\.-]+)_P(\d+)_GalaxyWatch_HR_chunk\.csv",
        r"activity_id_([\d\.-]+)_hr_(\d+)_merged_labels\.csv",
        r"activity_id_([\d\.-]+)_hr_(\d+)\.csv"
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            act = float(match.group(1))
            part = int(match.group(2))
            return act, part
    return None, None

def load_and_window_hr_only_data(hr_dir, dataset_type="Biopac", window_sec=60):
    print(f"\n[INFO] Scanning for HR chunks and building HR-Only Dataset for {dataset_type}...")
    
    # Determine native HR sampling rate
    if dataset_type.lower() == "biopac":
        sampling_rate = 50
    else:
        sampling_rate = 1 # GalaxyWatch HR is 1 Hz

    
    if dataset_type.lower() in ["carewear", "biopac"]:
        stress_mapping = {
            1: -1, 2: 1, 3: 1, 4: -1, 5: 1, 
            6: -1, 7: 0, 8: 0
        }
        print("[INFO] Applied CareWear/Biopac Stress Mapping (2,3,5 -> 1 | 7,8 -> 0)")
    else:
        # Mapping for GalaxyPPG dataset
        stress_mapping = {
            1: -1, 2: -1, 3: 0, 4: 0, 5: -1, 
            6: -1, 7: 0, 8: 0, 9: -1, 10: -1,
            11: -1, 12: -1, 13: -1, 14: -1, 15: -1,
            16: 1, 17: -1, 18: 1, 19: -1, 20: 1
        }
        print("[INFO] Applied GalaxyPPG Stress Mapping")
    
    window_size = int(window_sec * sampling_rate)
    overlap = int(window_size * 0.5)
    step_size = window_size - overlap
    
    all_X_hr = []
    all_y = []
    all_groups = []
    
    # Generic glob pattern to catch all possible HR files
    glob_pattern = "activity_id_*.csv"
        
    hr_files = list(Path(hr_dir).rglob(glob_pattern))
    print(f"  -> Found {len(hr_files)} potential HR chunk files.")
    
    for fpath_hr in hr_files:
        act_id, part_id = extract_metadata_from_hr_filename(fpath_hr.name)
        if act_id is None or part_id is None:
            continue
            
        target_label = stress_mapping.get(act_id, -1)
        if target_label == -1:
            continue
            
        try:
            df_hr = pd.read_csv(fpath_hr, low_memory=False)
            
            # Standardize HR column
            if 'hr' in df_hr.columns and 'Heart Rate' not in df_hr.columns:
                df_hr.rename(columns={'hr': 'Heart Rate'}, inplace=True)
            if 'HeartRate' in df_hr.columns and 'Heart Rate' not in df_hr.columns:
                df_hr.rename(columns={'HeartRate': 'Heart Rate'}, inplace=True)
                
            if 'Heart Rate' not in df_hr.columns:
                print(f"  [WARNING] Missing HR column in {fpath_hr.name}. Skipping.")
                continue
                
            df_hr['Heart Rate'] = pd.to_numeric(df_hr['Heart Rate'], errors='coerce')
            
            df_hr.loc[df_hr['Heart Rate'] <= 0, 'Heart Rate'] = np.nan
            df_hr.loc[df_hr['Heart Rate'] > 300, 'Heart Rate'] = np.nan
            df_hr['Heart Rate'] = df_hr['Heart Rate'].ffill().bfill()
            
            # Dynamic Resampling for HR-Only
            if dataset_type.lower() != "biopac":
                if 'datetime' in df_hr.columns and 'Timestamp_pd' not in df_hr.columns:
                    df_hr.rename(columns={'datetime': 'Timestamp_pd'}, inplace=True)
                if 'Timestamp_pd' in df_hr.columns:
                    df_hr['Timestamp_pd'] = pd.to_datetime(df_hr['Timestamp_pd'], errors='coerce')
                    df_hr = df_hr.dropna(subset=['Timestamp_pd'])
                    df_hr = df_hr.sort_values('Timestamp_pd')
                    df_hr = df_hr[['Timestamp_pd', 'Heart Rate']]
                    df_hr = df_hr.set_index('Timestamp_pd')
                    freq_str = f"{int(1000/sampling_rate)}ms"
                    df_hr = df_hr.resample(freq_str).mean().interpolate(method='linear')
                    df_hr = df_hr.reset_index()
            
            if df_hr['Heart Rate'].isna().all():
                continue
                
            hr_arr = df_hr[['Heart Rate']].values
            
            if len(hr_arr) < window_size:
                continue
                
            hr_scaler = StandardScaler()
            hr_arr = hr_scaler.fit_transform(hr_arr)
            
            for start_idx in range(0, len(hr_arr) - window_size + 1, step_size):
                window_hr = hr_arr[start_idx : start_idx + window_size]
                window_hr = window_hr.T
                
                all_X_hr.append(window_hr)
                all_y.append(target_label)
                all_groups.append(part_id)
                
        except Exception as e:
            print(f"  [ERROR] Failed processing {fpath_hr.name}: {e}")
            
    X_hr_arr = np.array(all_X_hr)
    y_arr = np.array(all_y)
    groups_arr = np.array(all_groups)
    
    print("\n[INFO] HR-Only Dataset built successfully!")
    print(f"  -> Total Windows: {len(X_hr_arr)}")
    if len(X_hr_arr) > 0:
        print(f"  -> HR Window Shape: {X_hr_arr.shape[1:]}")
    print(f"  -> Class Distribution: {Counter(y_arr)}")
    print(f"  -> Unique Participants: {len(np.unique(groups_arr))}")
    
    return X_hr_arr, y_arr, groups_arr

# ---------------------------------------------------------
# 3. TRAINING & EVALUATION FUNCTIONS
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, device, class_weights, epochs=30, lr=0.001, fold_out_dir=None, fold_idx=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_bacc = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = 7
    
    epoch_mean_grads = []
    epoch_max_grads = []
    final_val_probs = []
    final_val_labels = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        ep_max_g = []
        ep_mean_g = []
        
        for batch in train_loader:
            if len(batch) == 3:
                batch_acc, batch_hr, batch_y = batch
                batch_acc = batch_acc.to(device)
                batch_hr = batch_hr.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_acc, batch_hr)
            else:
                batch_hr, batch_y = batch
                batch_hr = batch_hr.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_hr)
                
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient Monitoring
            batch_max_grad = 0.0
            batch_sum_grad = 0.0
            batch_param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_max = p.grad.abs().max().item()
                    batch_max_grad = max(batch_max_grad, param_max)
                    batch_sum_grad += p.grad.abs().sum().item()
                    batch_param_count += p.grad.numel()
                    
            if batch_param_count > 0:
                ep_mean_g.append(batch_sum_grad / batch_param_count)
            ep_max_g.append(batch_max_grad)
            
            optimizer.step()
            
            train_loss += loss.item()
            
        if ep_mean_g:
            epoch_mean_grads.append(np.mean(ep_mean_g))
            epoch_max_grads.append(np.mean(ep_max_g))
            
        # Validation
        model.eval()
        val_y_true = []
        val_y_pred = []
        epoch_val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_acc, batch_hr, batch_y = batch
                    batch_acc = batch_acc.to(device)
                    batch_hr = batch_hr.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_acc, batch_hr)
                else:
                    batch_hr, batch_y = batch
                    batch_hr = batch_hr.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_hr)
                    
                probs = torch.softmax(outputs, dim=1)[:, 1]
                epoch_val_probs.extend(probs.cpu().numpy())
                _, preds = torch.max(outputs, 1)
                
                val_y_true.extend(batch_y.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())
                
        val_bacc = balanced_accuracy_score(val_y_true, val_y_pred)
        scheduler.step(val_bacc)
        
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_model_state = model.state_dict()
            final_val_probs = epoch_val_probs
            final_val_labels = val_y_true
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"      [Early Stopping] Epoch {epoch+1}")
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    if fold_out_dir is not None:
        fold_out_dir = Path(fold_out_dir)
        fold_str = f"Fold {fold_idx}" if fold_idx is not None else "Deployment"
        
        # 1. Plot Gradient Norms
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_mean_grads, label='Mean Absolute Gradient')
        plt.plot(epoch_max_grads, label='Max Absolute Gradient')
        plt.title(f'{fold_str} - Gradient Flow')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Magnitude')
        plt.yscale('log') # Gradients are best viewed in log scale
        plt.legend()
        plt.grid(True)
        plt.savefig(fold_out_dir / "gradient_flow.png")
        plt.close()
        
        # 2. Plot Class Probabilities
        if final_val_probs and final_val_labels:
            plt.figure(figsize=(10, 5))
            df_probs = pd.DataFrame({'Probability': final_val_probs, 'Label': final_val_labels})
            sns.histplot(data=df_probs, x='Probability', hue='Label', bins=20, kde=True, 
                         palette={0: 'blue', 1: 'red'})
            plt.title(f'{fold_str} - Validation Stress Probabilities')
            plt.xlabel('Predicted Probability of Stress (Class 1)')
            plt.ylabel('Count')
            plt.savefig(fold_out_dir / "class_probabilities.png")
            plt.close()
            
    return model

def main():
    root = tk.Tk()
    root.withdraw()
    
    window_sec_str = simpledialog.askstring("Window Size", "Enter window size in seconds:", initialvalue="60")
    window_sec = int(window_sec_str) if window_sec_str else 60
    
    mode = simpledialog.askstring("Mode", "Enter Mode (Fusion or HR-Only):", initialvalue="Fusion")
    mode = mode if mode else "Fusion"
    
    dataset_type = simpledialog.askstring("Dataset Type", "Enter Dataset Type (CareWear, GalaxyPPG, Biopac, or Combined):", initialvalue="CareWear")
    dataset_type = dataset_type if dataset_type else "CareWear"
    
    if mode.lower() == "hr-only":
        print(f"[INFO] Please select the directory containing {dataset_type} HR chunks...")
        hr_dir_str = filedialog.askdirectory(title=f"Select Folder containing {dataset_type} HR Chunks")
        if not hr_dir_str:
            print("[CANCELLED] No HR directory selected.")
            sys.exit(0)
        hr_dir = Path(hr_dir_str).resolve()
        acc_dir = hr_dir # Fallback for results dir
        
        X_hr, y, groups = load_and_window_hr_only_data(
            hr_dir=hr_dir, 
            dataset_type=dataset_type, 
            window_sec=window_sec
        )
        X_acc = None
        if len(y) == 0:
            print("[ERROR] No data extracted. Exiting.")
            return
    else:
        jobs = []
        if dataset_type.lower() == "combined":
            print("[INFO] Combined Mode selected. Using hardcoded directories for CareWear and GalaxyPPG.")
            
            # CareWear
            cw_acc_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/5_activity_chunks/GalaxyWatch/acc_chunks"
            cw_hr_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/5_activity_chunks/GalaxyWatch/hr_chunks/hr"
            
            jobs.append({
                "acc_dir": Path(cw_acc_str).resolve(),
                "hr_dir": Path(cw_hr_str).resolve(),
                "dataset_type": "CareWear",
                "part_offset": 0
            })
            
            # GalaxyPPG
            gal_acc_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/5_activity_chunks/GalaxyWatch/acc_chunks"
            gal_hr_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/5_activity_chunks/GalaxyWatch/hr_chunks"
            
            jobs.append({
                "acc_dir": Path(gal_acc_str).resolve(),
                "hr_dir": Path(gal_hr_str).resolve(),
                "dataset_type": "GalaxyPPG",
                "part_offset": 1000 # Offset to prevent train/test leakage
            })
            
            acc_dir = Path(cw_acc_str).resolve() # For saving results
            
        elif dataset_type.lower() == "carewear":
            print("[INFO] CareWear Mode selected. Using hardcoded directories.")
            acc_dir_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/5_activity_chunks/GalaxyWatch/acc_chunks"
            hr_dir_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/5_activity_chunks/GalaxyWatch/hr_chunks/hr"
            acc_dir = Path(acc_dir_str).resolve()
            jobs.append({
                "acc_dir": acc_dir,
                "hr_dir": Path(hr_dir_str).resolve(),
                "dataset_type": "CareWear",
                "part_offset": 0
            })
            
        elif dataset_type.lower() == "galaxyppg":
            print("[INFO] GalaxyPPG Mode selected. Using hardcoded directories.")
            acc_dir_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/5_activity_chunks/GalaxyWatch/acc_chunks"
            hr_dir_str = "/media/wbl-hpc/ss/Project_CareWear/DATASET/ss_drive/GalaxyPPG/5_activity_chunks/GalaxyWatch/hr_chunks"
            acc_dir = Path(acc_dir_str).resolve()
            jobs.append({
                "acc_dir": acc_dir,
                "hr_dir": Path(hr_dir_str).resolve(),
                "dataset_type": "GalaxyPPG",
                "part_offset": 0
            })
            
        else:
            print(f"[INFO] Please select the directory containing {dataset_type} ACC chunks...")
            acc_dir_str = filedialog.askdirectory(title=f"Select Folder containing {dataset_type} ACC Chunks")
            if not acc_dir_str: sys.exit(0)
            
            print(f"[INFO] Please select the directory containing {dataset_type} HR chunks...")
            hr_dir_str = filedialog.askdirectory(title=f"Select Folder containing {dataset_type} HR Chunks")
            if not hr_dir_str: sys.exit(0)
            
            acc_dir = Path(acc_dir_str).resolve()
            jobs.append({
                "acc_dir": acc_dir,
                "hr_dir": Path(hr_dir_str).resolve(),
                "dataset_type": dataset_type,
                "part_offset": 0
            })
            
        # Build Dataset
        X_acc, X_hr, y, groups = load_and_window_fusion_data(jobs=jobs, window_sec=window_sec)
        
        if len(X_acc) == 0:
            print("[ERROR] No data extracted. Exiting.")
            return
            
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using Device: {device}")
    
    # Class Weights
    counts = Counter(y)
    total = len(y)
    class_weights = torch.tensor([total / counts[0], total / counts[1]], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    
    n_splits = 5
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_true_global = []
    y_pred_global = []
    fold_details = []
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_str = "DeepHRNet" if mode.lower() == "hr-only" else "DeepFusionNet"
    modality_str = "HROnly" if mode.lower() == "hr-only" else "Fusion(ACC+HR)"
    prefix = f"{model_name_str}_{dataset_type}_{modality_str}_Results_"
    results_dir = acc_dir / f"{prefix}{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    if mode.lower() == "hr-only":
        print(f"[INFO] Starting DeepHRNet Stratified {n_splits}-Fold Evaluation")
        X_to_split = X_hr
    else:
        print(f"[INFO] Starting DeepFusionNet Stratified {n_splits}-Fold Evaluation")
        X_to_split = X_acc
    print("="*60)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_to_split, y, groups=groups)):
        test_participants = np.unique(groups[test_idx])
        print(f"\n  --- Fold {fold_idx + 1}/{n_splits} ---")
        print(f"    -> Held-out Test Participants: {test_participants}")
        
        from sklearn.model_selection import GroupShuffleSplit
        
        # Create a true validation set (20% of training groups) for early stopping
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_split_idx, val_split_idx = next(gss.split(X_to_split[train_idx], y[train_idx], groups=groups[train_idx]))
        
        # Map back to global indices
        val_idx_global = train_idx[val_split_idx]
        train_idx_global = train_idx[train_split_idx]
        
        y_train, y_val, y_test = y[train_idx_global], y[val_idx_global], y[test_idx]
        
        if len(np.unique(y_test)) < 2:
            print("  [WARNING] Test set missing classes. Skipping fold.")
            continue
            
        X_train_hr, X_val_hr, X_test_hr = X_hr[train_idx_global], X_hr[val_idx_global], X_hr[test_idx]
        
        if mode.lower() == "hr-only":
            train_dataset = HRWearableDataset(X_train_hr, y_train, groups[train_idx_global])
            val_dataset = HRWearableDataset(X_val_hr, y_val, groups[val_idx_global])
            test_dataset = HRWearableDataset(X_test_hr, y_test, groups[test_idx])
            model = DeepHRNet(num_classes=2).to(device)
        else:
            X_train_acc, X_val_acc, X_test_acc = X_acc[train_idx_global], X_acc[val_idx_global], X_acc[test_idx]
            train_dataset = FusionWearableDataset(X_train_acc, X_train_hr, y_train, groups[train_idx_global])
            val_dataset = FusionWearableDataset(X_val_acc, X_val_hr, y_val, groups[val_idx_global])
            test_dataset = FusionWearableDataset(X_test_acc, X_test_hr, y_test, groups[test_idx])
            model = DeepFusionNet(num_classes=2, acc_seq_len=3000, hr_seq_len=60).to(device)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        fold_out_dir = results_dir / "fold_details" / f"fold_{fold_idx + 1}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        # Pass val_loader instead of test_loader to prevent leakage
        model = train_model(model, train_loader, val_loader, device, class_weights, epochs=30, lr=0.001, fold_out_dir=fold_out_dir, fold_idx=fold_idx+1)
        train_time = time.time() - start_time
        
        model.eval()
        fold_y_true = []
        fold_y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    batch_acc, batch_hr, batch_y = batch
                    batch_acc = batch_acc.to(device)
                    batch_hr = batch_hr.to(device)
                    outputs = model(batch_acc, batch_hr)
                else:
                    batch_hr, batch_y = batch
                    batch_hr = batch_hr.to(device)
                    outputs = model(batch_hr)
                    
                _, preds = torch.max(outputs, 1)
                fold_y_true.extend(batch_y.cpu().numpy())
                fold_y_pred.extend(preds.cpu().numpy())
                
        fold_bacc = balanced_accuracy_score(fold_y_true, fold_y_pred)
        fold_f1 = f1_score(fold_y_true, fold_y_pred, zero_division=0)
        fold_acc = accuracy_score(fold_y_true, fold_y_pred)
        
        cm = confusion_matrix(fold_y_true, fold_y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        print(f"    -> B-Acc: {fold_bacc:.4f} | F1: {fold_f1:.4f} | Time: {train_time:.1f}s")
        
        y_true_global.extend(fold_y_true)
        y_pred_global.extend(fold_y_pred)
        
        save_confusion_matrix(fold_y_true, fold_y_pred, fold_out_dir / f"cm_fold_{fold_idx + 1}.png", f"Fold {fold_idx + 1} CM")
        
        fold_details.append({
            "Fold": fold_idx + 1,
            "Accuracy": fold_acc,
            "Balanced Acc": fold_bacc,
            "F1 Score": fold_f1,
            "Sensitivity": sens,
            "Specificity": spec,
            "Train Time (s)": train_time
        })
        
    print("\n" + "="*60)
    bacc_g = balanced_accuracy_score(y_true_global, y_pred_global)
    f1_g = f1_score(y_true_global, y_pred_global, zero_division=0)
    acc_g = accuracy_score(y_true_global, y_pred_global)
    
    cm_global = confusion_matrix(y_true_global, y_pred_global, labels=[0, 1])
    tn_g, fp_g, fn_g, tp_g = cm_global.ravel()
    spec_g = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0.0
    sens_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
    
    model_name = "DeepHRNet" if mode.lower() == "hr-only" else "DeepFusionNet"
    
    print(f"[FINAL RESULT] {model_name} (Stratified 5-Fold Validation)")
    print(f"B-Acc:       {bacc_g:.4f}")
    print(f"F1 Score:    {f1_g:.4f}")
    print(f"Accuracy:    {acc_g:.4f}")
    print(f"Sensitivity: {sens_g:.4f}")
    print(f"Specificity: {spec_g:.4f}")
    print("="*60)
    
    report = classification_report(y_true_global, y_pred_global, labels=[0, 1], target_names=["Non-Stress", "Stress"], zero_division=0)
    with open(results_dir / f"{model_name}_report.txt", "w") as f:
        f.write(f"{model_name} Evaluation - Subject Independent (StratifiedGroupKFold)\n")
        f.write("="*60 + "\n")
        f.write(report)
        
    save_confusion_matrix(y_true_global, y_pred_global, results_dir / f"{model_name}_global_cm.png", f"{model_name} Global CM")
        
    df_summary = pd.DataFrame(fold_details)
    mean_std_row = {"Fold": "Mean ± Std"}
    for col in [c for c in df_summary.columns if c != "Fold"]:
        mean_std_row[col] = f"{df_summary[col].mean():.4f} ± {df_summary[col].std():.4f}"
    df_summary = pd.concat([df_summary, pd.DataFrame([mean_std_row])], ignore_index=True)
    df_summary.to_csv(results_dir / f"{model_name}_fold_summary.csv", index=False)
    
    print(f"    [DEPLOYMENT] Training final deployment model on all data...")
    if mode.lower() == "hr-only":
        final_dataset = HRWearableDataset(X_hr, y, groups)
        final_model = DeepHRNet(num_classes=2).to(device)
    else:
        final_dataset = FusionWearableDataset(X_acc, X_hr, y, groups)
        final_model = DeepFusionNet(num_classes=2).to(device)
        
    final_loader = DataLoader(final_dataset, batch_size=64, shuffle=True)
    deploy_out_dir = results_dir / "deployment_metrics"
    deploy_out_dir.mkdir(parents=True, exist_ok=True)
    final_model = train_model(final_model, final_loader, final_loader, device, class_weights, epochs=30, lr=0.001, fold_out_dir=deploy_out_dir, fold_idx="Deploy")
    deploy_model_path = results_dir / f"{model_name}_deploy.pth"
    torch.save(final_model.state_dict(), deploy_model_path)
    print(f"    [DEPLOYMENT] Model weights saved to {deploy_model_path}")
    
    print(f"\n[INFO] Results saved to {results_dir}")
    messagebox.showinfo("Complete", f"{model_name} Training and Evaluation Finished.")

if __name__ == "__main__":
    main()
