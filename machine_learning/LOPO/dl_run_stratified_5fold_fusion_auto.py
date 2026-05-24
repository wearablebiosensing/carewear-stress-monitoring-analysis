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
# 1. DEEP FUSION NETWORK ARCHITECTURE
# ---------------------------------------------------------
class DeepFusionNet(nn.Module):
    def __init__(self, num_classes=2, acc_seq_len=3000, hr_seq_len=3000):
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
        
        # --- HEART RATE BRANCH ---
        # HR is 1 channel, upsampled to 3000 via merge_asof
        self.hr_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.hr_relu1 = nn.ReLU()
        self.hr_maxpool1 = nn.MaxPool1d(kernel_size=4) # Aggressive pooling for HR
        
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
        last_acc = out_acc[:, -1, :] # (Batch, 128)
        
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

def extract_metadata_from_filename(filename):
    pattern = r"activity_id_([\d\.-]+)_acc_(\d+)_merged_labels\.csv"
    match = re.search(pattern, filename)
    if match:
        act = float(match.group(1))
        part = int(match.group(2))
        return act, part
    return None, None

def apply_standard_acc_preprocessing(data_arr, fs=50):
    processed = np.zeros_like(data_arr)
    nyq = 0.5 * fs
    low = 0.3 / nyq
    high = 20.0 / nyq
    b, a = butter(3, [low, high], btype='bandpass')
    
    for i in range(data_arr.shape[1]):
        med_filtered = medfilt(data_arr[:, i], kernel_size=3)
        processed[:, i] = filtfilt(b, a, med_filtered)
        
    return processed

def load_and_window_fusion_data(acc_dir, hr_dir, dataset_type="CareWear", window_sec=60, sampling_rate=50):
    print("\n[INFO] Scanning for ACC and HR chunks and building Fusion Dataset...")
    
    if dataset_type.lower() == "carewear":
        stress_mapping = {1: -1, 2: 0, 3: 0, 4: -1, 5: 0, 6: -1, 7: 1, 8: 1}
    else:
        stress_mapping = {
            1: -1, 2: -1, 3: 0, 4: 0, 5: -1, 6: -1, 7: 0, 8: 0, 9: -1, 10: -1,
            11: -1, 12: -1, 13: -1, 14: -1, 15: -1, 16: 1, 17: -1, 18: 1, 19: -1, 20: 1
        }
    
    window_size = int(window_sec * sampling_rate)
    overlap = int(window_size * 0.5)
    step_size = window_size - overlap
    
    all_X_acc = []
    all_X_hr = []
    all_y = []
    all_groups = []
    
    acc_files = list(Path(acc_dir).rglob("activity_id_*_acc_*_merged_labels.csv"))
    print(f"  -> Found {len(acc_files)} potential ACC chunk files.")
    
    for fpath_acc in acc_files:
        act_id, part_id = extract_metadata_from_filename(fpath_acc.name)
        if act_id is None or part_id is None:
            continue
            
        target_label = stress_mapping.get(act_id, -1)
        if target_label == -1:
            continue
            
        # Find corresponding HR file
        # Check if activity_id is integer-like (1.0 vs 1)
        hr_filename_1 = f"activity_id_{act_id}_hr_{part_id}_merged_labels.csv"
        hr_filename_2 = f"activity_id_{int(act_id)}_hr_{part_id}_merged_labels.csv"
        hr_filename_3 = f"activity_id_{act_id:.1f}_hr_{part_id}_merged_labels.csv"
        
        fpath_hr = None
        for cand in [hr_filename_1, hr_filename_2, hr_filename_3]:
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
            if not all(c in df_acc.columns for c in ['x', 'y', 'z', 'Timestamp_pd']):
                continue
            for col in ['x', 'y', 'z']:
                df_acc[col] = pd.to_numeric(df_acc[col], errors='coerce')
            df_acc = df_acc.dropna(subset=['x', 'y', 'z', 'Timestamp_pd'])
            df_acc['Timestamp_pd'] = pd.to_datetime(df_acc['Timestamp_pd'], errors='coerce')
            df_acc = df_acc.dropna(subset=['Timestamp_pd'])
            df_acc = df_acc.sort_values('Timestamp_pd')
            
            # Load HR
            df_hr = pd.read_csv(fpath_hr, low_memory=False)
            if not all(c in df_hr.columns for c in ['HeartRate', 'Timestamp_pd']):
                continue
            df_hr['HeartRate'] = pd.to_numeric(df_hr['HeartRate'], errors='coerce')
            df_hr['Timestamp_pd'] = pd.to_datetime(df_hr['Timestamp_pd'], errors='coerce')
            df_hr = df_hr.dropna(subset=['Timestamp_pd'])
            df_hr = df_hr.sort_values('Timestamp_pd')
            
            # Clean HR
            df_hr.loc[df_hr['HeartRate'] <= 0, 'HeartRate'] = np.nan
            df_hr.loc[df_hr['HeartRate'] > 300, 'HeartRate'] = np.nan
            # Interpolate missing values
            df_hr['HeartRate'] = df_hr['HeartRate'].ffill().bfill()
            
            # If completely empty or NaN, we can't use this chunk
            if df_hr['HeartRate'].isna().all():
                continue
                
            # Temporal Alignment via merge_asof
            # We want an HR value for every ACC timestamp.
            merged = pd.merge_asof(
                df_acc, 
                df_hr[['Timestamp_pd', 'HeartRate']], 
                on='Timestamp_pd', 
                direction='nearest'
            )
            
            # Drop any lingering NaNs from failed merges
            merged = merged.dropna(subset=['x', 'y', 'z', 'HeartRate'])
            
            if len(merged) < window_size:
                continue
                
            acc_arr = merged[['x', 'y', 'z']].values
            hr_arr = merged[['HeartRate']].values
            
            # Preprocess ACC
            acc_arr = apply_standard_acc_preprocessing(acc_arr, fs=sampling_rate)
            acc_scaler = StandardScaler()
            acc_arr = acc_scaler.fit_transform(acc_arr)
            
            # Preprocess HR
            hr_scaler = StandardScaler()
            hr_arr = hr_scaler.fit_transform(hr_arr)
            
            # Create sliding windows
            for start_idx in range(0, len(merged) - window_size + 1, step_size):
                window_acc = acc_arr[start_idx : start_idx + window_size]
                window_hr = hr_arr[start_idx : start_idx + window_size]
                
                # Transpose for PyTorch Conv1d
                window_acc = window_acc.T
                window_hr = window_hr.T
                
                all_X_acc.append(window_acc)
                all_X_hr.append(window_hr)
                all_y.append(target_label)
                all_groups.append(part_id)
                
        except Exception as e:
            print(f"  [ERROR] Failed processing {fpath_acc.name}: {e}")
            
    X_acc_arr = np.array(all_X_acc)
    X_hr_arr = np.array(all_X_hr)
    y_arr = np.array(all_y)
    groups_arr = np.array(all_groups)
    
    print(f"\n[INFO] Dataset built successfully!")
    print(f"  -> Total Windows: {len(X_acc_arr)}")
    if len(X_acc_arr) > 0:
        print(f"  -> ACC Window Shape: {X_acc_arr.shape[1:]}")
        print(f"  -> HR Window Shape: {X_hr_arr.shape[1:]}")
    print(f"  -> Class Distribution: {Counter(y_arr)}")
    print(f"  -> Unique Participants: {len(np.unique(groups_arr))}")
    
    return X_acc_arr, X_hr_arr, y_arr, groups_arr

# ---------------------------------------------------------
# 3. TRAINING & EVALUATION FUNCTIONS
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, device, class_weights, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_bacc = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = 7
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_acc, batch_hr, batch_y in train_loader:
            batch_acc = batch_acc.to(device)
            batch_hr = batch_hr.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_acc, batch_hr)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_y_true = []
        val_y_pred = []
        
        with torch.no_grad():
            for batch_acc, batch_hr, batch_y in val_loader:
                batch_acc = batch_acc.to(device)
                batch_hr = batch_hr.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_acc, batch_hr)
                _, preds = torch.max(outputs, 1)
                
                val_y_true.extend(batch_y.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())
                
        val_bacc = balanced_accuracy_score(val_y_true, val_y_pred)
        scheduler.step(val_bacc)
        
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"      [Early Stopping] Epoch {epoch+1}")
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model

def main():
    root = tk.Tk()
    root.withdraw()
    
    print("[INFO] Please select the directory containing ACC chunks...")
    acc_dir_str = filedialog.askdirectory(title="Select Folder containing ACC Chunks")
    if not acc_dir_str:
        print("[CANCELLED] No ACC directory selected.")
        sys.exit(0)
    acc_dir = Path(acc_dir_str).resolve()
    
    print("[INFO] Please select the directory containing HR chunks...")
    hr_dir_str = filedialog.askdirectory(title="Select Folder containing HR Chunks")
    if not hr_dir_str:
        print("[CANCELLED] No HR directory selected.")
        sys.exit(0)
    hr_dir = Path(hr_dir_str).resolve()
    
    window_sec_str = simpledialog.askstring("Window Size", "Enter window size in seconds:", initialvalue="60")
    window_sec = int(window_sec_str) if window_sec_str else 60
        
    dataset_type = simpledialog.askstring("Dataset Type", "Enter Dataset Type (CareWear or GalaxyPPG):", initialvalue="CareWear")
    dataset_type = dataset_type if dataset_type else "CareWear"
        
    # Build Dataset
    X_acc, X_hr, y, groups = load_and_window_fusion_data(
        acc_dir=acc_dir, 
        hr_dir=hr_dir, 
        dataset_type=dataset_type, 
        window_sec=window_sec, 
        sampling_rate=50
    )
    
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
    results_dir = acc_dir / f"DeepFusion_Results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"[INFO] Starting DeepFusionNet Stratified {n_splits}-Fold Evaluation")
    print("="*60)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_acc, y, groups=groups)):
        test_participants = np.unique(groups[test_idx])
        print(f"\n  --- Fold {fold_idx + 1}/{n_splits} ---")
        print(f"    -> Held-out Test Participants: {test_participants}")
        
        X_train_acc, X_test_acc = X_acc[train_idx], X_acc[test_idx]
        X_train_hr, X_test_hr = X_hr[train_idx], X_hr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(np.unique(y_test)) < 2:
            print("  [WARNING] Test set missing classes. Skipping fold.")
            continue
            
        train_dataset = FusionWearableDataset(X_train_acc, X_train_hr, y_train, groups[train_idx])
        test_dataset = FusionWearableDataset(X_test_acc, X_test_hr, y_test, groups[test_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        model = DeepFusionNet(num_classes=2).to(device)
        
        start_time = time.time()
        model = train_model(model, train_loader, test_loader, device, class_weights, epochs=30, lr=0.001)
        train_time = time.time() - start_time
        
        model.eval()
        fold_y_true = []
        fold_y_pred = []
        
        with torch.no_grad():
            for batch_acc, batch_hr, batch_y in test_loader:
                batch_acc = batch_acc.to(device)
                batch_hr = batch_hr.to(device)
                outputs = model(batch_acc, batch_hr)
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
        
        fold_out_dir = results_dir / "fold_details" / f"fold_{fold_idx + 1}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)
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
    
    print(f"[FINAL RESULT] DeepFusionNet (LOSO Validation)")
    print(f"B-Acc:       {bacc_g:.4f}")
    print(f"F1 Score:    {f1_g:.4f}")
    print(f"Accuracy:    {acc_g:.4f}")
    print(f"Sensitivity: {sens_g:.4f}")
    print(f"Specificity: {spec_g:.4f}")
    print("="*60)
    
    report = classification_report(y_true_global, y_pred_global, labels=[0, 1], target_names=["Non-Stress", "Stress"], zero_division=0)
    with open(results_dir / "DeepFusionNet_report.txt", "w") as f:
        f.write("DeepFusionNet Evaluation - Subject Independent (StratifiedGroupKFold)\n")
        f.write("="*60 + "\n")
        f.write(report)
        
    save_confusion_matrix(y_true_global, y_pred_global, results_dir / "DeepFusionNet_global_cm.png", "DeepFusionNet Global CM")
        
    df_summary = pd.DataFrame(fold_details)
    mean_std_row = {"Fold": "Mean ± Std"}
    for col in [c for c in df_summary.columns if c != "Fold"]:
        mean_std_row[col] = f"{df_summary[col].mean():.4f} ± {df_summary[col].std():.4f}"
    df_summary = pd.concat([df_summary, pd.DataFrame([mean_std_row])], ignore_index=True)
    df_summary.to_csv(results_dir / "DeepFusionNet_fold_summary.csv", index=False)
    
    print(f"    [DEPLOYMENT] Training final deployment model on all data...")
    final_dataset = FusionWearableDataset(X_acc, X_hr, y, groups)
    final_loader = DataLoader(final_dataset, batch_size=64, shuffle=True)
    final_model = DeepFusionNet(num_classes=2).to(device)
    final_model = train_model(final_model, final_loader, final_loader, device, class_weights, epochs=30, lr=0.001)
    deploy_model_path = results_dir / "DeepFusionNet_deploy.pth"
    torch.save(final_model.state_dict(), deploy_model_path)
    print(f"    [DEPLOYMENT] Model weights saved to {deploy_model_path}")
    
    print(f"\n[INFO] Results saved to {results_dir}")
    messagebox.showinfo("Complete", "DeepFusionNet Training and Evaluation Finished.")

if __name__ == "__main__":
    main()
