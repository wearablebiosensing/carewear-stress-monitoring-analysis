import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
# 1. DEEPCONVLSTM MODEL ARCHITECTURE
# ---------------------------------------------------------
class DeepConvLSTM(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, sequence_length=3000):
        super(DeepConvLSTM, self).__init__()
        
        # 1D Convolutional blocks
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layer
        # Output sequence length after 3 max pools: sequence_length / 8
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        
        # Fully connected layer
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        # Reshape for LSTM: (Batch, Sequence_Length, Features)
        x = x.permute(0, 2, 1)
        
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        last_out = out[:, -1, :]
        
        x = self.dropout(last_out)
        x = self.fc(x)
        
        return x

# ---------------------------------------------------------
# 2. DATASET DEFINITION & LOADING
# ---------------------------------------------------------
class WearableDataset(Dataset):
    def __init__(self, X, y, groups):
        # X: (N, Channels, Seq_Len)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def extract_metadata_from_filename(filename):
    """
    Parses 'activity_id_1.0_acc_10_merged_labels.csv' to get activity and participant.
    """
    pattern = r"activity_id_([\d\.]+)_acc_(\d+)_merged_labels\.csv"
    match = re.search(pattern, filename)
    if match:
        act = float(match.group(1))
        part = int(match.group(2))
        return act, part
    return None, None

def apply_standard_acc_preprocessing(data_arr, fs=50):
    """
    Applies the standard HAR signal processing pipeline to raw accelerometer data:
    1. Median Filter (kernel=3) to remove sudden spikes/noise.
    2. 3rd-order Butterworth Bandpass Filter (0.3 Hz - 20 Hz):
       - High-pass (0.3 Hz) removes the standard gravity component.
       - Low-pass (20 Hz) removes high-frequency jitter.
    """
    processed = np.zeros_like(data_arr)
    
    # Bandpass filter design
    nyq = 0.5 * fs
    low = 0.3 / nyq
    high = 20.0 / nyq
    b, a = butter(3, [low, high], btype='bandpass')
    
    for i in range(data_arr.shape[1]): # Iterate over x, y, z
        # Apply median filter
        med_filtered = medfilt(data_arr[:, i], kernel_size=3)
        # Apply Butterworth bandpass
        processed[:, i] = filtfilt(b, a, med_filtered)
        
    return processed

def load_and_window_data(data_dir, dataset_type="CareWear", window_sec=60, sampling_rate=50):
    print("\n[INFO] Scanning for raw chunks and building Deep Learning dataset...")
    
    # Mappings
    if dataset_type.lower() == "carewear":
        stress_mapping = {1: -1, 2: 1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 0, 8: 0}
    else:
        # GalaxyPPG
        stress_mapping = {
            1: -1, 2: -1, 3: 0, 4: 0, 5: -1, 6: -1, 7: 0, 8: 0, 9: -1, 10: -1,
            11: -1, 12: -1, 13: -1, 14: -1, 15: -1, 16: 1, 17: -1, 18: 1, 19: -1, 20: 1
        }
    
    window_size = int(window_sec * sampling_rate)
    overlap = int(window_size * 0.5) # 50% overlap to increase dataset size
    step_size = window_size - overlap
    
    all_X = []
    all_y = []
    all_groups = []
    
    csv_files = list(Path(data_dir).rglob("activity_id_*_acc_*_merged_labels.csv"))
    
    print(f"  -> Found {len(csv_files)} potential chunk files.")
    
    scaler = StandardScaler()
    
    for fpath in csv_files:
        act_id, part_id = extract_metadata_from_filename(fpath.name)
        if act_id is None or part_id is None:
            continue
            
        target_label = stress_mapping.get(act_id, -1)
        if target_label == -1:
            continue # Ignore activities not mapped to stress (1) or non-stress (0)
            
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if not all(c in df.columns for c in ['x', 'y', 'z']):
                continue
                
            # Safely coerce columns to numeric to handle corrupted strings like '2.462.5139117'
            for col in ['x', 'y', 'z']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Filter NaNs
            df = df.dropna(subset=['x', 'y', 'z'])
            if len(df) < window_size:
                continue
                
            data_arr = df[['x', 'y', 'z']].values
            
            # Apply formal signal processing to remove gravity & noise
            data_arr = apply_standard_acc_preprocessing(data_arr, fs=sampling_rate)
            
            # Normalize processed data for Neural Network gradient stability
            data_arr = scaler.fit_transform(data_arr)
            
            # Create sliding windows
            for start_idx in range(0, len(data_arr) - window_size + 1, step_size):
                window = data_arr[start_idx : start_idx + window_size]
                # Transpose from (Seq_Len, Channels) to (Channels, Seq_Len) for PyTorch Conv1d
                window = window.T
                all_X.append(window)
                all_y.append(target_label)
                all_groups.append(part_id)
                
        except Exception as e:
            print(f"  [ERROR] Failed to read {fpath.name}: {e}")
            
    X_arr = np.array(all_X)
    y_arr = np.array(all_y)
    groups_arr = np.array(all_groups)
    
    print(f"\n[INFO] Dataset built successfully!")
    print(f"  -> Total Windows: {len(X_arr)}")
    if len(X_arr) > 0:
        print(f"  -> Window Shape: {X_arr.shape[1:]} (Channels, Seq_Len)")
    print(f"  -> Class Distribution: {Counter(y_arr)}")
    print(f"  -> Unique Participants: {len(np.unique(groups_arr))}")
    
    return X_arr, y_arr, groups_arr

# ---------------------------------------------------------
# 3. TRAINING & EVALUATION FUNCTIONS
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, device, class_weights, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # verbose keyword is deprecated in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_bacc = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = 7
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_y_true = []
        val_y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
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
    
    target_dir_str = filedialog.askdirectory(title="Select Folder containing ACC Chunks")
    if not target_dir_str:
        print("[CANCELLED] No directory selected.")
        sys.exit(0)
        
    target_dir = Path(target_dir_str).resolve()
    
    window_sec_str = simpledialog.askstring("Window Size", "Enter window size in seconds:", initialvalue="60")
    if not window_sec_str:
        window_sec = 60
    else:
        window_sec = int(window_sec_str)
        
    dataset_type = simpledialog.askstring(
        "Dataset Type", 
        "Enter Dataset Type (CareWear or GalaxyPPG):", 
        initialvalue="CareWear"
    )
    if not dataset_type:
        dataset_type = "CareWear"
        
    # Build Dataset
    X, y, groups = load_and_window_data(target_dir, dataset_type=dataset_type, window_sec=window_sec, sampling_rate=50)
    
    if len(X) == 0:
        print("[ERROR] No data extracted. Exiting.")
        return
        
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using Device: {device}")
    
    # Calculate Class Weights to handle imbalance
    counts = Counter(y)
    total = len(y)
    class_weights = torch.tensor([total / counts[0], total / counts[1]], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    
    # CV setup
    n_splits = 5
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_true_global = []
    y_pred_global = []
    fold_details = []
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = target_dir / f"DeepLearning_Results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"[INFO] Starting DeepConvLSTM Stratified {n_splits}-Fold Evaluation")
    print("="*60)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=groups)):
        test_participants = np.unique(groups[test_idx])
        print(f"\n  --- Fold {fold_idx + 1}/{n_splits} ---")
        print(f"    -> Held-out Test Participants: {test_participants}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check if fold is valid
        if len(np.unique(y_test)) < 2:
            print("  [WARNING] Test set missing classes. Skipping fold.")
            continue
            
        train_dataset = WearableDataset(X_train, y_train, groups[train_idx])
        test_dataset = WearableDataset(X_test, y_test, groups[test_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize model
        model = DeepConvLSTM(in_channels=3, num_classes=2, sequence_length=X.shape[2]).to(device)
        
        # Train
        start_time = time.time()
        model = train_model(model, train_loader, test_loader, device, class_weights, epochs=30, lr=0.001)
        train_time = time.time() - start_time
        
        # Evaluate
        model.eval()
        fold_y_true = []
        fold_y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
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
        
    # Global Evaluation
    print("\n" + "="*60)
    bacc_g = balanced_accuracy_score(y_true_global, y_pred_global)
    f1_g = f1_score(y_true_global, y_pred_global, zero_division=0)
    acc_g = accuracy_score(y_true_global, y_pred_global)
    
    cm_global = confusion_matrix(y_true_global, y_pred_global, labels=[0, 1])
    tn_g, fp_g, fn_g, tp_g = cm_global.ravel()
    spec_g = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0.0
    sens_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
    
    print(f"[FINAL RESULT] DeepConvLSTM (LOSO Validation)")
    print(f"B-Acc:       {bacc_g:.4f}")
    print(f"F1 Score:    {f1_g:.4f}")
    print(f"Accuracy:    {acc_g:.4f}")
    print(f"Sensitivity: {sens_g:.4f}")
    print(f"Specificity: {spec_g:.4f}")
    print("="*60)
    
    # Save Report
    report = classification_report(y_true_global, y_pred_global, labels=[0, 1], target_names=["Non-Stress", "Stress"], zero_division=0)
    with open(results_dir / "DeepConvLSTM_report.txt", "w") as f:
        f.write("DeepConvLSTM Evaluation - Subject Independent (StratifiedGroupKFold)\n")
        f.write("="*60 + "\n")
        f.write(report)
        
    save_confusion_matrix(y_true_global, y_pred_global, results_dir / "DeepConvLSTM_global_cm.png", "DeepConvLSTM Global CM")
        
    df_summary = pd.DataFrame(fold_details)
    mean_std_row = {"Fold": "Mean ± Std"}
    for col in [c for c in df_summary.columns if c != "Fold"]:
        mean_std_row[col] = f"{df_summary[col].mean():.4f} ± {df_summary[col].std():.4f}"
    df_summary = pd.concat([df_summary, pd.DataFrame([mean_std_row])], ignore_index=True)
    df_summary.to_csv(results_dir / "DeepConvLSTM_fold_summary.csv", index=False)
    
    print(f"    [DEPLOYMENT] Training final deployment model on all data...")
    final_dataset = WearableDataset(X, y, groups)
    final_loader = DataLoader(final_dataset, batch_size=64, shuffle=True)
    final_model = DeepConvLSTM(in_channels=3, num_classes=2, sequence_length=X.shape[2]).to(device)
    final_model = train_model(final_model, final_loader, final_loader, device, class_weights, epochs=30, lr=0.001)
    deploy_model_path = results_dir / "DeepConvLSTM_deploy.pth"
    torch.save(final_model.state_dict(), deploy_model_path)
    print(f"    [DEPLOYMENT] Model weights saved to {deploy_model_path}")
    
    print(f"\n[INFO] Results saved to {results_dir}")
    messagebox.showinfo("Complete", "DeepConvLSTM Training and Evaluation Finished.")

if __name__ == "__main__":
    main()
