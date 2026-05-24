import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import traceback
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from collections import Counter
from datetime import datetime

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, 
    recall_score, precision_score, classification_report, roc_auc_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# 1. SCIENTIFIC FEATURE SETS DEFINITION
FEATURE_SETS = {
    # --- HR FEATURES ---
    "HR_1_TimeDomain_Stats": [
        'hr_mean', 'hr_median', 'hr_std', 'hr_min', 'hr_max', 
        'hr_iqr', 'hr_skew', 'hr_kurtosis', 'hr_range'
    ],
    "HR_2_HRV_Proxies": [
        'hr_rmssd', 'hr_pnn50', 'hr_sdsd', 'hr_sampen'
    ],
    "HR_3_Temporal_Dynamics": [
        'hr_slope', 'hr_second_derivative', 'hr_macd_mean', 'hr_start', 'hr_end'
    ],
    "HR_4_Contextual_Baseline": [
        'hr_perc', 'hr_recovery_window_1', 'hr_recovery_window_2'
    ],
    "HR_5_HR_Zones": [
        'HR Range: 0–40 bpm', 'HR Range: 40–60 bpm', 'HR Range: 60–80 bpm',
        'HR Range: 80–100 bpm', 'HR Range: 100–120 bpm', 'HR Range: 120–140 bpm',
        'HR Range: 140–160 bpm', 'HR Range: 160–180 bpm', 'HR Range: 180–200 bpm',
        'HR Range: >200 bpm'
    ],

    # --- ACCELEROMETER FEATURES ---
    "ACC_1_TimeDomain_Kinematics": [
        'Filtered_x_mean', 'Filtered_x_std', 'Filtered_x_max', 'Filtered_x_min', 'Filtered_x_skew', 'Filtered_x_kurtosis', 'Filtered_x_iqr', 'Filtered_x_zcr', 'Filtered_x_mcr',
        'Filtered_y_mean', 'Filtered_y_std', 'Filtered_y_max', 'Filtered_y_min', 'Filtered_y_skew', 'Filtered_y_kurtosis', 'Filtered_y_iqr', 'Filtered_y_zcr', 'Filtered_y_mcr',
        'Filtered_z_mean', 'Filtered_z_std', 'Filtered_z_max', 'Filtered_z_min', 'Filtered_z_skew', 'Filtered_z_kurtosis', 'Filtered_z_iqr', 'Filtered_z_zcr', 'Filtered_z_mcr',
        'Filtered_max_acc_mean', 'Filtered_max_acc_std', 'Filtered_max_acc_max', 'Filtered_max_acc_min', 'Filtered_max_acc_skew', 'Filtered_max_acc_kurtosis', 'Filtered_max_acc_iqr', 'Filtered_max_acc_zcr', 'Filtered_max_acc_mcr'
    ],
    "ACC_2_Frequency_and_Band_Power": [
        'Filtered_x_max_power_psd', 'Filtered_x_min_power_psd', 'Filtered_x_skewness_power_psd', 'Filtered_x_kurtosis_power_psd', 'Filtered_x_mean_power_psd', 'Filtered_x_sum_freq_diff_power_psd', 'Filtered_x_average_power_psd', 'Filtered_x_num_peaks_power_psd',
        'Filtered_y_max_power_psd', 'Filtered_y_min_power_psd', 'Filtered_y_skewness_power_psd', 'Filtered_y_kurtosis_power_psd', 'Filtered_y_mean_power_psd', 'Filtered_y_sum_freq_diff_power_psd', 'Filtered_y_average_power_psd', 'Filtered_y_num_peaks_power_psd',
        'Filtered_z_max_power_psd', 'Filtered_z_min_power_psd', 'Filtered_z_skewness_power_psd', 'Filtered_z_kurtosis_power_psd', 'Filtered_z_mean_power_psd', 'Filtered_z_sum_freq_diff_power_psd', 'Filtered_z_average_power_psd', 'Filtered_z_num_peaks_power_psd',
        'Filtered_max_acc_max_power_psd', 'Filtered_max_acc_min_power_psd', 'Filtered_max_acc_skewness_power_psd', 'Filtered_max_acc_kurtosis_power_psd', 'Filtered_max_acc_mean_power_psd', 'Filtered_max_acc_sum_freq_diff_power_psd', 'Filtered_max_acc_average_power_psd', 'Filtered_max_acc_num_peaks_power_psd',
        'Filtered_x_pa_band_power', 'Filtered_x_stress_band_power', 'Filtered_x_stress_pa_power_ratio',
        'Filtered_y_pa_band_power', 'Filtered_y_stress_band_power', 'Filtered_y_stress_pa_power_ratio',
        'Filtered_z_pa_band_power', 'Filtered_z_stress_band_power', 'Filtered_z_stress_pa_power_ratio',
        'Filtered_max_acc_pa_band_power', 'Filtered_max_acc_stress_band_power', 'Filtered_max_acc_stress_pa_power_ratio'
    ],
    "ACC_3_Complexity_and_Coordination": [
        'Filtered_x_spectral_entropy', 'Filtered_y_spectral_entropy', 'Filtered_z_spectral_entropy', 'Filtered_max_acc_spectral_entropy',
        'corr_xy', 'corr_xz', 'corr_yz'
    ],
    "ACC_4_Literature_Optimal_Combined": [
        'Filtered_max_acc_spectral_entropy', 'corr_xy', 'corr_xz', 'corr_yz',
        'Filtered_x_stress_pa_power_ratio', 'Filtered_y_stress_pa_power_ratio', 'Filtered_z_stress_pa_power_ratio', 'Filtered_max_acc_stress_pa_power_ratio',
        'Filtered_max_acc_std', 'Filtered_max_acc_iqr', 'Filtered_max_acc_zcr',
        'Filtered_max_acc_mean_power_psd', 'Filtered_max_acc_average_power_psd'
    ]
}

# Add combined
all_features = set()
for feat_list in FEATURE_SETS.values():
    all_features.update(feat_list)
FEATURE_SETS["6_All_Combined"] = list(all_features)


def mild_under_strategy(y):
    counts = Counter(y)
    if counts[1] == 0 or counts[0] == 0:
        return counts
    target_0 = min(counts[0], counts[1] * 2)
    return {0: target_0, 1: counts[1]}

def get_model_and_grid(model_name, y_train_for_weights=None):
    # inner CV for search
    cv_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    if model_name == "RF":
        # n_jobs=1 prevents thread contention with RandomizedSearchCV
        estimator = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)
        grid = {
            'model__n_estimators': [50, 100, 300, 500],
            'model__max_depth': [10, 20, 30, None], 
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2']
        }
    elif model_name == "GB":
        estimator = GradientBoostingClassifier(random_state=42)
        grid = {
            'model__n_estimators': [50, 100, 300, 500],
            'model__max_depth': [3, 5, 7, 10], 
            'model__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.8, 1.0]
        }
    elif model_name == "XGB":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed.")
        # Calculate dynamic class weights 
        scale_pw = 1.0
        if y_train_for_weights is not None:
            counts = Counter(y_train_for_weights)
            if counts[1] > 0:
                scale_pw = float(counts[0]) / counts[1]
        
        # n_jobs=1 prevents thread contention
        estimator = XGBClassifier(scale_pos_weight=scale_pw, eval_metric='logloss', random_state=42, n_jobs=1)
        grid = {
            'model__n_estimators': [50, 100, 300, 500],
            'model__max_depth': [3, 5, 7, 10], 
            'model__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.8, 1.0]
        }
    elif model_name == "LR":
        estimator = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        grid = {
            'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga']
        }
    elif model_name == "SVM":
        estimator = SVC(class_weight="balanced", probability=True, random_state=42)
        grid = {
            # Pruned to avoid the O(n^3) computational trap
            'model__C': [0.1, 1.0, 10.0], 
            'model__kernel': ['linear']
        }
    elif model_name == "DT":
        estimator = DecisionTreeClassifier(class_weight="balanced", random_state=42)
        grid = {
            'model__max_depth': [5, 10, 20, 30, None], 
            'model__min_samples_split': [2, 5, 10, 20],
            'model__min_samples_leaf': [1, 2, 4, 8],
            'model__criterion': ['gini', 'entropy']
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return estimator, grid, cv_strategy

def standardize_participant_activity(X, groups, activities):
    X_scaled = X.copy().astype(float)
    scaler = StandardScaler()
    
    unique_combinations = pd.DataFrame({'pid': groups, 'act': activities}).drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        pid = row['pid']
        act = row['act']
        
        mask = (groups == pid) & (activities == act)
        if mask.sum() > 1:
            X_scaled.loc[mask, X_scaled.columns] = scaler.fit_transform(X_scaled.loc[mask])
        elif mask.sum() == 1:
            X_scaled.loc[mask, X_scaled.columns] = 0
            
    return X_scaled

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

def identify_columns(df):
    id_col = None
    target_col = None
    # common identifiers in both scripts
    if "Participant" in df.columns: id_col = "Participant"
    elif "participantId" in df.columns: id_col = "participantId"

    if "Activity_Int" in df.columns: target_col = "Activity_Int"
    elif "BehaviorCode" in df.columns: target_col = "BehaviorCode"
    
    return id_col, target_col

def compile_master_summaries(results_base_dir):
    print("\n" + "="*60)
    print("[INFO] Compiling Master Summaries...")
    
    for base_dir in results_base_dir.iterdir():
        if not base_dir.is_dir():
            continue
            
        master_rows = []
        for set_dir in base_dir.iterdir():
            if not set_dir.is_dir():
                continue
            set_name = set_dir.name
            
            for model_dir in set_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                
                summary_file = model_dir / f"{model_name}_fold_summary.csv"
                if summary_file.exists():
                    try:
                        df = pd.read_csv(summary_file)
                        mean_std_row = df[df['Fold'] == 'Mean ± Std'].copy()
                        if not mean_std_row.empty:
                            row_dict = mean_std_row.iloc[0].to_dict()
                            row_dict['Feature Set'] = set_name
                            row_dict['Model'] = model_name
                            master_rows.append(row_dict)
                    except Exception as e:
                        print(f"  [WARNING] Could not read {summary_file}: {e}")
                        
        if master_rows:
            master_df = pd.DataFrame(master_rows)
            cols = ['Feature Set', 'Model'] + [c for c in master_df.columns if c not in ['Feature Set', 'Model', 'Fold', 'Search Best Params']]
            master_df = master_df[cols]
            
            out_path = base_dir / f"{base_dir.name}_master_summary.csv"
            master_df.to_csv(out_path, index=False)
            print(f"  -> Saved master summary: {out_path}")
            
    print("="*60 + "\n")

def main():
    root = tk.Tk()
    root.withdraw()
    
    target_dir_str = filedialog.askdirectory(title="Select Folder containing Feature CSVs")
    if not target_dir_str:
        print("[CANCELLED] No directory selected.")
        sys.exit(0)
        
    target_dir = Path(target_dir_str).resolve()
    
    version = simpledialog.askstring("Version", "Enter a version name for this run (e.g., v1, v2):", initialvalue="v1")
    if not version:
        version = "v1"
        
    results_base_dir = target_dir / f"Results_Stratified_5Fold_{version}"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    models_str = simpledialog.askstring(
        "Models Setup", 
        "Select Models to run (comma separated):\nAvailable: RF, XGB, GB, LR, SVM, DT", 
        initialvalue="RF, XGB, GB, LR, SVM, DT"
    )
    if not models_str:
        models_to_run = ["RF", "XGB", "GB", "LR", "SVM", "DT"]
    else:
        models_to_run = [m.strip().upper() for m in models_str.split(',')]

    if "SVM" in models_to_run:
        models_to_run.remove("SVM")
        models_to_run.append("SVM")

    dataset_type = simpledialog.askstring(
        "Dataset Type", 
        "Enter Dataset Type (CareWear or GalaxyPPG):", 
        initialvalue="CareWear"
    )
    if not dataset_type:
        dataset_type = "CareWear"

    print("="*60)
    print(f"[INFO] 5-Fold Stratified Auto-Experiment Runner Initialized (Optimized Version)")
    print(f"[INFO] Selected Directory: {target_dir}")
    print(f"[INFO] Base Output Folder: {results_base_dir}")
    print(f"[INFO] Models: {', '.join(models_to_run)}")
    print(f"[INFO] Dataset Type: {dataset_type}")
    print("="*60)

    if dataset_type.lower() == "carewear":
        stress_mapping = {
            1: -1, 2: 1, 3: 1, 4: -1, 5: 1, 
            6: -1, 7: 0, 8: 0
        }
        print("[INFO] Applied CareWear Stress Mapping (2,3,5 -> 1 | 7,8 -> 0)")
    else:
        # Mapping for GalaxyPPG dataset
        # Stress mapped to 0 (3,4,7,8)
        # Not-Stress mapped to 1 (16,18,20)
        # Rest mapped to -1 to be excluded
        stress_mapping = {
            1: -1, 2: -1, 3: 0, 4: 0, 5: -1, 
            6: -1, 7: 0, 8: 0, 9: -1, 10: -1,
            11: -1, 12: -1, 13: -1, 14: -1, 15: -1,
            16: 1, 17: -1, 18: 1, 19: -1, 20: 1
        }
        print("[INFO] Applied GalaxyPPG Stress Mapping")

    # Smart CSV validation logic
    csv_files = [f for f in target_dir.rglob("*.csv") if "Results_" not in str(f)]
    valid_csvs = []
    
    for f in csv_files:
        try:
            cols = pd.read_csv(f, nrows=0).columns
            valid_feats = [c for c in FEATURE_SETS["6_All_Combined"] if c in cols]
            id_c, _ = identify_columns(pd.DataFrame(columns=cols))
            
            # Identify it as a valid feature table if it has an ID grouping column
            # and AT LEAST 5 of the known features mapped.
            if id_c is not None and len(valid_feats) > 5:
                valid_csvs.append(f)
        except Exception as e:
            continue

    if not valid_csvs:
        print("[ERROR] No valid feature tables found in the selected directory.")
        sys.exit(1)

    print(f"[INFO] Found {len(valid_csvs)} valid feature tables to process.\n")

    models_first_pass = [m for m in models_to_run if m != "SVM"]
    models_second_pass = ["SVM"] if "SVM" in models_to_run else []
    
    # Reorder FEATURE_SETS to ensure "6_All_Combined" runs first for early exit logic
    ordered_sets = ["6_All_Combined"] + [k for k in FEATURE_SETS.keys() if k != "6_All_Combined"]
    
    for pass_name, active_models in [("Pass 1: Core Models", models_first_pass), ("Pass 2: SVM Only", models_second_pass)]:
        if not active_models:
            continue
        print(f"\n{'='*80}\n STARTING {pass_name} -> {active_models}\n{'='*80}\n")
        
        for csv_path in valid_csvs:
            base_filename = csv_path.stem
            print(f"\n{'#'*60}\n# Processing File: {base_filename}\n{'#'*60}")

            # If the base folder for this CSV already exists, skip it entirely
            file_results_dir = results_base_dir / base_filename
            if file_results_dir.exists() and any(file_results_dir.iterdir()):
                print(f"[SKIP] {base_filename} already exists in results folder. Skipping entirely.")
                continue

            df = pd.read_csv(csv_path)
            id_col, target_col = identify_columns(df)

            if not id_col or not target_col:
                print(f"[SKIP] {base_filename} is missing ID/Target columns.")
                continue

            df = df.dropna(subset=[target_col, id_col])

            if target_col == "Activity_Int":
                df['target'] = df[target_col].map(stress_mapping)
            else:
                df['target'] = pd.to_numeric(df[target_col], errors='coerce')

            df = df.dropna(subset=['target'])
            df['target'] = df['target'].astype(int)
            df = df[df['target'].isin([0, 1])]

            # Data Quality Mask: Drop rows where > 20% of ALL possible features are NaN (e.g., motion artifacts)
            combined_feats = [f for f in FEATURE_SETS["6_All_Combined"] if f in df.columns]
            if len(combined_feats) > 0:
                missing_ratio = df[combined_feats].isnull().mean(axis=1)
                df = df[missing_ratio <= 0.2]

            groups = df[id_col]
            activity_col = "activityID" if "activityID" in df.columns else target_col
            activities = df[activity_col]

            # Track combined B-Acc for early exits
            model_performance_tracker = {}

            for set_name in ordered_sets:
                selected_features = FEATURE_SETS[set_name]
                print(f"\n  *** FEATURE SET: {set_name} ***")
                valid_selected = [f for f in selected_features if f in df.columns]

                if len(valid_selected) == 0:
                    print(f"  [SKIP] No valid features found for set {set_name}.")
                    continue

                X_raw = df[valid_selected]
                y = df['target']

                print("  Applying participant + activity wise standardization...")
                X_scaled = standardize_participant_activity(X_raw, groups, activities)

                for model_name in active_models:
                    # Early Exit Check (DISABLED BY REQUEST)
                    # if set_name != "6_All_Combined":
                    #     if model_performance_tracker.get(model_name, 1.0) < 0.60:
                    #         print(f"  [EARLY EXIT] Skipping {model_name} on {set_name} (Combined B-Acc was < 0.60)")
                    #         continue

                    model_out_dir = results_base_dir / base_filename / set_name / model_name
                    model_out_dir.mkdir(parents=True, exist_ok=True)

                    global_report_path = model_out_dir / f"{model_name}_skfold_report.txt"
                    deploy_model_path = model_out_dir / f"{base_filename}_{set_name}_{model_name}_deploy.pkl"
                    
                    if global_report_path.exists() and deploy_model_path.exists():
                        print(f"  [SKIP] {model_name} already completed. (Found report and deploy model)")
                        
                        # If we are skipping the combined set, we must recover the B-Acc for early exit checks
                        if set_name == "6_All_Combined":
                            summary_path = model_out_dir / f"{model_name}_fold_summary.csv"
                            if summary_path.exists():
                                try:
                                    sum_df = pd.read_csv(summary_path)
                                    mean_bacc_str = sum_df[sum_df['Fold'] == 'Mean ± Std']['Fold Balanced Acc'].values[0]
                                    mean_bacc = float(mean_bacc_str.split(' ± ')[0])
                                    model_performance_tracker[model_name] = mean_bacc
                                except Exception as e:
                                    print(f"  [WARNING] Could not read B-Acc for early exit tracking from {summary_path}: {e}")
                                    model_performance_tracker[model_name] = 1.0 # Fail safe, don't early exit
                            else:
                                model_performance_tracker[model_name] = 1.0

                        continue

                    print(f"  --- Running Model: {model_name} ---")

                    n_splits = 5
                    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

                    y_true_out, y_pred_out, y_prob_out = [], [], []
                    fold_details = []
                    fold_idx = 1

                    try:
                        for train_idx, test_idx in skf.split(X_scaled, y, groups=groups):
                            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                            if len(np.unique(y_test)) == 0 or len(np.unique(y_train)) < 2:
                                continue

                            estimator, param_grid, cv_inner = get_model_and_grid(model_name, y_train)

                            pipeline = ImbPipeline([
                                ('imputer', SimpleImputer(strategy="median")),
                                ('sampler', RandomUnderSampler(sampling_strategy=mild_under_strategy, random_state=42)),
                                ('model', estimator)
                            ])

                            # Replaced GridSearchCV with RandomizedSearchCV
                            random_search = RandomizedSearchCV(
                                estimator=pipeline,
                                param_distributions=param_grid,
                                n_iter=20, # Cap at 20 searches for 10x speedup
                                cv=cv_inner,
                                scoring='balanced_accuracy',
                                n_jobs=-1,
                                random_state=42
                            )

                            print(f"      -> Fold {fold_idx}/{n_splits} ...", end=" ", flush=True)
                            start_time = time.time()
                            random_search.fit(X_train, y_train)
                            train_time = time.time() - start_time

                            best_model = random_search.best_estimator_
                            y_pred = best_model.predict(X_test)

                            if hasattr(best_model, "predict_proba"):
                                y_prob = best_model.predict_proba(X_test)[:, 1]
                            else:
                                y_prob = [0.5] * len(y_test)

                            fold_bacc = balanced_accuracy_score(y_test, y_pred)
                            fold_f1 = f1_score(y_test, y_pred, zero_division=0)

                            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                            tn, fp, fn, tp = cm.ravel()
                            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            fold_acc = accuracy_score(y_test, y_pred)

                            try:
                                fold_roc = roc_auc_score(y_test, y_prob)
                            except ValueError:
                                fold_roc = float('nan')

                            print(f"Done! | B-Acc: {fold_bacc:.2f} | F1: {fold_f1:.2f}")

                            y_true_out.extend(y_test)
                            y_pred_out.extend(y_pred)
                            y_prob_out.extend(y_prob)

                            fold_details.append({
                                "Fold": fold_idx,
                                "Train Samples": len(X_train),
                                "Test Samples": len(X_test),
                                "Fold Accuracy": fold_acc,
                                "Fold Balanced Acc": fold_bacc,
                                "Fold F1 Score": fold_f1,
                                "Fold Sensitivity": sens,
                                "Fold Specificity": spec,
                                "Fold ROC-AUC": fold_roc,
                                "Train Time (s)": train_time,
                                "Search Best Params": str(random_search.best_params_)
                            })

                            fold_out_dir = model_out_dir / "fold_details" / f"fold_{fold_idx}"
                            fold_out_dir.mkdir(parents=True, exist_ok=True)
                            save_confusion_matrix(y_test, y_pred, fold_out_dir / f"cm_fold_{fold_idx}.png", f"Fold {fold_idx} CM")

                            fold_idx += 1

                        if len(y_true_out) > 0:
                            bacc = balanced_accuracy_score(y_true_out, y_pred_out)
                            f1_g = f1_score(y_true_out, y_pred_out, zero_division=0)
                            
                            # Log combined B-Acc for early exit strategy
                            if set_name == "6_All_Combined":
                                model_performance_tracker[model_name] = bacc

                            cm_global = confusion_matrix(y_true_out, y_pred_out, labels=[0, 1])
                            tn_g, fp_g, fn_g, tp_g = cm_global.ravel()
                            spec_g = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0.0
                            sens_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
                            mean_train_time = np.mean([f["Train Time (s)"] for f in fold_details])

                            df_summary = pd.DataFrame(fold_details)
                            numeric_cols = [c for c in df_summary.columns if c not in ["Fold", "Search Best Params"]]
                            mean_std_row = {"Fold": "Mean ± Std"}
                            for col in numeric_cols:
                                m = df_summary[col].mean()
                                s = df_summary[col].std()
                                mean_std_row[col] = f"{m:.4f} ± {s:.4f}"
                            
                            df_summary = pd.concat([df_summary, pd.DataFrame([mean_std_row])], ignore_index=True)
                            df_summary.to_csv(model_out_dir / f"{model_name}_fold_summary.csv", index=False)

                            report = classification_report(y_true_out, y_pred_out, labels=[0, 1], target_names=["Non-Stress", "Stress"], zero_division=0)
                            with open(global_report_path, "w") as f:
                                f.write(report)

                            save_confusion_matrix(y_true_out, y_pred_out, model_out_dir / f"{model_name}_skfold_cm.png", f"{model_name} Global CM")

                            print(f"    [RESULT] Global B-Acc: {bacc:.4f} | F1: {f1_g:.4f} | Sens: {sens_g:.4f} | Spec: {spec_g:.4f} | Mean Train Time: {mean_train_time:.2f}s")

                            print(f"    [DEPLOYMENT] Training final model on all data...")
                            final_est, final_param_grid, final_cv = get_model_and_grid(model_name, y)

                            final_pipeline = ImbPipeline([
                                ('imputer', SimpleImputer(strategy="median")),
                                ('sampler', RandomUnderSampler(sampling_strategy=mild_under_strategy, random_state=42)),
                                ('model', final_est)
                            ])

                            n_groups = groups.nunique()
                            inner_cv = StratifiedGroupKFold(n_splits=min(3, n_groups)) if n_groups > 1 else 2

                            # Replaced GridSearchCV with RandomizedSearchCV for Deployment model as well
                            final_search = RandomizedSearchCV(
                                estimator=final_pipeline,
                                param_distributions=final_param_grid,
                                n_iter=20,
                                cv=inner_cv,
                                scoring='balanced_accuracy',
                                n_jobs=-1,
                                random_state=42
                            )

                            deploy_start_time = time.time()
                            final_search.fit(X_scaled, y, groups=groups)
                            deploy_train_time = time.time() - deploy_start_time

                            deploy_model_path = model_out_dir / f"{base_filename}_{set_name}_{model_name}_deploy.pkl"
                            joblib.dump(final_search.best_estimator_, deploy_model_path)
                            print(f"    [DEPLOYMENT] Model saved to {deploy_model_path} (Trained in {deploy_train_time:.2f}s)")

                    except Exception as e:
                        print(f"  [ERROR] {model_name} failed: {e}")
                        traceback.print_exc()

    compile_master_summaries(results_base_dir)

    messagebox.showinfo("Complete", "Automated 5-Fold Feature Set Benchmarking Finished.")

if __name__ == "__main__":
    main()