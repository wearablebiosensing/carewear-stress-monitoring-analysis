import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def mild_under_strategy(y):
    counts = Counter(y)
    if counts[1] == 0 or counts[0] == 0:
        return counts
    # Ensure majority class (0) is exactly 2x the minority class (1)
    target_0 = min(counts[0], counts[1] * 2)
    return {0: target_0, 1: counts[1]}

# Directories
PROJECT_DIR = "/home/wbl-hpc/Desktop/FIdgetSense/q2behave_data_analysis-ss-dev-clanup"
DATA_DIR = "/media/wbl-hpc/ss/Project_Q2behave/DATASET/code_outputs/2026/"
OUT_DIR = os.path.join(DATA_DIR, "results_stratified_kfold")

# Creating Base Output Folder
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "Time_Domain": "time_domain_features_two_second_0.5_10.csv",
    "PSD": "psd_features_two_second_0.5_10.csv",
    "Statistical": "stat_features_two_second_0.5_10.csv",
    "Combined": "all_features_two_second_0.5_10.csv"
}

def load_and_preprocess(filepath):
    print(f"Loading data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None, None, None
        
    df = pd.read_csv(filepath)
    df = df.dropna()
    
    if 'BehaviorCode' not in df.columns:
        print("BehaviorCode missing.")
        return None, None, None, None
        
    df['BehaviorCode'] = pd.to_numeric(df['BehaviorCode'], errors='coerce')
    
    # Keep only Class 0 (non-fidgeting) and Class 1 (fidgeting)
    df = df[df['BehaviorCode'].isin([0, 1])]
    
    # Needs participantId and activityID
    if 'participantId' not in df.columns or 'activityID' not in df.columns:
        print("Missing participantId or activityID.")
        return None, None, None, None

    # Keeping all participants (including those without Class 1 behaviors) as requested
    valid_participants = df['participantId'].unique()
    print(f"Total participants included: {len(valid_participants)}")

    y = df['BehaviorCode'].astype(int)
    groups = df['participantId']
    
    drop_bases = ['type', 'filename', 'participantId', 'activityID', 'BehaviorCode', 'FeatureName', 'WindowNumber', 'Unnamed']
    cols_to_drop = [c for c in df.columns if any(base in c for base in drop_bases)]
    X = df.drop(columns=cols_to_drop)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, y, groups, df['activityID']

def standardize_participant_activity(X, groups, activities):
    """
    Standardize based strictly on participant + activity combinations.
    """
    print("Applying participant + activity wise standardization...")
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

def build_models():
    # 80/20 train/test split for inner cross validation (Grid search)
    cv_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    return {
        "RF": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            "params": {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [10, 20, None],
                'clf__min_samples_split': [2, 5, 10]
            },
            "cv": cv_strategy
        },
        "DT": {
            "model": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            "params": {
                'clf__max_depth': [5, 10, 20, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
            },
            "cv": cv_strategy
        },
        "GB": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [3, 5, 10],
                'clf__learning_rate': [0.01, 0.1, 0.2]
            },
            "cv": cv_strategy
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42, class_weight='balanced'),
            "params": {
                'clf__C': [0.1, 1, 10],
                'clf__kernel': ['rbf', 'linear'],
                'clf__gamma': ['scale', 'auto']
            },
            "cv": cv_strategy
        }
    }

def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fidgeting', 'Fidgeting'], yticklabels=['Non-Fidgeting', 'Fidgeting'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    print(f"      -> Confusion Matrix saved to: {path}")
    plt.close()

def main():
    print("Initializing Stratified K-Fold GridSearch Pipeline...")
    model_configs = build_models()
    
    global_summary = []
    
    for fs_name, fs_filename in DATASETS.items():
        filepath = os.path.join(DATA_DIR, fs_filename)
        print(f"\n{'='*50}\nStarting Dataset: {fs_name}\n{'='*50}")
        
        # Folder for this feature set
        fs_out_dir = os.path.join(OUT_DIR, fs_name)
        os.makedirs(fs_out_dir, exist_ok=True)
        
        X, y, groups, activities = load_and_preprocess(filepath)
        
        if X is None or len(X) == 0:
            print(f"Skipping {fs_name} due to lack of valid data.")
            continue
            
        # Standardize Custom Participant+Activity Wise (local normalization)
        X_scaled = standardize_participant_activity(X, groups, activities)
        
        # Stratified Group K-Fold across the entire dataset (Fixes Data Leakage)
        n_splits = 5
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Load existing Global Summary for this Feature Set if it exists
        summary_path = os.path.join(OUT_DIR, f"comprehensive_summary_metrics_{fs_name}.csv")
        if os.path.exists(summary_path):
            print(f"Loading existing summary from: {summary_path}")
            global_summary = pd.read_csv(summary_path).to_dict('records')
        else:
            global_summary = []

        for m_name, m_info in model_configs.items():
            # Dynamic Completion Check
            details_path = os.path.join(fs_out_dir, f"{m_name}_fold_summary.csv")
            if os.path.exists(details_path):
                print(f"\n[SKIP] Model {m_name} already completed for {fs_name}. (Found: {details_path})")
                continue

            print(f"\n--- Running Model: {m_name} ---")
            
            y_true_out = []
            y_pred_out = []
            fold_details = []
            y_prob_out = []
            fold_idx = 1
            
            # Calculate dynamic class weights for XGBoost natively if evaluating GB
            if m_name == "GB":
                ratio = float(np.sum(y == 0)) / np.sum(y == 1) if np.sum(y == 1) > 0 else 1.0
                m_info['model'].set_params(scale_pos_weight=ratio)
                
            # Setup Mild Undersampling inside the pipeline (Layered approach for imbalance)
            pipeline = Pipeline([
                ('sampler', RandomUnderSampler(sampling_strategy=mild_under_strategy, random_state=42)),
                ('clf', m_info['model'])
            ])
            
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=m_info['params'],
                cv=m_info['cv'],
                scoring='balanced_accuracy',
                n_jobs=-1
            )
            print(f"Running {n_splits}-Fold Group Stratified CV (preventing leakage)...")

            for train_idx, test_idx in skf.split(X_scaled, y, groups=groups):
                X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Check target validness
                if len(np.unique(y_test)) == 0 or len(np.unique(y_train)) < 2:
                    continue
                
                print(f"      -> {m_name} | Training & Tuning for Fold: {fold_idx}/{n_splits} ...", end=" ", flush=True)

                # Grid Search runs internal 80/20 train/val split for the train-set to find best configurations
                grid_search.fit(X_train, y_train)
                
                # Predict on test fold
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                
                if hasattr(best_model, "predict_proba"):
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = [0.5]*len(y_test) # Fallback
                
                fold_bacc = balanced_accuracy_score(y_test, y_pred)
                fold_f1 = f1_score(y_test, y_pred)
                
                # specificty & sensitivity
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                fold_acc = accuracy_score(y_test, y_pred)
                fold_prec = precision_score(y_test, y_pred, zero_division=0)
                fold_rec = recall_score(y_test, y_pred, zero_division=0)
                
                try:
                    fold_roc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    fold_roc = float('nan')
                
                print(f"Done! | B-Acc: {fold_bacc:.2f} | F1: {fold_f1:.2f} | Sens: {sens:.2f} | ROC-AUC: {fold_roc:.2f}")

                y_true_out.extend(y_test)
                y_pred_out.extend(y_pred)
                y_prob_out.extend(y_prob)
                
                # Save Fold Details
                fold_details.append({
                    "Fold": fold_idx,
                    "Train Samples": len(X_train),
                    "Test Samples": len(X_test),
                    "Fold Accuracy": fold_acc,
                    "Fold Balanced Acc": fold_bacc,
                    "Fold F1 Score": fold_f1,
                    "Fold Precision": fold_prec,
                    "Fold Recall": fold_rec,
                    "Fold Sensitivity": sens,
                    "Fold Specificity": spec,
                    "Fold ROC-AUC": fold_roc,
                    "GridSearch Best Params": str(grid_search.best_params_)
                })

                # --- Save Per-Fold Results ---
                fold_out_dir = os.path.join(fs_out_dir, m_name, "fold_details", f"fold_{fold_idx}")
                os.makedirs(fold_out_dir, exist_ok=True)
                
                # Per-Fold CM
                fold_cm_path = os.path.join(fold_out_dir, f"fold_{fold_idx}_confusion_matrix.png")
                fold_title = f"{m_name} | {fs_name} | Fold: {fold_idx}"
                save_confusion_matrix(y_test, y_pred, fold_cm_path, fold_title)
                
                # Per-Fold Report
                fold_report_str = classification_report(y_test, y_pred, labels=[0, 1], target_names=["Non-Fidgeting", "Fidgeting"], zero_division=0)
                fold_report_path = os.path.join(fold_out_dir, f"fold_{fold_idx}_report.txt")
                with open(fold_report_path, "w") as f:
                    f.write(f"Fold: {fold_idx}\nFeature Set: {fs_name}\nModel: {m_name}\n\n")
                    f.write(fold_report_str)
                print(f"      -> Fold {fold_idx} details saved to: {fold_out_dir}")
                
                fold_idx += 1
                
            # Collect metrics
            if len(y_true_out) > 0:
                bacc = balanced_accuracy_score(y_true_out, y_pred_out)
                f1_g = f1_score(y_true_out, y_pred_out)
                
                cm_global = confusion_matrix(y_true_out, y_pred_out, labels=[0, 1])
                tn_g, fp_g, fn_g, tp_g = cm_global.ravel()
                spec_g = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0.0
                sens_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
                
                try:
                    roc_g = roc_auc_score(y_true_out, y_prob_out)
                except ValueError:
                    roc_g = float('nan')
                
                global_summary.append({
                    "Feature Set": fs_name,
                    "Model": m_name,
                    "Balanced Accuracy": bacc,
                    "F1 Score": f1_g,
                    "Sensitivity": sens_g,
                    "Specificity": spec_g,
                    "ROC AUC": roc_g
                })
                
                print(f"Result for {m_name} - B-Acc: {bacc:.4f} | F1: {f1_g:.4f} | Sens: {sens_g:.4f} | Spec: {spec_g:.4f} | ROC-AUC: {roc_g:.4f}")
                
                # Save Fold Summary CSV
                details_df = pd.DataFrame(fold_details)
                details_path = os.path.join(fs_out_dir, f"{m_name}_fold_summary.csv")
                details_df.to_csv(details_path, index=False)
                print(f"      -> Fold summary saved to: {details_path}")
                
                # Global Report
                report = classification_report(y_true_out, y_pred_out, labels=[0, 1], target_names=["Non-Fidgeting", "Fidgeting"], zero_division=0)
                report_path = os.path.join(fs_out_dir, f"{m_name}_skfold_report.txt")
                with open(report_path, "w") as f:
                    f.write(f"Dataset: {fs_name}\nModel: {m_name}\nStratified 5-Fold Cross Validation Used.\n\n")
                    f.write(report)
                print(f"      -> Classification report saved to: {report_path}")
                    
                # Confusion Matrix
                cm_path = os.path.join(fs_out_dir, f"{m_name}_skfold_cm.png")
                title = f"{m_name} | {fs_name} Features (SKFold)"
                save_confusion_matrix(y_true_out, y_pred_out, cm_path, title)

                # Incremental Save of Global Summary
                summary_df = pd.DataFrame(global_summary)
                summary_df.to_csv(summary_path, index=False)
                print(f"      -> {fs_name} summary metrics updated: {summary_path}")

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
