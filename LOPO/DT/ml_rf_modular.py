import os
import json
import numpy as np
import pandas as pd
import traceback
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    # Sensitivity (Recall) Macro
    sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Specificity Macro
    classes = cm.shape[0]
    specificities = []
    for i in range(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    specificity = np.mean(specificities) if classes > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "cm": cm
    }

def run_lopo_pipeline(
    df, id_col, target_col, cols_to_drop=None, target_mapping=None,
    undersample_method="Rus", scaling_method="MinMax",
    save_dir=".", file_prefix="results"
):
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    df = df.copy()
    if target_mapping:
        df['target'] = df[target_col].map(target_mapping)
        df = df.dropna(subset=['target'])
    else:
        df['target'] = df[target_col]

    internal_drops = {id_col, target_col, 'target'} | set(cols_to_drop or [])
    feature_cols = [c for c in df.columns if c not in internal_drops]
    feature_list_str = ", ".join(feature_cols)
    
    results = []
    pids = sorted(df[id_col].unique())
    print(f"--- Starting Pipeline: {len(feature_cols)} features, {len(pids)} participants ---")

    for pid in pids:
        print(f"Processing Participant: {pid}...", end=" ", flush=True)
        try:
            train_df, test_df = df[df[id_col] != pid], df[df[id_col] == pid]
            
            # Keep track of train participant groups for GroupKFold
            groups_train = train_df[id_col]
            
            X_train, y_train = train_df[feature_cols], train_df['target']
            X_test, y_test = test_df[feature_cols], test_df['target']

            if X_test.empty:
                print("Skipped (No data)")
                continue

            # Create an imbalanced-learn pipeline to prevent preprocessing/resampling leakage
            pipeline_steps = [
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler() if scaling_method == "Standard" else MinMaxScaler())
            ]
            
            if undersample_method == "Rus":
                pipeline_steps.append(('sampler', RandomUnderSampler(random_state=42)))
            elif undersample_method == "Cc":
                pipeline_steps.append(('sampler', ClusterCentroids(random_state=42)))
            
            # Replaced DecisionTreeClassifier with RandomForestClassifier
            pipeline_steps.append(('rf', RandomForestClassifier(class_weight="balanced", random_state=8)))
            pipeline = ImbPipeline(pipeline_steps)

            # Prevent Subject Overlap Leakage using GroupKFold
            n_groups = groups_train.nunique()
            n_splits = min(3, n_groups) if n_groups > 0 else 3
            inner_cv = GroupKFold(n_splits=n_splits) if n_splits >= 2 else 2

            # Updated GridSearchCV to use rf__ prefix parameters
            grid = GridSearchCV(
                pipeline,
                {'rf__max_depth': [10, 20, None], 'rf__min_samples_split': [2, 5]},
                cv=inner_cv,
                scoring="balanced_accuracy", n_jobs=-1
            )
            
            if isinstance(inner_cv, GroupKFold):
                grid.fit(X_train, y_train, groups=groups_train)
            else:
                grid.fit(X_train, y_train)
            
            y_pred = grid.best_estimator_.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            
            # 1. Build Row
            row = {
                "participant": pid,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": metrics['accuracy'],
                "balanced_accuracy": metrics['balanced_accuracy'],
                "f1_score": metrics['f1_score'],
                "sensitivity": metrics['sensitivity'],
                "specificity": metrics['specificity'],
                "features_used": feature_list_str
            }
            results.append(row)

            # 2. SAVE INDIVUDAL CM IMMEDIATELY
            cm_filename = os.path.join(save_dir, f"{file_prefix}_pid_{pid}_cm.csv")
            pd.DataFrame(metrics['cm']).to_csv(cm_filename, index=False)
            
            # 3. SAVE CUMULATIVE RESULTS IMMEDIATELY (Safety Write)
            incremental_csv = os.path.join(save_dir, f"{file_prefix}_fold_results.csv")
            pd.DataFrame(results).to_csv(incremental_csv, index=False)
            
            print(f"Done (Saved files to {os.path.basename(save_dir)})")
        except Exception as e:
            print(f"FAILED for PID {pid}: {e}")
            traceback.print_exc()

    # Final Summary Export
    print("\n--- Finalizing Summary ---")
    try:
        results_df = pd.DataFrame(results)
        summary = results_df.mean(numeric_only=True).to_dict()
        summary_path = os.path.join(save_dir, f"{file_prefix}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[SUCCESS] Summary JSON written: {summary_path}")
    except Exception as e:
        print(f"[ERROR] Final summary calculation failed: {e}")

    return pd.DataFrame(results), summary
