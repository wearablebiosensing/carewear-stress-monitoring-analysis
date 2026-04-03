import os
import json
import time
import numpy as np
import pandas as pd
import traceback
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, recall_score
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
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

def get_model_and_grid(model_name):
    # Returns (estimator, param_grid)
    if model_name == "RF":
        return RandomForestClassifier(class_weight="balanced", random_state=8), {'model__max_depth': [10, 20, None], 'model__min_samples_split': [2, 5]}
    elif model_name == "GB":
        return GradientBoostingClassifier(random_state=8), {'model__max_depth': [3, 5, 10], 'model__learning_rate': [0.01, 0.1]}
    elif model_name == "XGB":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed. Please pip install xgboost.")
        # eval_metric to suppress warnings
        return XGBClassifier(eval_metric='mlogloss', random_state=8), {'model__max_depth': [3, 5, 10], 'model__learning_rate': [0.01, 0.1]}
    elif model_name == "LR":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=8), {'model__C': [0.1, 1.0, 10.0]}
    elif model_name == "SVM":
        return SVC(class_weight="balanced", probability=True, random_state=8), {'model__C': [0.1, 1.0, 10.0], 'model__kernel': ['linear', 'rbf']}
    elif model_name == "DT":
        return DecisionTreeClassifier(class_weight="balanced", random_state=8), {'model__max_depth': [5, 10, None], 'model__min_samples_split': [2, 5]}
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def run_lopo_pipeline(
    df, id_col, target_col, model_name="RF", cols_to_drop=None, target_mapping=None,
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
    print(f"--- Starting {model_name} Pipeline: {len(feature_cols)} features, {len(pids)} participants ---")

    for pid in pids:
        print(f"Processing Participant: {pid}...", end=" ", flush=True)
        try:
            train_df, test_df = df[df[id_col] != pid], df[df[id_col] == pid]
            
            groups_train = train_df[id_col]
            
            X_train, y_train = train_df[feature_cols], train_df['target']
            X_test, y_test = test_df[feature_cols], test_df['target']

            if X_test.empty:
                print("Skipped (No data)")
                continue

            # Standard processing block strictly replicated from the Random Forest logic
            pipeline_steps = [
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler() if scaling_method == "Standard" else MinMaxScaler())
            ]
            
            if undersample_method == "Rus":
                pipeline_steps.append(('sampler', RandomUnderSampler(random_state=42)))
            elif undersample_method == "Cc":
                pipeline_steps.append(('sampler', ClusterCentroids(random_state=42)))
            
            # Fetch Model + Grid Configuration
            estimator, param_grid = get_model_and_grid(model_name)
            pipeline_steps.append(('model', estimator))
            pipeline = ImbPipeline(pipeline_steps)

            n_groups = groups_train.nunique()
            n_splits = min(3, n_groups) if n_groups > 0 else 3
            inner_cv = GroupKFold(n_splits=n_splits) if n_splits >= 2 else 2

            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=inner_cv,
                scoring="balanced_accuracy", n_jobs=-1
            )
            
            start_time = time.time()
            if isinstance(inner_cv, GroupKFold):
                grid.fit(X_train, y_train, groups=groups_train)
            else:
                grid.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = grid.best_estimator_.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            
            row = {
                "participant": pid,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": metrics['accuracy'],
                "balanced_accuracy": metrics['balanced_accuracy'],
                "f1_score": metrics['f1_score'],
                "sensitivity": metrics['sensitivity'],
                "specificity": metrics['specificity'],
                "training_time": train_time,
                "features_used": feature_list_str
            }
            results.append(row)

            # SAVE INDIVIDUAL CM IMMEDIATELY
            cm_filename = os.path.join(save_dir, f"{file_prefix}_{model_name}_pid_{pid}_cm.csv")
            pd.DataFrame(metrics['cm']).to_csv(cm_filename, index=False)
            
            # SAVE CUMULATIVE RESULTS IMMEDIATELY
            incremental_csv = os.path.join(save_dir, f"{file_prefix}_{model_name}_fold_results.csv")
            pd.DataFrame(results).to_csv(incremental_csv, index=False)
            
            print(f"Done")
        except Exception as e:
            print(f"FAILED for PID {pid}: {e}")
            traceback.print_exc()

    # Final Summary Export
    print(f"\n--- Finalizing Summary for {model_name} ---")
    try:
        results_df = pd.DataFrame(results)
        summary = results_df.mean(numeric_only=True).to_dict()
        summary_path = os.path.join(save_dir, f"{file_prefix}_{model_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[SUCCESS] {model_name} Summary JSON written: {summary_path}\n")
    except Exception as e:
        print(f"[ERROR] Final summary calculation failed for {model_name}: {e}\n")

    # Train and save final deployment model on all data
    print(f"--- Training Final Deployment Model ---")
    try:
        X_all, y_all = df[feature_cols], df['target']
        pipeline_steps = [
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler() if scaling_method == "Standard" else MinMaxScaler())
        ]
        
        if undersample_method == "Rus":
            pipeline_steps.append(('sampler', RandomUnderSampler(random_state=42)))
        elif undersample_method == "Cc":
            pipeline_steps.append(('sampler', ClusterCentroids(random_state=42)))
            
        estimator, param_grid = get_model_and_grid(model_name)
        pipeline_steps.append(('model', estimator))
        pipeline = ImbPipeline(pipeline_steps)
        
        n_groups = df[id_col].nunique()
        n_splits = min(3, n_groups) if n_groups > 0 else 3
        inner_cv = GroupKFold(n_splits=n_splits) if n_splits >= 2 else 2
        
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy", n_jobs=-1
        )
        
        if isinstance(inner_cv, GroupKFold):
            grid.fit(X_all, y_all, groups=df[id_col])
        else:
            grid.fit(X_all, y_all)
            
        full_model = grid.best_estimator_
        model_path = os.path.join(save_dir, f"{file_prefix}_{model_name}_deploy_model.pkl")
        joblib.dump(full_model, model_path)
        print(f"[SUCCESS] Final deploy model saved: {model_path}\n")
    except Exception as e:
        print(f"[ERROR] Final model training failed for {model_name}: {e}\n")
        traceback.print_exc()

    return pd.DataFrame(results), summary
