#!/usr/bin/env python3
"""
Neptune-tracked Random Forest classification pipeline with GridSearchCV.
To Run:

# 10-fold CV, 20% test split
python neptune_rf_pipeline.py \
  --csv_path /Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/feature_set/winsize_300_all_participants_hr_watch_features.csv \
  --target_col activity \
  --project "shehjar/CareWear" \
  --cv 10 \
  --test_size 0.2 \
  --random_state 42 \
  --run_name "RF-HR-Exp" \
  --tags HR,RF,baseline,window-300

"""

import argparse, json, os, time, pickle, hashlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import neptune


# === Legend ===
# Expanded ACTIVITY_LEGEND to include exact strings from CSV and new activities
ACTIVITY_LEGEND = {
    "Rest 1": 1, "Prepare Speech": 2, "Give Speech": 3,
    "Rest 2": 4, "Mental Math": 5, "Rest 3": 6,
    "Bike Legs": 7, "Bike Hand": 8,
    "Rest1": 1, "Rest2": 4, "Rest3": 6,
    "Prepare_Speech": 2, "Give_Speech": 3, "Mental_Math": 5,
    "Bike_Legs": 7, "Bike_Hand": 8,
    # Adding lowercase versions found in the CSV
    "prepare speech": 2,
    "give speech": 3,
    "mental math": 5,
    # Adding new activities from the CSV and assigning new numerical labels
    "stationary_Bike1": 7, # Reusing labels 7 and 8, assuming these map to "Bike Legs" and "Bike Hand" logically.
    "stationary_Bike2": 8, # If these are distinct activities, consider using 9 and 10 and updating labels list.
}

def map_activity(series: pd.Series) -> pd.Series:
    """
    Maps activity names to numerical labels. Handles non-finite values by filling them
    before converting to integer type.
    """
    if pd.api.types.is_numeric_dtype(series):
        # If the series is already numeric, fill NaNs and convert to int.
        # Using -1 as a placeholder for NaN activities.
        return series.fillna(-1).astype(int)

    # Convert to string, strip whitespace, and then map
    # This handles cases where activity names might have leading/trailing spaces
    # or are non-string types that could cause issues with .str.strip()
    clean_series = series.astype(str).str.strip()
    mapped = clean_series.map(ACTIVITY_LEGEND)
    # Fill any NaN values that result from mapping (e.g., if an activity is not in LEGEND)
    # before converting to integer.
    return mapped.fillna(-1).astype(int) # Using -1 as a placeholder for unmapped activities

def get_cv(cv_choice: str, random_state: int):
    if cv_choice == "5":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    if cv_choice == "10":
        return StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    if cv_choice == "15":
        return StratifiedKFold(n_splits=15, shuffle=True, random_state=random_state)
    if cv_choice.lower() in ["loo","loocv"]:
        return LeaveOneOut()
    raise ValueError("cv must be 5, 10, 15, or LOOCV")

def hash_features(cols: List[str]) -> str:
    return hashlib.md5("|".join(cols).encode()).hexdigest()

def plot_cm(y_true, y_pred, labels, path):
    # Ensure labels passed to confusion_matrix correspond to the actual y_true values
    # Filtering out -1 earlier should make sure y_true only contains 1-8.
    cm = confusion_matrix(y_true, y_pred, labels=range(1,len(labels)+1))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, xticks_rotation=45, cmap=None, values_format="d")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return cm

def plot_roc(y_true, y_proba, labels, model_classes, path):
    """
    Plots ROC curves for multi-class classification, handling cases where
    not all classes are present in the test set.

    Args:
        y_true (np.array): True labels of the test set.
        y_proba (np.array): Predicted probabilities for each class from the model.
                           Columns must correspond to model_classes.
        labels (list): List of all possible string display labels.
        model_classes (np.array): The classes the model was trained on,
                                  defining the order of columns in y_proba.
        path (str): File path to save the ROC plot.
    """
    # Get the unique numerical classes actually present in y_true (test set)
    actual_y_true_classes = np.unique(y_true).tolist()

    fig, ax = plt.subplots(figsize=(6,6))
    aucs = {}

    # Iterate through the classes that the model was trained on (model_classes)
    # as these dictate the order of columns in y_proba.
    for i, model_class_val in enumerate(model_classes):
        # Check if this model_class_val is actually present in y_true.
        # We only plot ROC for classes that exist in the true labels of the test set.
        if model_class_val in actual_y_true_classes:
            # Create a binary true label array for the current class (one-vs-rest)
            y_true_binary = (y_true == model_class_val).astype(int)

            # Select the corresponding probability column from y_proba
            y_proba_for_class = y_proba[:, i]

            # Find the display name for this class from the full labels list.
            display_name = ""
            # Iterate through ACTIVITY_LEGEND to find the string name corresponding to model_class_val
            for name_str, val in ACTIVITY_LEGEND.items():
                if val == model_class_val:
                    # Prefer the original case from labels list if available
                    if name_str in labels: # Check if this specific string name is in the ordered 'labels' list
                        display_name = name_str
                    else: # Fallback to any matching name from ACTIVITY_LEGEND
                        display_name = name_str
                    break
            if not display_name:
                display_name = f"Class {model_class_val}" # Fallback name

            # Plot ROC curve for the current class
            RocCurveDisplay.from_predictions(y_true_binary, y_proba_for_class, name=display_name, ax=ax)
            try:
                aucs[display_name] = roc_auc_score(y_true_binary, y_proba_for_class)
            except Exception:
                aucs[display_name] = np.nan # Log NaN if ROC AUC cannot be calculated for this class
        else:
            # If a class from model_classes is not in y_true, its ROC AUC is undefined for this test set.
            # We skip plotting for this class but acknowledge it might contribute to warnings.
            pass

    # Calculate macro/micro AUC scores using only classes that were actually present in y_true
    try:
        y_bin_all_present = label_binarize(y_true, classes=actual_y_true_classes)

        # Select columns from y_proba that correspond to classes present in actual_y_true_classes
        y_proba_relevant_cols = []
        for class_val in actual_y_true_classes:
            try:
                idx_in_model_classes = model_classes.tolist().index(class_val)
                y_proba_relevant_cols.append(y_proba[:, idx_in_model_classes])
            except ValueError:
                # This should not happen if actual_y_true_classes is a subset of model_classes
                pass
        if y_proba_relevant_cols: # Ensure there are columns to process
            y_proba_relevant = np.array(y_proba_relevant_cols).T # Transpose to get correct shape (samples, n_classes)
            macro = roc_auc_score(y_bin_all_present, y_proba_relevant, average="macro", multi_class="ovr")
            micro = roc_auc_score(y_bin_all_present, y_proba_relevant, average="micro", multi_class="ovr")
        else:
            macro, micro = np.nan, np.nan # No relevant probabilities to calculate AUC
    except Exception as e:
        print(f"Error calculating macro/micro ROC AUC: {e}")
        macro, micro = np.nan, np.nan

    ax.set_title(f"ROC macro={macro:.3f}, micro={micro:.3f}")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

    # Convert np.nan to None for Neptune logging, as Neptune might interpret np.nan as unsupported type
    # if it's not handled gracefully by your client version for direct dictionary assignment.
    # This explicitly converts numpy NaNs to Python's None type.
    for key, value in aucs.items():
        if pd.isna(value):
            aucs[key] = None
    if pd.isna(macro):
        macro = None
    if pd.isna(micro):
        micro = None

    return {"per_class": aucs,"macro":macro,"micro":micro}

def main():
    print("🚀 Script started: Initializing Neptune run and parsing arguments.") # Stage log
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--target_col", default="activity")
    ap.add_argument("--project", required=True)
    ap.add_argument("--cv", default="5")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--run_name"); ap.add_argument("--tags", default="")
    args = ap.parse_args()

    run = neptune.init_run(project=args.project, name=args.run_name,
                           tags=[t.strip() for t in args.tags.split(",") if t.strip()])

    print(f"📊 Stage: Data Loading from {args.csv_path}") # Stage log
    df = pd.read_csv(args.csv_path)
    print(f"DEBUG: Initial DataFrame shape: {df.shape}") # Debug log

    # Define the specific features to be used
    feature_cols = [
        'hr_mean', 'hr_median', 'hr_std', 'hr_min', 'hr_max', 'hr_iqr',
        'hr_skew', 'hr_kurtosis', 'hr_rmssd', 'hr_pnn50', 'hr_range', 'hr_slope'
    ]
    print(f"DEBUG: Selected feature columns: {feature_cols}") # Debug log

    print("🧩 Stage: Activity Mapping and Filtering") # Stage log
    # Map target column to numerical values, handling potential NaNs
    df['mapped_activity'] = map_activity(df[args.target_col])
    print(f"DEBUG: DataFrame shape after activity mapping: {df.shape}") # Debug log
    print(f"DEBUG: Unique mapped activities before filtering: {df['mapped_activity'].unique()}") # Debug log

    # Filter out rows where the activity mapping resulted in -1
    df_filtered = df[df['mapped_activity'] != -1].copy()
    print(f"DEBUG: DataFrame shape after filtering invalid activities: {df_filtered.shape}") # Debug log

    if df_filtered.empty:
        print("🛑 Error: After filtering for valid activities, the DataFrame is empty. Cannot proceed with training.") # Stage log
        run.stop()
        return

    print("🔧 Stage: Feature and Target Preparation") # Stage log
    # Select only the specified feature columns from the filtered DataFrame
    X = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')
    print(f"DEBUG: X (features) DataFrame shape after numeric conversion and before numpy conversion: {X.shape}") # Debug log
    X = X.to_numpy() # Convert X to numpy array
    print(f"DEBUG: X (features) NumPy array shape: {X.shape}") # Debug log


    # Use the mapped and filtered activity column as the target variable
    y = df_filtered['mapped_activity'].values
    print(f"DEBUG: y (target) NumPy array shape: {y.shape}") # Debug log
    print(f"DEBUG: Unique values in y (target): {np.unique(y)}") # Debug log


    run["features/hash"] = hash_features(feature_cols) # Use the explicit feature list for hashing
    run["features/count"] = len(feature_cols)

    print("📊 Stage: Train-Test Splitting") # Stage log
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=args.test_size,
                                       stratify=y,random_state=args.random_state)
    print(f"DEBUG: Xtr shape: {Xtr.shape}, Xte shape: {Xte.shape}") # Debug log
    print(f"DEBUG: ytr shape: {ytr.shape}, yte shape: {yte.shape}") # Debug log
    print(f"DEBUG: Unique classes in ytr: {np.unique(ytr)}")
    print(f"DEBUG: Unique classes in yte: {np.unique(yte)}")


    print("⚙️ Stage: Defining Pipeline and Grid Search Setup") # Stage log
    # Define pipeline
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(random_state=args.random_state,n_jobs=-1))])

    # Grid search setup
    param_grid = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2]
    }
    cv_split = get_cv(args.cv, args.random_state)
    gs = GridSearchCV(pipe, param_grid, cv=cv_split, scoring="accuracy", n_jobs=-1, verbose=1)

    print("⏳ Stage: Model Training (GridSearchCV)") # Stage log
    t0 = time.time(); gs.fit(Xtr,ytr); elapsed = time.time()-t0
    print(f"DEBUG: Grid search training time: {elapsed:.2f} seconds") # Debug log
    run["train/gridsearch_time"] = elapsed
    run["cv/best_score"] = gs.best_score_
    run["cv/best_params"] = gs.best_params_ # Corrected: Directly assign dictionary

    print(f"DEBUG: Best parameters found by GridSearchCV: {gs.best_params_}") # Debug log
    print(f"DEBUG: Best cross-validation score: {gs.best_score_}") # Debug log


    print("💾 Stage: Saving Grid Search Results") # Stage log
    # Save full grid results
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv("grid_results.csv", index=False)
    run["cv/grid_results"].upload("grid_results.csv")
    print("DEBUG: Grid search results saved to grid_results.csv and uploaded to Neptune.") # Debug log

    print("✨ Stage: Model Evaluation") # Stage log
    # Evaluate best model
    best_model = gs.best_estimator_
    ypred = best_model.predict(Xte)
    print(f"DEBUG: Test set predictions generated. Shape of ypred: {ypred.shape}") # Debug log

    run["test/accuracy"] = accuracy_score(yte,ypred)
    run["test/precision_macro"] = precision_score(yte,ypred,average="macro",zero_division=0)
    run["test/recall_macro"] = recall_score(yte,ypred,average="macro",zero_division=0)
    run["test/f1_macro"] = f1_score(yte,ypred,average="macro",zero_division=0)
    print(f"DEBUG: Test accuracy: {run['test/accuracy'].fetch()}") # Debug log
    print(f"DEBUG: Test precision (macro): {run['test/precision_macro'].fetch()}") # Debug log

    print("🖼️ Stage: Plotting Confusion Matrix") # Stage log
    # Confusion matrix
    # Labels must match the numerical values found in y_true after mapping.
    # Updated to include new stationary bike activities in the display labels.
    labels = ["Rest 1","Prepare Speech","Give Speech","Rest 2","Mental Math","Rest 3","Bike Legs","Bike Hand", "Stationary Bike 1", "Stationary Bike 2"]
    cm = plot_cm(yte, ypred, labels, "cm.png")
    run["plots/confusion_matrix"].upload("cm.png")
    print("DEBUG: Confusion matrix plotted and uploaded to Neptune.") # Debug log

    print("📈 Stage: Plotting ROC Curves") # Stage log
    # ROC
    if hasattr(best_model.named_steps["clf"],"predict_proba"):
        yprob = best_model.predict_proba(Xte)
        print(f"DEBUG: yprob (predicted probabilities) shape: {yprob.shape}") # Debug log
        print(f"DEBUG: Model classes: {best_model.named_steps['clf'].classes_}") # Debug log
        # Pass the model's classes_ to plot_roc to correctly align y_proba columns
        roc = plot_roc(yte, yprob, labels, best_model.named_steps["clf"].classes_, "roc.png")
        run["plots/roc"].upload("roc.png")
        # Corrected: Directly assign the dictionary for ROC metrics
        run["metrics/test/roc"] = roc
        print("DEBUG: ROC curves plotted and uploaded to Neptune.") # Debug log
    else:
        print("DEBUG: Classifier does not have 'predict_proba' method. Skipping ROC plot.") # Debug log

    print("📦 Stage: Saving Model Artifact") # Stage log
    # Save model
    with open("rf_best.pkl","wb") as f: pickle.dump(best_model,f)
    run["artifacts/model"].upload("rf_best.pkl")
    print("DEBUG: Best model saved as rf_best.pkl and uploaded to Neptune.") # Debug log

    print("\n--- Final Results ---")
    print("Best params:", gs.best_params_)
    print("Test accuracy:", accuracy_score(yte,ypred))
    print("✅ Script finished successfully!") # Final stage log

if __name__=="__main__":
    main()
