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
import shap # Import SHAP library for feature importance analysis

# === Legend ===
# Expanded ACTIVITY_LEGEND to include exact strings from CSV and new activities
ACTIVITY_LEGEND = {
    'rest1':1,
'prepare speech':2,
'give speech':2,
'rest2':1,
'mental math':5,
'rest3':1,
'stationary_Bike1':7,
'stationary_Bike2':8,
'Stationary_Bike':8, # Ensure this is correct if it's a distinct activity or maps to an existing one
    "Rest 1": 1,
    "Prepare Speech": 2,
    "Give Speech": 2,
    "Rest 2": 1,
    "Mental Math": 5,
      "Rest 3": 1,
    "Bike Legs": 7,
      "Bike Hand": 8,
    "Rest1": 1,
    "Rest2": 1,
    "Rest3": 1,
    "Prepare_Speech": 2,
    "Give_Speech": 2,
      "Mental_Math": 5,
    "Bike_Legs": 7,
      "Bike_Hand": 8,
    # Adding lowercase versions found in the CSV
    "prepare speech": 2,
    "give speech": 2,
    "mental math": 5,
    # Assigning stationary bikes to existing Bike Legs (7) and Bike Hand (8) labels as per user's clarification
    "stationary_Bike1": 7,
    "stationary_Bike2": 8,
}

# Define a consistent map for display labels in plots (numerical ID to human-readable string)
DISPLAY_LABEL_MAP = {
    1: "Rest", # Consolidated Rest 1, 2, 3 into "Rest"
    2: "Speech", # Consolidated Prepare Speech, Give Speech into "Speech"
    5: "Mental Math",
    7: "Bike Legs",
    8: "Bike Hand",
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

def plot_cm(y_true, y_pred, path): # Removed `all_display_labels` argument, now uses DISPLAY_LABEL_MAP
    """
    Plots confusion matrix, dynamically determining active labels for calculation and display.
    """
    # Get unique numerical classes present in y_true and y_pred
    active_numerical_labels = np.unique(np.concatenate((y_true, y_pred))).tolist()
    active_numerical_labels.sort() # Ensure labels are sorted for consistent plotting

    # Get display names for only the active numerical labels using the global map
    display_labels_for_plot = [DISPLAY_LABEL_MAP.get(val, f"Class {val}") for val in active_numerical_labels]

    # Pass only the active numerical labels to confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=active_numerical_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels_for_plot) # Use the specific display labels
    fig, ax = plt.subplots(figsize=(8,8)) # Increased size for better readability
    disp.plot(ax=ax, xticks_rotation=45, cmap=plt.cm.Blues, values_format="d") # Use a cmap for better visuals
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return cm

def plot_roc(y_true, y_proba, model_classes, path): # Removed `labels` argument, now uses DISPLAY_LABEL_MAP
    """
    Plots ROC curves for multi-class classification, handling cases where
    not all classes are present in the test set.

    Args:
        y_true (np.array): True labels of the test set.
        y_proba (np.array): Predicted probabilities for each class from the model.
                           Columns must correspond to model_classes.
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

            # Get display name using the global map
            display_name = DISPLAY_LABEL_MAP.get(model_class_val, f"Class {model_class_val}")

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

    # Log dataset version (file tracking)
    print(f"📁 Stage: Tracking Dataset Version for {args.csv_path}") # Stage log
    run["dataset/csv_path"].track_files(args.csv_path)
    # If there were other related files or a directory, you could track them like this:
    # run["dataset/train_dataset_folder"].track_files("./datasets/train_data_folder")
    print(f"DEBUG: Dataset '{args.csv_path}' tracked to Neptune.")

    print(f"📊 Stage: Data Loading from {args.csv_path}") # Stage log
    df = pd.read_csv(args.csv_path)
    print(f"DEBUG: Initial DataFrame shape: {df.shape}") # Debug log

    # Define the specific features to be used
    feature_cols = [
        'hr_mean', 'hr_median', 'hr_std', 'hr_min', 'hr_max', 'hr_iqr',
        'hr_skew', 'hr_kurtosis', 'hr_rmssd', 'hr_pnn50', 'hr_range', 'hr_slope'
    ]
    print(f"DEBUG: Selected feature columns: {feature_cols}") # Debug log

    # Log feature names to Neptune
    print(f"📝 Stage: Tracking Feature Names") # Stage log
    run["features/feature_names"].log(feature_cols)
    print(f"DEBUG: Feature names logged to Neptune: {feature_cols}")


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

    test_accuracy_val = accuracy_score(yte,ypred)
    test_precision_macro_val = precision_score(yte,ypred,average="macro",zero_division=0)
    test_recall_macro_val = recall_score(yte,ypred,average="macro",zero_division=0)
    test_f1_macro_val = f1_score(yte,ypred,average="macro",zero_division=0)

    # Log training metrics
    ypred_tr = best_model.predict(Xtr)
    train_accuracy_val = accuracy_score(ytr, ypred_tr)
    train_precision_macro_val = precision_score(ytr, ypred_tr, average="macro", zero_division=0)
    train_recall_macro_val = recall_score(ytr, ypred_tr, average="macro", zero_division=0)
    train_f1_macro_val = f1_score(ytr, ypred_tr, average="macro", zero_division=0)

    run["test/accuracy"] = test_accuracy_val
    run["test/precision_macro"] = test_precision_macro_val
    run["test/recall_macro"] = test_recall_macro_val
    run["test/f1_macro"] = test_f1_macro_val

    run["train/accuracy"] = train_accuracy_val
    run["train/precision_macro"] = train_precision_macro_val
    run["train/recall_macro"] = train_recall_macro_val
    run["train/f1_macro"] = train_f1_macro_val


    # Add run.wait() to ensure all pending operations are synchronized before printing fetched values
    # or before the run finishes (though printing local variables is preferred for debug logs)
    run.wait()

    # FIX: Use the locally stored variables for printing, not fetching from run object
    print(f"DEBUG: Test accuracy: {test_accuracy_val}") # Debug log
    print(f"DEBUG: Test precision (macro): {test_precision_macro_val}") # Debug log
    print(f"DEBUG: Test recall (macro): {test_recall_macro_val}") # Debug log
    print(f"DEBUG: Test F1 (macro): {test_f1_macro_val}") # Debug log

    print(f"DEBUG: Train accuracy: {train_accuracy_val}") # Debug log
    print(f"DEBUG: Train precision (macro): {train_precision_macro_val}") # Debug log
    print(f"DEBUG: Train recall (macro): {train_recall_macro_val}") # Debug log
    print(f"DEBUG: Train F1 (macro): {train_f1_macro_val}") # Debug log


    print("🖼️ Stage: Plotting Confusion Matrix") # Stage log
    # Confusion matrix
    # The 'labels' list for display names should correspond to the numerical labels 1-8
    # as defined by DISPLAY_LABEL_MAP.
    cm = plot_cm(yte, ypred, "cm.png")
    run["plots/confusion_matrix"].upload("cm.png")
    print("DEBUG: Confusion matrix plotted and uploaded to Neptune.") # Debug log

    print("📈 Stage: Plotting ROC Curves") # Stage log
    # ROC
    if hasattr(best_model.named_steps["clf"],"predict_proba"):
        yprob = best_model.predict_proba(Xte)
        print(f"DEBUG: yprob (predicted probabilities) shape: {yprob.shape}") # Debug log
        print(f"DEBUG: Model classes: {best_model.named_steps['clf'].classes_}") # Debug log
        # Pass the model's classes_ to plot_roc to correctly align y_proba columns
        roc = plot_roc(yte, yprob, best_model.named_steps["clf"].classes_, "roc.png")
        run["plots/roc"].upload("roc.png")
        # Corrected: Directly assign the dictionary for ROC metrics
        run["metrics/test/roc"] = roc
        print("DEBUG: ROC curves plotted and uploaded to Neptune.") # Debug log
    else:
        print("DEBUG: Classifier does not have 'predict_proba' method. Skipping ROC plot.") # Debug log

    print("🧠 Stage: Feature Selection Analysis (SHAP & RF Importance)") # Stage log
    # --- SHAP Analysis for Feature Importance ---
    try:
        # We need the raw classifier from the pipeline for SHAP
        rf_classifier = best_model.named_steps["clf"]
        imputer = best_model.named_steps["imputer"]
        Xtr_imputed = imputer.transform(Xtr) # SHAP expects imputed data if imputer is used

        # Create a SHAP explainer
        explainer = shap.TreeExplainer(rf_classifier)
        shap_values = explainer.shap_values(Xtr_imputed)

        # For multi-class, shap_values is a list of arrays. Sum absolute SHAP values per feature.
        # Ensure we handle the case where shap_values might be a single array for binary.
        if isinstance(shap_values, list):
            # Sum absolute SHAP values across all classes for each feature
            # The structure is (n_classes, n_samples, n_features)
            shap_abs_sum = np.sum(np.abs(shap_values), axis=0)
            # Take mean across samples to get overall feature importance
            shap_mean_abs = np.mean(shap_abs_sum, axis=0)
        else:
            # For binary classification, shap_values is (n_samples, n_features)
            shap_mean_abs = np.mean(np.abs(shap_values), axis=0)

        shap_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP_Importance': shap_mean_abs
        }).sort_values(by='SHAP_Importance', ascending=False)

        shap_importance_df.to_csv("shap_importance.csv", index=False)
        run["features/feature_selection/shap_importance"].upload("shap_importance.csv")
        run["features/feature_selection/shap_importance_table"] = neptune.types.File.as_html(shap_importance_df)

        # --- Log SHAP Summary Plot ---
        plt.figure(figsize=(10, 6)) # Create a new figure for the plot
        if isinstance(shap_values, list):
            # For multi-class, use the list of shap_values
            shap.summary_plot(shap_values, Xtr_imputed, feature_names=feature_cols, show=False)
        else:
            # For binary, use the single shap_values array
            shap.summary_plot(shap_values, Xtr_imputed, feature_names=feature_cols, show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        shap_plot_path = "shap_summary_plot.png"
        plt.savefig(shap_plot_path, dpi=200)
        plt.close() # Close the plot to free up memory

        run["features/feature_selection/shap_summary_plot"].upload(shap_plot_path)
        print("DEBUG: SHAP summary plot generated and logged to Neptune.")

        print("DEBUG: SHAP feature importance calculated and logged to Neptune.")
    except Exception as e:
        print(f"⚠️ Warning: Could not perform SHAP analysis or generate plot. Error: {e}")
        run["features/feature_selection/shap_status"] = f"Failed: {e}"

    # --- Random Forest Feature Importance ---
    try:
        rf_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'RF_Importance': best_model.named_steps["clf"].feature_importances_
        }).sort_values(by='RF_Importance', ascending=False)

        rf_importance_df.to_csv("rf_importance.csv", index=False)
        run["features/feature_selection/rf_importance"].upload("rf_importance.csv")
        run["features/feature_selection/rf_importance_table"] = neptune.types.File.as_html(rf_importance_df)
        print("DEBUG: Random Forest built-in feature importance logged to Neptune.")
    except Exception as e:
        print(f"⚠️ Warning: Could not retrieve Random Forest feature importance. Error: {e}")
        run["features/feature_selection/rf_importance_status"] = f"Failed: {e}"


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
