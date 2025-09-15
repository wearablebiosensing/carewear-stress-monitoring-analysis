#!/usr/bin/env python3
"""
Neptune-tracked Random Forest classification pipeline with GridSearchCV.
To Run:

# 10-fold CV, 20% test split, with class mapping and selection
python neptune_rf_pipeline.py \
  --csv_path /path/to/your/data.csv \
  --target_col activity_int \
  --project "your/project" \
  --cv 10 \
  --test_size 0.2 \
  --random_state 42 \
  --run_name "RF-Binned-Selected-Classes-Exp" \
  --tags RF,binned,configurable,selected \
  --class_map '{"1": 101, "4": 101, "6": 101, "2": 102, "3": 102, "7": 100, "8": 100}' \
  --include_classes 100 101 102
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

def plot_cm(y_true, y_pred, path):
    """
    Plots confusion matrix, dynamically determining active labels for calculation and display.
    """
    active_numerical_labels = np.unique(np.concatenate((y_true, y_pred))).tolist()
    active_numerical_labels.sort()

    # Use numerical labels directly for display
    display_labels_for_plot = [str(val) for val in active_numerical_labels]

    cm = confusion_matrix(y_true, y_pred, labels=active_numerical_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels_for_plot)
    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, xticks_rotation=45, cmap=plt.cm.Blues, values_format="d")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return cm

def plot_roc(y_true, y_proba, model_classes, path):
    """
    Plots ROC curves for multi-class classification.
    """
    actual_y_true_classes = np.unique(y_true).tolist()
    fig, ax = plt.subplots(figsize=(6,6))
    aucs = {}

    for i, model_class_val in enumerate(model_classes):
        if model_class_val in actual_y_true_classes:
            y_true_binary = (y_true == model_class_val).astype(int)
            y_proba_for_class = y_proba[:, i]

            # Use numerical value as display name
            display_name = f"Class {model_class_val}"

            RocCurveDisplay.from_predictions(y_true_binary, y_proba_for_class, name=display_name, ax=ax)
            try:
                aucs[display_name] = roc_auc_score(y_true_binary, y_proba_for_class)
            except Exception:
                aucs[display_name] = np.nan
        else:
            pass

    try:
        # Binarize labels only for classes present in y_true
        if len(actual_y_true_classes) > 1:
            y_bin_all_present = label_binarize(y_true, classes=actual_y_true_classes)
            
            # Select probability columns corresponding to the classes in y_true
            y_proba_relevant_cols_indices = [list(model_classes).index(c) for c in actual_y_true_classes]
            y_proba_relevant = y_proba[:, y_proba_relevant_cols_indices]
            
            macro = roc_auc_score(y_bin_all_present, y_proba_relevant, average="macro", multi_class="ovr")
            micro = roc_auc_score(y_bin_all_present, y_proba_relevant, average="micro", multi_class="ovr")
        else: # Cannot calculate multiclass AUC with a single class
            macro, micro = np.nan, np.nan
    except Exception as e:
        print(f"Error calculating macro/micro ROC AUC: {e}")
        macro, micro = np.nan, np.nan

    ax.set_title(f"ROC macro={macro:.3f}, micro={micro:.3f}")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

    return {"per_class": aucs,"macro":macro,"micro":micro}
def main():
    print("🚀 Script started: Initializing Neptune run and parsing arguments.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--target_col", default="activity_int")
    ap.add_argument("--project", required=True)
    ap.add_argument("--cv", default="5")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--run_name"); ap.add_argument("--tags", default="")
    ap.add_argument("--class_map", type=str, default=None,
                        help="JSON string for class mapping. E.g., '{\"1\": 101, \"4\": 101, \"2\": 102}'")
    ap.add_argument("--include_classes", nargs='+', type=int, default=None,
                        help="List of class integers to include in the analysis. E.g., --include_classes 100 101 102")
    args = ap.parse_args()

    run = neptune.init_run(project=args.project, name=args.run_name,
                           tags=[t.strip() for t in args.tags.split(",") if t.strip()])
    
    run["parameters"] = vars(args)

    class_map = None
    if args.class_map:
        try:
            class_map = json.loads(args.class_map)
            class_map = {int(k): v for k, v in class_map.items()}
            print(f"DEBUG: Successfully parsed class map: {class_map}")
        except json.JSONDecodeError:
            print(f"🛑 Error: Invalid JSON string provided for --class_map: {args.class_map}")
            run.stop()
            return

    print(f"📁 Stage: Tracking Dataset Version for {args.csv_path}")
    run["dataset/csv_path"].track_files(args.csv_path)
    print(f"DEBUG: Dataset '{args.csv_path}' tracked to Neptune.")

    print(f"📊 Stage: Data Loading from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"DEBUG: Initial DataFrame shape: {df.shape}")

    feature_cols = [c for c in df.columns if 'accel' in c or 'hr' in c]
    print(f"DEBUG: Selected {len(feature_cols)} feature columns.")

    print(f"📝 Stage: Tracking Feature Names")
    run["features/feature_names"].log(feature_cols)
    print(f"DEBUG: Feature names logged to Neptune.")

    df.dropna(subset=[args.target_col], inplace=True)

    if class_map:
        print("🔄 Stage: Applying Class Mapping")
        print(f"DEBUG: Unique classes before remapping: {df[args.target_col].unique()}")
        df[args.target_col] = df[args.target_col].astype(int).replace(class_map)
        print(f"DEBUG: Unique classes after remapping: {df[args.target_col].unique()}")

    if args.include_classes:
        print("🔍 Stage: Filtering for Selected Classes")
        print(f"DEBUG: Filtering to include only these classes: {args.include_classes}")
        df = df[df[args.target_col].isin(args.include_classes)]
        print(f"DEBUG: DataFrame shape after filtering: {df.shape}")
        if df.empty:
            print("🛑 Error: DataFrame is empty after filtering for selected classes. No data to process.")
            run.stop()
            return
            
    print("🔧 Stage: Feature and Target Preparation")
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    print(f"DEBUG: X (features) DataFrame shape after numeric conversion: {X.shape}")
    X = X.to_numpy()
    print(f"DEBUG: X (features) NumPy array shape: {X.shape}")

    y = df[args.target_col].astype(int).values
    print(f"DEBUG: y (target) NumPy array shape: {y.shape}")
    print(f"DEBUG: Unique values in final y (target): {np.unique(y)}")

    run["features/hash"] = hash_features(feature_cols)
    run["features/count"] = len(feature_cols)

    print("📊 Stage: Train-Test Splitting")
    if len(np.unique(y)) > 1:
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=args.test_size,
                                           stratify=y,random_state=args.random_state)
    else:
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=args.test_size,
                                           random_state=args.random_state)

    print(f"DEBUG: Xtr shape: {Xtr.shape}, Xte shape: {Xte.shape}")
    print(f"DEBUG: ytr shape: {ytr.shape}, yte shape: {yte.shape}")
    print(f"DEBUG: Unique classes in ytr: {np.unique(ytr)}")
    print(f"DEBUG: Unique classes in yte: {np.unique(yte)}")

    print("⚙️ Stage: Defining Pipeline and Grid Search Setup")
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(random_state=args.random_state,n_jobs=-1))])

    # MODIFICATION IS HERE
    param_grid = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__class_weight": ["balanced", None] 
    }
    
    cv_split = get_cv(args.cv, args.random_state)
    gs = GridSearchCV(pipe, param_grid, cv=cv_split, scoring="accuracy", n_jobs=-1, verbose=1)

    print("⏳ Stage: Model Training (GridSearchCV)")
    t0 = time.time(); gs.fit(Xtr,ytr); elapsed = time.time()-t0
    print(f"DEBUG: Grid search training time: {elapsed:.2f} seconds")
    run["train/gridsearch_time"] = elapsed
    run["cv/best_score"] = gs.best_score_
    run["cv/best_params"] = gs.best_params_

    print(f"DEBUG: Best parameters found by GridSearchCV: {gs.best_params_}")
    print(f"DEBUG: Best cross-validation score: {gs.best_score_}")

    print("💾 Stage: Saving Grid Search Results")
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv("grid_results.csv", index=False)
    run["cv/grid_results"].upload("grid_results.csv")
    print("DEBUG: Grid search results saved to grid_results.csv and uploaded to Neptune.")

    print("✨ Stage: Model Evaluation")
    best_model = gs.best_estimator_
    ypred = best_model.predict(Xte)
    print(f"DEBUG: Test set predictions generated. Shape of ypred: {ypred.shape}")

    test_accuracy_val = accuracy_score(yte,ypred)
    test_precision_macro_val = precision_score(yte,ypred,average="macro",zero_division=0)
    test_recall_macro_val = recall_score(yte,ypred,average="macro",zero_division=0)
    test_f1_macro_val = f1_score(yte,ypred,average="macro",zero_division=0)

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

    run.wait()

    print(f"DEBUG: Test accuracy: {test_accuracy_val}")
    print(f"DEBUG: Test precision (macro): {test_precision_macro_val}")
    print(f"DEBUG: Test recall (macro): {test_recall_macro_val}")
    print(f"DEBUG: Test F1 (macro): {test_f1_macro_val}")

    print(f"DEBUG: Train accuracy: {train_accuracy_val}")
    print(f"DEBUG: Train precision (macro): {train_precision_macro_val}")
    print(f"DEBUG: Train recall (macro): {train_recall_macro_val}")
    print(f"DEBUG: Train F1 (macro): {train_f1_macro_val}")

    print("🖼️ Stage: Plotting Confusion Matrix")
    cm = plot_cm(yte, ypred, "cm.png")
    run["plots/confusion_matrix"].upload("cm.png")
    print("DEBUG: Confusion matrix plotted and uploaded to Neptune.")

    print("📈 Stage: Plotting ROC Curves")
    if hasattr(best_model.named_steps["clf"],"predict_proba"):
        yprob = best_model.predict_proba(Xte)
        print(f"DEBUG: yprob (predicted probabilities) shape: {yprob.shape}")
        print(f"DEBUG: Model classes: {best_model.named_steps['clf'].classes_}")
        roc = plot_roc(yte, yprob, best_model.named_steps["clf"].classes_, "roc.png")
        run["plots/roc"].upload("roc.png")
        run["metrics/test/roc"] = roc
        print("DEBUG: ROC curves plotted and uploaded to Neptune.")
    else:
        print("DEBUG: Classifier does not have 'predict_proba' method. Skipping ROC plot.")

    print("🧠 Stage: Feature Selection Analysis (SHAP & RF Importance)")
    try:
        rf_classifier = best_model.named_steps["clf"]
        imputer = best_model.named_steps["imputer"]
        Xtr_imputed = imputer.transform(Xtr)

        explainer = shap.TreeExplainer(rf_classifier)
        shap_values = explainer.shap_values(Xtr_imputed)

        if isinstance(shap_values, list):
            shap_abs_sum = np.sum(np.abs(shap_values), axis=0)
            shap_mean_abs = np.mean(shap_abs_sum, axis=0)
        else:
            shap_mean_abs = np.mean(np.abs(shap_values), axis=0)

        shap_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP_Importance': shap_mean_abs
        }).sort_values(by='SHAP_Importance', ascending=False)

        shap_importance_df.to_csv("shap_importance.csv", index=False)
        run["features/feature_selection/shap_importance"].upload("shap_importance.csv")
        run["features/feature_selection/shap_importance_table"] = neptune.types.File.as_html(shap_importance_df)
        
        plt.figure(figsize=(10, 8))
        top_20_features = shap_importance_df.head(20)
        plt.barh(top_20_features['Feature'], top_20_features['SHAP_Importance'])
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("SHAP Feature Importance (Top 20)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        shap_bar_plot_path = "shap_bar_plot.png"
        plt.savefig(shap_bar_plot_path, dpi=200)
        plt.close()
        
        run["plots/shap_bar_plot"].upload(shap_bar_plot_path)
        print("DEBUG: SHAP bar plot generated and logged to Neptune.")

        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, Xtr_imputed, feature_names=feature_cols, show=False)
        else:
            shap.summary_plot(shap_values, Xtr_imputed, feature_names=feature_cols, show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        shap_plot_path = "shap_summary_plot.png"
        plt.savefig(shap_plot_path, dpi=200)
        plt.close()

        run["plots/shap_summary_plot"].upload(shap_plot_path)
        print("DEBUG: SHAP summary plot generated and logged to Neptune.")

        print("DEBUG: SHAP feature importance calculated and logged to Neptune.")
    except Exception as e:
        print(f"⚠️ Warning: Could not perform SHAP analysis or generate plot. Error: {e}")
        run["features/feature_selection/shap_status"] = f"Failed: {e}"

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

    print("📦 Stage: Saving Model Artifact")
    with open("rf_best.pkl","wb") as f: pickle.dump(best_model,f)
    run["artifacts/model"].upload("rf_best.pkl")
    print("DEBUG: Best model saved as rf_best.pkl and uploaded to Neptune.")

    print("\n--- Final Results ---")
    print("Best params:", gs.best_params_)
    print("Test accuracy:", accuracy_score(yte,ypred))
    print("✅ Script finished successfully!")
    
if __name__=="__main__":
    main()