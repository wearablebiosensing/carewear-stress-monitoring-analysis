import os
import sys
import pandas as pd
from pathlib import Path
from ml_models_modular import run_lopo_pipeline
import traceback
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# 1. SCIENTIFIC FEATURE SETS DEFINITION
FEATURE_SETS = {
    "1_TimeDomain_Stats": [
        'hr_mean', 'hr_median', 'hr_std', 'hr_min', 'hr_max', 
        'hr_iqr', 'hr_skew', 'hr_kurtosis', 'hr_range'
    ],
    "2_HRV_Proxies": [
        'hr_rmssd', 'hr_pnn50', 'hr_sdsd', 'hr_sampen'
    ],
    "3_Temporal_Dynamics": [
        'hr_slope', 'hr_second_derivative', 'hr_macd_mean', 'hr_start', 'hr_end'
    ],
    "4_Contextual_Baseline": [
        'hr_perc', 'hr_recovery_window_1', 'hr_recovery_window_2'
    ],
    "5_HR_Zones": [
        'HR Range: 0–40 bpm', 'HR Range: 40–60 bpm', 'HR Range: 60–80 bpm',
        'HR Range: 80–100 bpm', 'HR Range: 100–120 bpm', 'HR Range: 120–140 bpm',
        'HR Range: 140–160 bpm', 'HR Range: 160–180 bpm', 'HR Range: 180–200 bpm',
        'HR Range: >200 bpm'
    ]
}

def main():
    root = tk.Tk()
    root.withdraw()
    
    file_path_str = filedialog.askopenfilename(title="Select Extracted Feature CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not file_path_str:
        print("[CANCELLED] No CSV file selected.")
        sys.exit(0)
        
    file_path = Path(file_path_str).resolve()
    base_filename = file_path.stem
    target_dir = file_path.parent
    results_dir_name = f"Auto_Experiments_Results_{base_filename}"
    
    models_str = simpledialog.askstring(
        "Models Setup", 
        "Select Models to run (comma separated):\nAvailable: RF, XGB, GB, LR, SVM, DT", 
        initialvalue="RF, XGB, GB, LR, SVM, DT"
    )
    if not models_str:
        print("[CANCELLED] No models selected.")
        sys.exit(0)

    models_to_run = [m.strip().upper() for m in models_str.split(',')]

    sampling = simpledialog.askstring("Input", "Sampling (Rus or Cc):", initialvalue="Rus")
    if not sampling: sampling = "Rus"

    scaling = simpledialog.askstring("Input", "Scaling (MinMax or Standard):", initialvalue="MinMax")
    if not scaling: scaling = "MinMax"

    print("="*60)
    print(f"[INFO] Auto-Experiment Runner Initialized")
    print(f"[INFO] Dataset: {file_path}")
    print(f"[INFO] Output Folder: {target_dir}")
    print(f"[INFO] Models: {', '.join(models_to_run)}")
    print(f"[CONFIG] Sampling: {sampling} | Scaling: {scaling}")
    print("="*60)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read CSV: {e}")
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(1)

    id_col = "Participant"
    target_col = "Activity_Int"
    
    if id_col not in df.columns or target_col not in df.columns:
        print(f"[ERROR] Missing required columns: '{id_col}' or '{target_col}'.")
        sys.exit(1)

    # dynamically add ALL combined set
    all_features = set()
    for feat_list in FEATURE_SETS.values():
        all_features.update(feat_list)
    FEATURE_SETS["6_All_Combined"] = list(all_features)

    stress_mapping = {1:0, 4:0, 6:0, 2:1, 3:1, 5:1, 7:2, 8:2}

    print("\nStarting Pipeline across 6 Feature Configurations...\n")
    all_time_results = []

    for set_name, selected_features in FEATURE_SETS.items():
        print("*"*60)
        print(f" EXPERIMENTING ON FEATURE SET: {set_name}")
        print("*"*60)

        valid_selected = [f for f in selected_features if f in df.columns]
        missing = set(selected_features) - set(valid_selected)
        
        if missing:
            print(f"[WARNING] Features missing in dataset: {missing}")
            
        if len(valid_selected) == 0:
            print(f"[SKIP] No valid features found for set {set_name}. Skipping...")
            continue
            
        print(f"[INFO] Evaluating {len(valid_selected)} features.")

        # Compute columns to drop (all columns in DF not in valid_selected and not reserved)
        cols_to_drop = [
            c for c in df.columns 
            if c not in valid_selected and c not in [id_col, target_col]
        ]
        
        for model_name in models_to_run:
            model_results_dir = target_dir / results_dir_name / set_name / model_name
            model_results_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = model_results_dir / f"{base_filename}_{model_name}_summary.json"
            if summary_path.exists():
                print(f"\n[SKIP] Results already exist for Model: {model_name} | Set: {set_name}")
                incremental_csv = model_results_dir / f"{base_filename}_{model_name}_fold_results.csv"
                if incremental_csv.exists():
                    df_prev = pd.read_csv(incremental_csv)
                    if 'training_time' in df_prev.columns:
                        df_prev['model'] = model_name
                        df_prev['feature_set'] = set_name
                        all_time_results.append(df_prev[['feature_set', 'model', 'participant', 'training_time']])
                continue
            
            print(f"\n[RUNNING] Model: {model_name} | Set: {set_name}")
            try:
                results_df, summary = run_lopo_pipeline(
                    df=df,
                    id_col=id_col,
                    target_col=target_col,
                    model_name=model_name,
                    target_mapping=stress_mapping,
                    undersample_method=sampling,
                    scaling_method=scaling,
                    cols_to_drop=cols_to_drop,
                    save_dir=str(model_results_dir),
                    file_prefix=base_filename
                )
                if 'training_time' in results_df.columns:
                    results_df['model'] = model_name
                    results_df['feature_set'] = set_name
                    all_time_results.append(results_df[['feature_set', 'model', 'participant', 'training_time']])
            except Exception as e:
                print(f"[CRITICAL ERROR] in {model_name} on {set_name}: {e}")
                traceback.print_exc()
                
    print("\n" + "="*60)
    print("ALL COMBINATIONS EXPLORED AND FINALIZED.")
    print(f"Results are nested within: {target_dir / results_dir_name}")
    print("="*60)
    
    if all_time_results:
        summary_time_df = pd.concat(all_time_results, ignore_index=True)
        time_summary_path = target_dir / results_dir_name / f"{base_filename}_training_times_summary.csv"
        summary_time_df.to_csv(time_summary_path, index=False)
        print(f"[INFO] Saved Unified Training Time Summary to: {time_summary_path}")

    messagebox.showinfo(
        "Complete", 
        f"Automated Feature Sets Benchmarking Finished.\n\nFiles saved in:\n{target_dir / results_dir_name}"
    )

if __name__ == "__main__":
    main()
