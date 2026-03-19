import os
import pandas as pd
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, simpledialog, messagebox
from ml_models_modular import run_lopo_pipeline

def start_experiment():
    root = tk.Tk()
    root.withdraw()

    # 1. Select File
    file_path = filedialog.askopenfilename(title="Select Feature CSV")
    if not file_path: 
        print("[CANCELLED] No file selected.")
        return
    
    p = Path(file_path).resolve()
    source_dir = p.parent
    base_filename = p.stem
    
    print(f"\n" + "="*60)
    print(f"[INFO] Source File: {p}")
    print(f"[INFO] Target Directory: {source_dir}")
    print("="*60)
    
    # PRE-FLIGHT CHECK
    test_file_path = source_dir / "write_permission_test.txt"
    try:
        test_file_path.write_text("Permission Test")
        test_file_path.unlink()
    except Exception as e:
        print(f"[CRITICAL] Permission Denied on {source_dir}")
        messagebox.showerror("Disk Error", f"Cannot write to the selected folder.\nError: {e}")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        messagebox.showerror("Error", f"Could not read CSV: {e}")
        return

    # 2. Setup Columns
    id_col = simpledialog.askstring("Setup", "Participant ID column:", initialvalue="Participant")
    target_col = simpledialog.askstring("Setup", "Target Class column:", initialvalue="Activity_Int")
    
    if id_col not in df.columns or target_col not in df.columns:
        messagebox.showerror("Error", f"Column '{id_col}' or '{target_col}' not found in CSV!")
        return

    # 3. Feature Input Handling (Robust Cleaning)
    messagebox.showinfo(
        "Feature Selection Important Note",
        "If you are selecting explicit features, ensure they were NOT selected using statistical techniques "
        "(like ANOVA/Correlation) on the full dataset, as this causes Data Leakage. \n\n"
        "Manual domain-knowledge feature selection prior to this experiment is acceptable."
    )
    include_str = simpledialog.askstring("Setup", "Paste features (comma separated):")
    cols_to_drop = []
    
    if include_str:
        selected = [f.strip().replace("'", "").replace('"', "") for f in include_str.split(',')]
        valid_selected = [f for f in selected if f in df.columns]
        
        missing = set(selected) - set(valid_selected)
        if missing:
            print(f"[WARNING] Some requested features were not found in CSV: {missing}")
            
        if not valid_selected:
            messagebox.showerror("Error", "No matching features found in the dataset!")
            return
            
        cols_to_drop = [c for c in df.columns if c not in valid_selected and c not in [id_col, target_col]]
        print(f"[INFO] Using {len(valid_selected)} features for classification.")
    else:
        drop_str = simpledialog.askstring("Setup", "Metadata to IGNORE:", initialvalue="FileName,Fs,StartTime")
        cols_to_drop = [c.strip() for c in drop_str.split(',')] if drop_str else []
        print("[INFO] No specific features selected; using default exclusion list.")

    # 4. Model Parameters Configuration
    sampling = simpledialog.askstring("Input", "Sampling (Rus or Cc):", initialvalue="Rus")
    scaling = simpledialog.askstring("Input", "Scaling (MinMax or Standard):", initialvalue="MinMax")

    # Target Mapping Setup
    stress_mapping = {1:0, 4:0, 6:0, 2:1, 3:1, 5:1, 7:2, 8:2}

    # Which Models to run
    available_models = ["RF", "GB", "XGB", "LR", "SVM", "DT"]
    models_str = simpledialog.askstring(
        "Input", 
        "Select Models to run (comma separated):\nAvailable: RF, GB, XGB, LR, SVM, DT", 
        initialvalue="RF, GB, XGB, LR, SVM, DT"
    )
    
    if not models_str:
        print("[CANCELLED] No models selected.")
        return
        
    models_to_run = [m.strip().upper() for m in models_str.split(',') if m.strip().upper() in available_models]
    
    if not models_to_run:
        messagebox.showerror("Error", "No valid models selected from the pool!")
        return

    print(f"\n[PROCESS] Starting Multi-Model LOPO pipeline for {base_filename}...")
    print(f"[CONFIG] Sampling: {sampling} | Scaling: {scaling}")
    print(f"[MODELS] {', '.join(models_to_run)}")
    
    success_models = []
    
    # 5. Execute Loop over Models
    for model_name in models_to_run:
        try:
            # Output directory format: source_dir / ModelName / base_filename
            model_results_dir = source_dir / model_name / base_filename
            model_results_dir = str(model_results_dir)
            
            print(f"\n[EXECUTION] -> Training {model_name}")
            
            results, summary = run_lopo_pipeline(
                df=df, 
                id_col=id_col, 
                target_col=target_col,
                model_name=model_name,
                target_mapping=stress_mapping, 
                undersample_method=sampling,
                scaling_method=scaling, 
                cols_to_drop=cols_to_drop,
                save_dir=model_results_dir,
                file_prefix=base_filename
            )
            success_models.append(model_name)
            
        except Exception as e:
            print("\n" + "!"*60)
            print(f"CRITICAL PIPELINE ERROR in {model_name}")
            traceback.print_exc()
            print("!"*60)
            
    if success_models:
        print(f"\n[SUCCESS] Benchmarks finished for: {', '.join(success_models)}")
        messagebox.showinfo(
            "Complete", 
            f"Benchmarking Finished.\nModels completed: {', '.join(success_models)}\n\nFiles saved in their respective directories (RF, GB, etc.) within:\n{source_dir}"
        )
    else:
        messagebox.showerror("Failed", "No models completed successfully. Check terminal for traces.")

if __name__ == "__main__":
    start_experiment()
