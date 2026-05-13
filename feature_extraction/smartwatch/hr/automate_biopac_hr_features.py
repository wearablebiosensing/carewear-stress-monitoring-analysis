import os
import glob
import pandas as pd
import numpy as np
import argparse
from hr_feature_extractor_sample_windowing import extract_features, generate_activity_summary

def process_biopac_folder(input_folder, fs, window_sizes, overlap_pct=50):
    files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not files:
        print(f"No CSV files found in {input_folder}")
        return

    output_folder = os.path.join(input_folder, "extracted_features")
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing {len(files)} files in {input_folder} with Fs={fs}Hz")

    # Pass 1: Determine Rest Baselines (activity_int_merged == 1)
    participant_rest_vals = {}
    participant_rest_hr = {}

    for file_path in files:
        file_name = os.path.basename(file_path)
        if "label_mapping" in file_name:
            continue
        try:
            df = pd.read_csv(file_path, usecols=['Heart Rate', 'activity_int_merged'])
            p_id = file_name.split("_biopac_")[1].split("_")[0] if "_biopac_" in file_name else "Unknown"

            rest_df = df[df['activity_int_merged'] == 1]
            if not rest_df.empty:
                if p_id not in participant_rest_vals:
                    participant_rest_vals[p_id] = []
                participant_rest_vals[p_id].extend(rest_df['Heart Rate'].dropna().tolist())
        except Exception as e:
            pass

    for p_id, vals in participant_rest_vals.items():
        if len(vals) > 0:
            participant_rest_hr[p_id] = np.nanmean(vals)
            print(f"Participant {p_id} Rest HR: {participant_rest_hr[p_id]:.2f} bpm")

    # Pass 2: Extract features for each window size
    for window_sec in window_sizes:
        print(f"\n--- Extracting features for {window_sec}s window ---")
        all_features_rows = []
        window_size_samples = int(window_sec * fs)
        step_samples = max(1, int(window_size_samples * (1 - overlap_pct / 100)))

        for file_path in files:
            file_name = os.path.basename(file_path)
            if "label_mapping" in file_name:
                continue

            try:
                # Need Heart Rate and activity_int_merged
                df = pd.read_csv(file_path)
                if 'Heart Rate' not in df.columns:
                    continue
                
                df['HeartRate'] = pd.to_numeric(df['Heart Rate'], errors='coerce')
                df = df.dropna(subset=['HeartRate']).reset_index(drop=True)
                if len(df) < window_size_samples:
                    continue

                p_id = file_name.split("_biopac_")[1].split("_")[0] if "_biopac_" in file_name else "Unknown"
                hr_rest = participant_rest_hr.get(p_id, np.nan)
                if np.isnan(hr_rest):
                    hr_rest = df['HeartRate'].quantile(0.10)

                # MACD Setup
                span_fast = max(1, int(30 * fs))
                span_slow = max(1, int(120 * fs))
                df['MACD'] = df['HeartRate'].ewm(span=span_fast, adjust=False).mean() - df['HeartRate'].ewm(span=span_slow, adjust=False).mean()

                prev_means = []
                windows_processed = 0

                for start in range(0, len(df) - window_size_samples + 1, step_samples):
                    window_df = df.iloc[start : start + window_size_samples]
                    
                    macd_window = window_df['MACD'].values if 'MACD' in window_df.columns else None
                    features = extract_features(window_df['HeartRate'].values, hr_rest=hr_rest, macd_series=macd_window, prev_means=prev_means)
                    
                    if np.isnan(features.get('hr_mean', np.nan)): continue
                    
                    prev_means.append(features['hr_mean'])
                    if len(prev_means) > 2:
                        prev_means.pop(0)

                    act_int = 0
                    if 'activity_int_merged' in window_df.columns:
                        mode_res = window_df['activity_int_merged'].mode()
                        if not mode_res.empty: act_int = mode_res.iloc[0]

                    features.update({
                        "FileName": file_name,
                        "Participant": p_id,
                        "Activity_Int": act_int,
                        "Fs": fs,
                        "StartTime": start / fs # seconds from start instead of timestamp
                    })
                    all_features_rows.append(features)
                    windows_processed += 1

                if windows_processed > 0:
                    print(f"  [OK] {file_name}: {windows_processed} windows.")

            except Exception as e:
                print(f"  [ERROR] {file_name}: {e}")

        if all_features_rows:
            output_df = pd.DataFrame(all_features_rows)
            out_csv = os.path.join(output_folder, f"BioPac_Features_{window_sec}s_{overlap_pct}.csv")
            output_df.to_csv(out_csv, index=False)
            
            summary_csv = os.path.join(output_folder, f"BioPac_Activity_Summary_{window_sec}s_{overlap_pct}.csv")
            generate_activity_summary(output_df, summary_csv)
            
            print(f"Saved {window_sec}s features -> {out_csv}")
        else:
            print(f"No valid windows for {window_sec}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate BioPac HR Feature Extraction for multiple window sizes")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing BioPac CSV chunks")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling frequency (Hz) of BioPac data")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap percentage")
    args = parser.parse_args()

    window_sizes = [2, 5, 10, 30, 60]
    process_biopac_folder(args.input, args.fs, window_sizes, args.overlap)
