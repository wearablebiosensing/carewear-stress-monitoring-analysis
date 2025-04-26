import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# Step 1: Load ECG Data from Specific Sheets
def load_ecg_data(file_name):
    # Define the specific sheet names
    sheet_names = ['Rest 1', 'Prepare Speech', 'Give Speech', 'Rest 2',
                   'Mental Math', 'Rest 3', 'Bike 1', 'Bike 2']
    
    # Load all sheets from the Excel file
    xl = pd.ExcelFile(file_name)
    
    # Create a dictionary to hold data from all tasks
    task_data = {}
    
    for sheet in sheet_names:
        if sheet in xl.sheet_names:  # Ensure the sheet exists
            # Read each sheet (task)
            df = xl.parse(sheet)
            
            # Check if 'BELT' and 'BIOPAC' columns exist
            if 'BELT' in df.columns and 'BIOPAC' in df.columns:
                task_data[sheet] = df[['BELT', 'BIOPAC']]  # Extract the ECG columns
            else:
                print(f"Error: 'BELT' or 'BIOPAC' columns not found in sheet '{sheet}'.")
                print(f"Available columns: {df.columns.tolist()}")
                return None  # Stop execution if columns are missing
        else:
            print(f"Sheet '{sheet}' not found in the Excel file.")
            return None  # Stop execution if a sheet is missing
    
    return task_data

# Step 2: Calculate HR and Detect R-peaks from ECG Data
def calculate_hr_and_r_peaks(ecg_signal, sampling_rate=125):
    # Detect R-peaks in the ECG signal
    peaks, _ = find_peaks(ecg_signal, distance=sampling_rate * 0.6)
    
    # Calculate R-R intervals
    rr_intervals = np.diff(peaks) / sampling_rate  # In seconds
    
    # Convert to beats per minute (BPM)
    heart_rate = 60 / rr_intervals if len(rr_intervals) > 0 else np.array([0])
    
    return heart_rate, peaks  # Return HR array and R-peaks

# Step 3: Plot HR changes for BELT and BIOPAC over time (without ECG and R-peaks)
def plot_hr_only(task_name, hr_values_belt, hr_values_biopac, peaks_belt, peaks_biopac):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot HR trend for both BELT and BIOPAC on the same y-axis
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Heart Rate (BPM)', color='orange')
    
    # Plot HR trend as thicker lines for both BELT and BIOPAC
    ax.plot(peaks_belt[1:], hr_values_belt, label='HR Trend (BELT)', color='orange', linewidth=3)
    ax.plot(peaks_biopac[1:], hr_values_biopac, label='HR Trend (BIOPAC)', color='purple', linewidth=3)
    ax.tick_params(axis='y', labelcolor='orange')
    
    # Add legends and grid
    ax.legend(loc='upper right')
    ax.grid(True)

    plt.title(f'HR Trend - {task_name}')
    plt.tight_layout()
    plt.show()

# Step 4: Bland-Altman Analysis
def bland_altman_plot(belt_hr, biopac_hr):
    mean_hr = np.mean([belt_hr, biopac_hr], axis=0)
    diff_hr = belt_hr - biopac_hr  # Difference between methods
    mean_diff = np.mean(diff_hr)  # Mean difference
    std_diff = np.std(diff_hr)  # Standard deviation of difference

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_hr, diff_hr, color='blue', label='Differences')
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean difference')
    plt.axhline(mean_diff + 1.96 * std_diff, color='green', linestyle='--', label='+1.96 SD')
    plt.axhline(mean_diff - 1.96 * std_diff, color='green', linestyle='--', label='-1.96 SD')

    plt.title('Bland-Altman Plot: BELT vs. BIOPAC HR')
    plt.xlabel('Mean of BELT and BIOPAC HR (BPM)')
    plt.ylabel('Difference between BELT and BIOPAC HR (BPM)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 5: Correlation Analysis
def correlation_analysis(belt_hr, biopac_hr):
    corr, _ = pearsonr(belt_hr, biopac_hr)
    print(f'Correlation coefficient between BELT and BIOPAC HR: {corr:.2f}')

# Step 6: Analyze HR Trends, Bland-Altman, and Correlation Analysis
def analyze_hr_trends_and_peaks(task_data, sampling_rate=125):
    belt_hr_trend = []
    biopac_hr_trend = []
    tasks = []

    for task, data in task_data.items():
        belt_ecg = data['BELT']
        biopac_ecg = data['BIOPAC']
        
        # Calculate HR and R-peaks for both sources
        rr_intervals_belt = np.diff(find_peaks(belt_ecg, distance=sampling_rate*0.6)[0]) / sampling_rate
        hr_values_belt = 60 / rr_intervals_belt if len(rr_intervals_belt) > 0 else np.array([0])
        
        rr_intervals_biopac = np.diff(find_peaks(biopac_ecg, distance=sampling_rate*0.6)[0]) / sampling_rate
        hr_values_biopac = 60 / rr_intervals_biopac if len(rr_intervals_biopac) > 0 else np.array([0])
        
        # Append HR trends
        belt_hr_trend.extend(hr_values_belt)
        biopac_hr_trend.extend(hr_values_biopac)
        
        # Plot HR trends
        plot_hr_only(task, hr_values_belt, hr_values_biopac, 
                     find_peaks(belt_ecg, distance=sampling_rate*0.6)[0], 
                     find_peaks(biopac_ecg, distance=sampling_rate*0.6)[0])

    # Perform Bland-Altman analysis
    bland_altman_plot(np.array(belt_hr_trend), np.array(biopac_hr_trend))
    
    # Perform Correlation analysis
    correlation_analysis(np.array(belt_hr_trend), np.array(biopac_hr_trend))

    return belt_hr_trend, biopac_hr_trend

# Step 7: Run Full Analysis
def run_full_analysis(file_name, sampling_rate=125):
    # Step 1: Load ECG data from specific sheets
    task_data = load_ecg_data(file_name)
    
    if task_data is None:
        print("Terminating the analysis due to missing columns or sheets.")
        return
    
    # Step 2: Analyze HR trends, Bland-Altman, and Correlation
    belt_hr_trend, biopac_hr_trend = analyze_hr_trends_and_peaks(task_data, sampling_rate)

    print("Analysis complete.")

# Main execution
if __name__ == "__main__":
    # ADD YOUR EXCEL FILE NAME HERE. Example: 'C:/path/to/your/file.xlsx'
    excel_file = 'P6 Data.xlsx'  # <<<<< Change this line to the path of your Excel file
    
    if excel_file:
        run_full_analysis(excel_file)
    else:
        print("No file selected!")
