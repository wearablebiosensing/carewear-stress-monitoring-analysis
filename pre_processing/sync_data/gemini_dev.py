import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Set your Gemini API key here
# It's best practice to use an environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Functions ---

def load_data(file_path):
    """
    Loads time-series data from a CSV file.
    Assumes the first column is the timestamp.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        print("✅ Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
        return None

def generate_summary(df):
    """
    Generates a statistical summary of the time-series data.
    """
    summary = {}
    for column in df.columns:
        series = df[column]
        summary[column] = {
            'mean': series.mean(),
            'std_dev': series.std(),
            'min': series.min(),
            'max': series.max(),
            'quartiles': series.quantile([0.25, 0.5, 0.75]).to_dict(),
            'missing_values': series.isnull().sum(),
            'total_points': len(series)
        }
    print("✅ Statistical summary generated.")
    return summary

def plot_data(df, file_name="timeseries_plot.png"):
    """
    Creates and saves a plot of the time-series data.
    """
    plt.figure(figsize=(12, 6))
    df.plot(ax=plt.gca())
    plt.title("Time-Series Data Visualization")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"✅ Plot saved to {file_name}")
    return file_name

def analyze_with_gemini(summary, plot_path):
    """
    Uses the Gemini API to analyze the data summary and plot for quality issues.
    """
    # Create the prompt with the statistical summary and a clear instruction
    summary_text = "Here is a statistical summary of the time-series data:\n"
    for col, data in summary.items():
        summary_text += f"\nColumn: {col}\n"
        for key, value in data.items():
            summary_text += f"  - {key}: {value}\n"

    prompt = (
        f"Analyze the following time-series data summary and plot to identify "
        f"any potential data quality issues, inconsistencies, or anomalies. "
        f"Look for things like significant outliers, unexpected missing value counts, "
        f"abrupt changes in mean or standard deviation, or inconsistent patterns. "
        f"Provide a clear, concise summary of your findings. "
        f"\n\n{summary_text}"
    )

    # Use the Gemini Pro Vision model for analysis with both text and image
    model = genai.GenerativeModel('gemini-pro-vision')

    with open(plot_path, "rb") as image_file:
        image_data = image_file.read()

    # Pass the prompt and the image to the model
    response = model.generate_content([prompt, {"mime_type": "image/png", "data": image_data}])
    
    print("✅ Analysis from Gemini received.")
    return response.text

# --- Main Execution ---
if __name__ == "__main__":
    # --- Sample Data Creation ---
    # Create a dummy CSV file for demonstration
    sample_data = {
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='H'),
        'sensor_A': [100] * 50 + [10, 12, 11] + [98] * 47,  # Abrupt change
        'sensor_B': [50] * 100,
        'sensor_C': [i + 5* (i % 2) for i in range(100)], # Inconsistent pattern
    }
    sample_df = pd.DataFrame(sample_data).set_index('timestamp')
    sample_df.loc['2025-01-02 12:00:00', 'sensor_B'] = pd.NA # Missing value
    sample_df.to_csv("timeseries_data.csv")
    print("✅ Sample data file 'timeseries_data.csv' created.")
    
    # --- The Tool's Workflow ---
    file_path = "timeseries_data.csv"
    
    # Step 1: Load the data
    df = load_data(file_path)
    if df is not None:
        # Step 2: Generate the summary
        summary = generate_summary(df)

        # Step 3: Plot the data for visual context
        plot_file = plot_data(df)

        # Step 4: Use Gemini to analyze the summary and plot
        analysis_result = analyze_with_gemini(summary, plot_file)
        
        # Step 5: Present the findings
        print("\n" + "="*50)
        print("🔍 Data Quality Analysis Report")
        print("="*50)
        print("\n➡️ Statistical Summary:")
        print(pd.DataFrame.from_dict({(i,j): summary[i][j] for i in summary.keys() for j in summary[i].keys()}, orient='index').T)
        print("\n➡️ Gemini's Analysis:")
        print(analysis_result)
        
        # Clean up the generated plot file
        os.remove(plot_file)
        os.remove(file_path)
        print(f"\n✅ Cleaned up temporary files.")