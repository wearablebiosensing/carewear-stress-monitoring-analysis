import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Path to the folder containing CSV files
folder_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/merged_lables"  # Replace with the actual path

# Define the name to filter by
filter_name = "heart"  # Replace with the desired keyword

# Get list of CSV files containing the filter name
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and filter_name in f]

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Participant Activity Dashboard")
        self.root.geometry("1200x800")  # Set window size

        # Create dropdown for selecting files
        self.file_var = tk.StringVar()
        self.file_var.set(csv_files[0] if csv_files else None)  # Default value
        self.file_dropdown = ttk.Combobox(self.root, textvariable=self.file_var)
        self.file_dropdown['values'] = csv_files
        self.file_dropdown.pack(pady=10)

        # Button to load data and plot
        self.load_button = tk.Button(self.root, text="Load Data", command=self.load_and_plot)
        self.load_button.pack()

        # Frame to hold plots
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True)

    def load_and_plot(self):
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Load data from the selected file
        selected_file = self.file_var.get()
        if not selected_file:
            return

        file_path = os.path.join(folder_path, selected_file)
        df = pd.read_csv(file_path)

        # Ensure the data has the correct columns (adjust as per your dataset)
        if 'activity' not in df.columns or 'Timestamp_pd' not in df.columns or 'HeartRate' not in df.columns:
            label = tk.Label(self.plot_frame, text="Invalid data format", fg="red")
            label.pack()
            return

        # Get unique activities
        unique_activities = df['Activity_Lables'].dropna().unique()

        # Create plots dynamically for each unique activity
        row_val = 0
        for activity in unique_activities:
            activity_df = df[df['Activity_Lables'] == activity]
            activity_df = activity_df.sort_values(by='Timestamp_pd')

            # Create figure and axis
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)

            # Plot data
            ax.plot(activity_df['Timestamp_pd'], activity_df['HeartRate'], marker='o', linestyle='--', color='blue')
            ax.set_title(f"Activity_Lables: {activity}")
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Heart Rate')

            # Add plot to frame
            plot_frame = tk.Frame(self.plot_frame)
            plot_frame.pack(side=tk.LEFT, padx=10)
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

if __name__ == '__main__':
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()
