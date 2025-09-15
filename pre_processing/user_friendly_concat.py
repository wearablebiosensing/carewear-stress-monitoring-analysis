import pandas as pd
import numpy as np
import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
from tkinterdnd2 import DND_FILES, TkinterDnD

# Matplotlib imports for plotting
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class ConcatApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Concatenator, Merger, and Plotter")
        self.geometry("1000x900") # Increased size for the plot
        self.output_df = None
        self.merged_df = None

        # --- Variables ---
        self.folder_path = tk.StringVar(self)
        self.search_string = tk.StringVar(self)
        self.labels_file_path = tk.StringVar(self)
        self.plot_column = tk.StringVar(self)

        # --- Widgets ---
        self.create_widgets()

    def create_widgets(self):
        # --- Main layout frames ---
        top_frame = tk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        log_frame_container = tk.Frame(self)
        log_frame_container.pack(side="top", fill="both", expand=True, padx=10, pady=5)
        
        plot_frame_container = tk.Frame(self, relief="groove", borderwidth=2)
        plot_frame_container.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

        # --- Drag and Drop Frame ---
        drag_drop_frame = tk.Frame(top_frame, relief="groove", borderwidth=2)
        drag_drop_frame.pack(pady=5, fill="x")
        self.drag_label = tk.Label(drag_drop_frame, text="Drag and Drop Folder Here", font=("Helvetica", 12))
        self.drag_label.pack(pady=10)
        self.drag_label.drop_target_register(DND_FILES)
        self.drag_label.dnd_bind('<<Drop>>', self.handle_drop)

        # --- Input Frame for Concatenation ---
        input_frame = tk.Frame(top_frame)
        input_frame.pack(pady=5, fill="x")

        # --- Button Frame ---
        button_frame = tk.Frame(top_frame)
        button_frame.pack(pady=5)
        
        # --- Frame for Label Merging ---
        merge_frame = tk.Frame(top_frame, relief="groove", borderwidth=2)
        merge_frame.pack(pady=5, fill="x")

        # --- Log Display Frame ---
        log_label = tk.Label(log_frame_container, text="Log", font=("Helvetica", 12, "bold"))
        log_label.pack(pady=5)
        self.log_text = tk.Text(log_frame_container, wrap="word", state="disabled", font=("Courier", 10), height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Plotting Frame ---
        plot_label = tk.Label(plot_frame_container, text="Plotting", font=("Helvetica", 12, "bold"))
        plot_label.pack(pady=5)
        
        plot_controls_frame = tk.Frame(plot_frame_container)
        plot_controls_frame.pack(fill="x", pady=5)
        
        tk.Label(plot_controls_frame, text="Select column to plot:").pack(side="left", padx=5)
        self.plot_option_menu = ttk.OptionMenu(plot_controls_frame, self.plot_column, "No data loaded", command=self.draw_plot)
        self.plot_option_menu.pack(side="left", padx=5)
        self.plot_option_menu.config(state="disabled")

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame_container)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame_container)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # --- (Re-adding the other widgets here for context) ---
        folder_label = tk.Label(input_frame, text="Selected Folder:")
        folder_label.pack(side="left", padx=5)
        self.folder_entry = tk.Entry(input_frame, textvariable=self.folder_path, width=40)
        self.folder_entry.pack(side="left", padx=5)
        string_label = tk.Label(input_frame, text="File String:")
        string_label.pack(side="left", padx=5)
        self.string_entry = tk.Entry(input_frame, textvariable=self.search_string, width=15)
        self.string_entry.pack(side="left", padx=5)

        concat_button = tk.Button(button_frame, text="1. Concatenate Files", command=self.concatenate_files, bg="#AED6F1")
        concat_button.pack(side="left", padx=10)
        save_button = tk.Button(button_frame, text="Download Concatenated CSV", command=self.save_csv)
        save_button.pack(side="left", padx=10)

        merge_label_title = tk.Label(merge_frame, text="Step 2: Merge Activity Labels (Optional)", font=("Helvetica", 12, "bold"))
        merge_label_title.pack(pady=5)
        labels_input_frame = tk.Frame(merge_frame)
        labels_input_frame.pack(pady=5)
        labels_file_label = tk.Label(labels_input_frame, text="Labels File:")
        labels_file_label.pack(side="left", padx=5)
        self.labels_file_entry = tk.Entry(labels_input_frame, textvariable=self.labels_file_path, width=50)
        self.labels_file_entry.pack(side="left", padx=5)
        browse_button = tk.Button(labels_input_frame, text="Browse...", command=self.browse_labels_file)
        browse_button.pack(side="left", padx=5)
        merge_button = tk.Button(merge_frame, text="2. Merge Labels & Save", command=self.merge_and_save, bg="#A9DFBF")
        merge_button.pack(pady=10)
    
    def log_message(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def handle_drop(self, event):
        path = re.sub(r'^{|}$', '', event.data)
        if os.path.isdir(path):
            self.folder_path.set(path)
            self.drag_label.config(text=f"Folder Dropped: {os.path.basename(path)}")
            self.log_text.config(state="normal")
            self.log_text.delete("1.0", tk.END)
            self.log_text.config(state="disabled")
        else:
            messagebox.showerror("Invalid Drop", "Please drop a folder, not a file.")

    def browse_labels_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.labels_file_path.set(file_path)
            self.log_message(f"Selected labels file: {os.path.basename(file_path)}\n")

    def concatenate_files(self):
        folder_path, search_string = self.folder_path.get(), self.search_string.get()
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

        if not folder_path or not os.path.isdir(folder_path):
            self.log_message("Error: Please select or drop a valid folder.\n")
            return
        if not search_string:
            self.log_message("Error: Please enter a file search string.\n")
            return

        df_list = []
        self.log_message(f"Searching for files containing '{search_string}' in '{folder_path}'...\n")
        
        for file in glob.glob(os.path.join(folder_path, f"*{search_string}*.csv")):
            try:
                df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
                df_list.append(df)
                self.log_message(f"  - Processed {os.path.basename(file)}: {df.shape[0]} rows, {df.shape[1]} columns\n")
            except Exception as e:
                self.log_message(f"  - Error reading {os.path.basename(file)}: {e}\n")

        if df_list:
            self.output_df = pd.concat(df_list, ignore_index=True)
            self.log_message(f"\nSuccessfully concatenated {len(df_list)} file(s).\n")
            self.log_message(f"Final DataFrame has {self.output_df.shape[0]} rows and {self.output_df.shape[1]} columns.\n")
        else:
            self.output_df = None
            self.log_message(f"\nNo files containing '{search_string}' were found or processed.\n")

    def _prepare_data_timestamps(self, df):
        df_copy = df.copy()
        source_col = 'Timestamp' if 'Timestamp' in df_copy.columns else 'Timestamp_pd'
        if not source_col in df_copy.columns:
            raise ValueError("Data file must contain a 'Timestamp' or 'Timestamp_pd' column.")
        
        df_copy['Timestamp_pd'] = pd.to_datetime(df_copy[source_col], errors='coerce')
        
        self.log_message("Correcting raw data timestamps from AM to PM...\n")
        condition = df_copy['Timestamp_pd'].dt.hour < 12
        df_copy.loc[condition, 'Timestamp_pd'] += pd.Timedelta(hours=12)
        
        return df_copy

    def save_csv(self):
        if self.output_df is None:
            messagebox.showerror("Error", "No data to save. Please concatenate files first.")
            return
        try:
            combined_df = self._prepare_data_timestamps(self.output_df)
        except ValueError as e:
            messagebox.showerror("Timestamp Error", str(e))
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                combined_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data successfully saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file: {e}")

    def add_activity_labels(self, data_df, labels_df):
        data_df.dropna(subset=['Timestamp_pd'], inplace=True)
        if data_df.empty:
            raise ValueError("Data is empty after handling timestamps.")
        base_date = data_df['Timestamp_pd'].min().date()

        labels_df.dropna(subset=['start_time', 'end_time', 'manual_labels_activity'], inplace=True)

        for col in ('start_time', 'end_time'):
            parsed_times = pd.to_datetime(labels_df[col], errors='coerce').dt.time
            
            labels_df[col] = [
                pd.to_datetime(f"{base_date} {t}") if pd.notna(t) else pd.NaT
                for t in parsed_times
            ]

        rows_before_drop = len(labels_df)
        labels_df.dropna(subset=['start_time', 'end_time'], inplace=True)
        rows_after_drop = len(labels_df)

        if rows_before_drop > rows_after_drop:
            self.log_message(f"  - Skipped {rows_before_drop - rows_after_drop} label rows due to invalid time format (e.g., '2:45:PM').\n")

        data_df["manual_labels_activity"] = "None"
        
        for _, row in labels_df.iterrows():
            start, stop, activity = row['start_time'], row['end_time'], row['manual_labels_activity']
            mask = (data_df['Timestamp_pd'] >= start) & (data_df['Timestamp_pd'] <= stop)
            data_df.loc[mask, "manual_labels_activity"] = activity
        
        return data_df


    def merge_and_save(self):
        if self.output_df is None:
            messagebox.showerror("Error", "No data available. Please 'Concatenate Files' first.")
            return
        
        labels_path = self.labels_file_path.get()
        if not labels_path or not os.path.exists(labels_path):
            messagebox.showerror("Error", "Please select a valid labels file.")
            return

        self.log_message("\nStarting label merge process...\n")
        try:
            data_df = self._prepare_data_timestamps(self.output_df)
            labels_df = pd.read_csv(labels_path)

            self.merged_df = self.add_activity_labels(data_df, labels_df)
            
            # **NEW**: Sort the dataframe by Timestamp_pd before plotting or saving
            self.log_message("Sorting merged data by timestamp...\n")
            self.merged_df.sort_values(by='Timestamp_pd', inplace=True)
            
            unique_labels = self.merged_df['manual_labels_activity'].unique()
            self.log_message(f"Merge successful. Found labels: {unique_labels}\n")

            self.update_plot_options()

            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile="merged_data_with_labels.csv"
            )
            if file_path:
                self.merged_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Merged data successfully saved to {file_path}")
                self.log_message(f"Saved merged file to: {file_path}\n")

        except Exception as e:
            messagebox.showerror("Merge Error", f"An error occurred during merging: {e}")
            self.log_message(f"Error during merge: {e}\n")
    
    def update_plot_options(self):
        if self.merged_df is None:
            return
        
        numeric_cols = self.merged_df.select_dtypes(include=np.number).columns.tolist()
        
        menu = self.plot_option_menu["menu"]
        menu.delete(0, "end")
        
        if not numeric_cols:
            self.plot_column.set("No numeric data")
            self.plot_option_menu.config(state="disabled")
            return

        for col in numeric_cols:
            menu.add_command(label=col, command=lambda value=col: self.plot_column.set(value))
        
        self.plot_column.set(numeric_cols[0]) 
        self.plot_option_menu.config(state="normal")
        self.draw_plot()

    def draw_plot(self, event=None):
        if self.merged_df is None or not self.plot_column.get() or self.plot_column.get() == "No data loaded":
            return
        
        selected_col = self.plot_column.get()
        
        self.ax.clear()
        
        self.ax.plot(self.merged_df['Timestamp_pd'], self.merged_df[selected_col], label=selected_col)
        
        self.ax.set_title(f"'{selected_col}' over Time")
        self.ax.set_xlabel("Timestamp")
        self.ax.set_ylabel(selected_col)
        self.ax.grid(True)
        self.ax.legend()
        self.fig.autofmt_xdate()
        
        self.canvas.draw()


if __name__ == "__main__":
    app = ConcatApp()
    app.mainloop()