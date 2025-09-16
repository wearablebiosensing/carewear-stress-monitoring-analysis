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
        self.title("CSV Data Tool")
        self.geometry("1000x950")
        self.output_df = None
        self.plot_df = None

        # --- Variables ---
        self.folder_path = tk.StringVar(self)
        self.search_string = tk.StringVar(self)
        self.labels_file_path = tk.StringVar(self)
        self.plot_column = tk.StringVar(self)
        self.show_labels_var = tk.BooleanVar(value=False)
        self.timestamp_format_var = tk.StringVar(value="Auto-Detect")
        self.original_bg_color = None

        self.activity_mapping_num = {
            "None": -1, 'rest1': 1, 'prepare speech': 2, 'give speech': 3,
            'rest2': 4, 'mental math': 5, 'rest3': 6, 'stationary_Bike1': 7,
            'stationary_Bike2': 8,
        }

        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        log_frame_container = tk.Frame(self)
        log_frame_container.pack(side="top", fill="both", expand=True, padx=10, pady=5)
        
        plot_frame_container = tk.Frame(self, relief="groove", borderwidth=2)
        plot_frame_container.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

        self.setup_top_widgets(top_frame)

        log_label = tk.Label(log_frame_container, text="Log", font=("Helvetica", 12, "bold"))
        log_label.pack(pady=5)
        self.log_text = tk.Text(log_frame_container, wrap="word", state="disabled", font=("Courier", 10), height=8)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.setup_plotting_widgets(plot_frame_container)
    
    def setup_top_widgets(self, top_frame):
        concat_frame = tk.Frame(top_frame, relief="groove", borderwidth=2)
        concat_frame.pack(pady=5, fill="x")
        
        tk.Label(concat_frame, text="Step 1: Concatenate & Merge (Optional)", font=("Helvetica", 12, "bold")).pack(pady=5)

        drag_drop_frame = tk.Frame(concat_frame)
        drag_drop_frame.pack(pady=5, padx=10, fill="x")
        self.concat_drag_label = tk.Label(drag_drop_frame, text="Drag & Drop FOLDER Here to Concatenate", font=("Helvetica", 11), relief="groove", borderwidth=2, height=2)
        self.concat_drag_label.pack(fill="x")
        self.concat_drag_label.drop_target_register(DND_FILES)
        self.concat_drag_label.dnd_bind('<<Drop>>', self.handle_concat_drop)

        self.original_bg_color = self.concat_drag_label.cget("background")

        input_frame = tk.Frame(concat_frame)
        input_frame.pack(pady=5, fill="x", padx=10)
        tk.Label(input_frame, text="Selected Folder:").pack(side="left", padx=5)
        self.folder_entry = tk.Entry(input_frame, textvariable=self.folder_path, width=40)
        self.folder_entry.pack(side="left", padx=5)
        tk.Label(input_frame, text="File String Filter:").pack(side="left", padx=5)
        self.string_entry = tk.Entry(input_frame, textvariable=self.search_string, width=15)
        self.string_entry.pack(side="left", padx=5)
        
        tk.Button(input_frame, text="Concatenate Files", command=self.concatenate_files, bg="#AED6F1").pack(side="left", padx=10)
        
        # --- THIS BUTTON IS NOW INCLUDED ---
        tk.Button(input_frame, text="Save Concatenated", command=self.save_concatenated_file).pack(side="left", padx=10)

        merge_frame = tk.Frame(concat_frame)
        merge_frame.pack(pady=5, fill="x", padx=10)
        tk.Label(merge_frame, text="Labels File:").pack(side="left", padx=5)
        self.labels_file_entry = tk.Entry(merge_frame, textvariable=self.labels_file_path, width=50)
        self.labels_file_entry.pack(side="left", padx=5)
        tk.Button(merge_frame, text="Browse...", command=self.browse_labels_file).pack(side="left", padx=5)
        tk.Button(merge_frame, text="Merge & Save", command=self.merge_and_save, bg="#A9DFBF").pack(side="left", padx=10)

    def setup_plotting_widgets(self, plot_frame_container):
        plot_label = tk.Label(plot_frame_container, text="Plotting", font=("Helvetica", 12, "bold"))
        plot_label.pack(pady=5)

        self.plot_drag_label = tk.Label(plot_frame_container, text="Drag & Drop a Single CSV FILE Here to Plot", font=("Helvetica", 11), relief="groove", borderwidth=2, height=2)
        self.plot_drag_label.pack(pady=5, padx=10, fill="x")
        self.plot_drag_label.drop_target_register(DND_FILES)
        self.plot_drag_label.dnd_bind('<<Drop>>', self.handle_plot_file_drop)

        plot_controls_frame = tk.Frame(plot_frame_container)
        plot_controls_frame.pack(fill="x", pady=5, padx=10)
        
        tk.Label(plot_controls_frame, text="Select column to plot:").pack(side="left", padx=5)
        self.plot_option_menu = ttk.OptionMenu(plot_controls_frame, self.plot_column, "No data loaded", command=self.draw_plot)
        self.plot_option_menu.pack(side="left", padx=5)
        self.plot_option_menu.config(state="disabled")

        self.show_labels_chk = tk.Checkbutton(plot_controls_frame, text="Show Activity Labels", variable=self.show_labels_var, command=self.draw_plot)
        self.show_labels_chk.pack(side="left", padx=10)
        self.show_labels_chk.config(state="disabled")
        
        tk.Label(plot_controls_frame, text="Timestamp Interpretation:").pack(side="left", padx=5)
        timestamp_options = ["Auto-Detect", "12-Hour PM Fix"]
        self.timestamp_option_menu = ttk.OptionMenu(plot_controls_frame, self.timestamp_format_var, timestamp_options[0], *timestamp_options)
        self.timestamp_option_menu.pack(side="left", padx=5)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame_container)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame_container)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def log_message(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def reset_drop_labels(self):
        self.concat_drag_label.config(bg=self.original_bg_color)
        self.plot_drag_label.config(bg=self.original_bg_color)

    def handle_concat_drop(self, event):
        self.reset_drop_labels()
        path = re.sub(r'^{|}$', '', event.data)
        if os.path.isdir(path):
            self.folder_path.set(path)
            self.concat_drag_label.config(text=f"Folder Dropped: {os.path.basename(path)}", bg="#90EE90")
            self.log_message(f"Folder for concatenation selected: {path}\n")
        else:
            messagebox.showerror("Invalid Drop", "Please drop a FOLDER in this area.")

    def handle_plot_file_drop(self, event):
        self.reset_drop_labels()
        path = re.sub(r'^{|}$', '', event.data)
        if os.path.isfile(path) and path.lower().endswith('.csv'):
            self.plot_drag_label.config(text=f"File Dropped: {os.path.basename(path)}", bg="#90EE90")
            self.log_message(f"File for plotting selected: {path}\n")
            try:
                df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
                self.plot_df = self._prepare_data_timestamps(df)
                if 'Timestamp_pd' in self.plot_df.columns:
                    self.plot_df.sort_values(by='Timestamp_pd', inplace=True, ignore_index=True)
                self.update_plot_options()
            except Exception as e:
                messagebox.showerror("File Read Error", f"Failed to read or process the file:\n{e}")
                self.log_message(f"Error processing dropped file: {e}\n")
        else:
            messagebox.showerror("Invalid Drop", "Please drop a single CSV FILE in this area.")

    def browse_labels_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.labels_file_path.set(file_path)
            self.log_message(f"Selected labels file: {os.path.basename(file_path)}\n")

    def concatenate_files(self):
        folder_path, search_string = self.folder_path.get(), self.search_string.get()
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
                self.log_message(f"  - Processed {os.path.basename(file)}: {df.shape[0]} rows\n")
            except Exception as e:
                self.log_message(f"  - Error reading {os.path.basename(file)}: {e}\n")
        if df_list:
            self.output_df = pd.concat(df_list, ignore_index=True)
            self.log_message(f"\nSuccessfully concatenated {len(df_list)} file(s).\n")
            self.log_message(f"Result has {self.output_df.shape[0]} rows. Ready to merge or save.\n")
        else:
            self.output_df = None
            self.log_message(f"\nNo files containing '{search_string}' were found.\n")

    def _prepare_data_timestamps(self, df):
        df_copy = df.copy()
        source_col = None
        for col_name in ['Timestamp_pd', 'Timestamp', 'date_time']:
            if col_name in df_copy.columns:
                source_col = col_name
                break
        
        if not source_col:
            self.log_message("Warning: No timestamp column found. Plotting against index.\n")
            return df_copy

        df_copy['Timestamp_pd'] = pd.to_datetime(df_copy[source_col], errors='coerce')
        
        if self.timestamp_format_var.get() == "12-Hour PM Fix":
            self.log_message("Applying '12-Hour PM Fix' to timestamps...\n")
            condition = df_copy['Timestamp_pd'].dt.hour < 12
            df_copy.loc[condition, 'Timestamp_pd'] += pd.Timedelta(hours=12)
        
        return df_copy

    # --- NEW FUNCTION FOR THE BUTTON ---
    def save_concatenated_file(self):
        if self.output_df is None:
            messagebox.showerror("Error", "No concatenated data to save. Please 'Concatenate Files' first.")
            return

        self.log_message("\nPreparing concatenated data for saving...\n")
        try:
            df_to_save = self._prepare_data_timestamps(self.output_df)
        except Exception as e:
            messagebox.showerror("Timestamp Error", f"Failed to prepare timestamps:\n{e}")
            self.log_message(f"Error preparing timestamps for save: {e}\n")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="concatenated_data.csv"
        )
        if file_path:
            try:
                df_to_save.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Concatenated data successfully saved to {file_path}")
                self.log_message(f"Saved concatenated file to: {file_path}\n")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file: {e}")

    def add_activity_labels(self, data_df, labels_df):
        data_df.dropna(subset=['Timestamp_pd'], inplace=True)
        if data_df.empty:
            raise ValueError("Data is empty after handling timestamps.")
        labels_df.dropna(subset=['start_time', 'end_time', 'manual_labels_activity'], inplace=True)
        numeric_start_times = pd.to_numeric(labels_df['start_time'], errors='coerce')
        first_valid_time = numeric_start_times.dropna().iloc[0] if not numeric_start_times.dropna().empty else None
        if first_valid_time is not None and first_valid_time > 1000000000:
            self.log_message("  - Detected millisecond timestamp format in labels file.\n")
            for col in ('start_time', 'end_time'):
                labels_df[col] = pd.to_datetime(labels_df[col], unit='ms', errors='coerce')
                if labels_df[col].dt.tz is not None:
                    labels_df[col] = labels_df[col].dt.tz_localize(None)
        else:
            self.log_message("  - Detected string time format in labels file.\n")
            base_date = data_df['Timestamp_pd'].min().date()
            for col in ('start_time', 'end_time'):
                parsed_times = pd.to_datetime(labels_df[col], errors='coerce').dt.time
                labels_df[col] = [pd.to_datetime(f"{base_date} {t}") if pd.notna(t) else pd.NaT for t in parsed_times]
        rows_before_drop = len(labels_df)
        labels_df.dropna(subset=['start_time', 'end_time'], inplace=True)
        if rows_before_drop > len(labels_df):
            self.log_message(f"  - Skipped {rows_before_drop - len(labels_df)} label rows due to invalid time format.\n")
        data_df["manual_labels_activity"] = "None"
        for _, row in labels_df.iterrows():
            mask = (data_df['Timestamp_pd'] >= row['start_time']) & (data_df['Timestamp_pd'] <= row['end_time'])
            data_df.loc[mask, "manual_labels_activity"] = row['manual_labels_activity']
        return data_df

    def merge_and_save(self):
        if self.output_df is None:
            messagebox.showerror("Error", "No data available to merge. Please 'Concatenate Files' first.")
            return
        labels_path = self.labels_file_path.get()
        if not labels_path or not os.path.exists(labels_path):
            messagebox.showerror("Error", "Please select a valid labels file.")
            return
        self.log_message("\nStarting label merge process...\n")
        try:
            data_df = self._prepare_data_timestamps(self.output_df)
            labels_df = pd.read_csv(labels_path)
            merged_data = self.add_activity_labels(data_df, labels_df)
            self.log_message("Sorting merged data by timestamp...\n")
            merged_data.sort_values(by='Timestamp_pd', inplace=True, ignore_index=True)
            self.plot_df = merged_data
            unique_labels = self.plot_df['manual_labels_activity'].unique()
            self.log_message(f"Merge successful. Found labels: {unique_labels}\n")
            self.update_plot_options()
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                initialfile="merged_data_with_labels.csv")
            if file_path:
                self.plot_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Merged data successfully saved to {file_path}")
                self.log_message(f"Saved merged file to: {file_path}\n")
        except Exception as e:
            messagebox.showerror("Merge Error", f"An error occurred during merging: {e}")
            self.log_message(f"Error during merge: {e}\n")
    
    def update_plot_options(self):
        df_to_plot = self.plot_df
        if df_to_plot is None: return
        numeric_cols = df_to_plot.select_dtypes(include=np.number).columns.tolist()
        menu = self.plot_option_menu["menu"]
        menu.delete(0, "end")
        if 'manual_labels_activity' in df_to_plot.columns:
            self.show_labels_chk.config(state="normal")
        else:
            self.show_labels_chk.config(state="disabled")
            self.show_labels_var.set(False)
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
        df_to_plot = self.plot_df
        if df_to_plot is None or not self.plot_column.get() or self.plot_column.get() in ["No data loaded", "No numeric data"]:
            return
        selected_col = self.plot_column.get()
        self.ax.clear()
        
        if 'Timestamp_pd' in df_to_plot.columns:
            x_axis = df_to_plot['Timestamp_pd']
            x_label = "Timestamp"
            self.ax.plot(x_axis, df_to_plot[selected_col], label=selected_col, zorder=2)
            self.fig.autofmt_xdate()
        else:
            x_axis = df_to_plot.index
            x_label = "Index"
            self.ax.plot(x_axis, df_to_plot[selected_col], label=selected_col, zorder=2)

        if self.show_labels_var.get() and 'manual_labels_activity' in df_to_plot.columns and 'Timestamp_pd' in df_to_plot.columns:
            self.plot_activity_labels()

        self.ax.set_title(f"'{selected_col}' over Time")
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(selected_col)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_activity_labels(self):
        df = self.plot_df
        if df is None: return
        activity_changes = df[df['manual_labels_activity'] != df['manual_labels_activity'].shift()]
        ymin, ymax = self.ax.get_ylim()
        text_y_pos = ymin + (ymax - ymin) * 0.9
        for index, row in activity_changes.iterrows():
            current_activity = row['manual_labels_activity']
            prev_activity = df['manual_labels_activity'].iloc[index - 1] if index > 0 else "None"
            if prev_activity != "None":
                end_time = df['Timestamp_pd'].iloc[index - 1]
                self.ax.axvline(x=end_time, color='r', linestyle='--', linewidth=1.2, zorder=3)
            if current_activity != "None":
                start_time = row['Timestamp_pd']
                self.ax.axvline(x=start_time, color='g', linestyle='--', linewidth=1.2, zorder=3)
                activity_num = self.activity_mapping_num.get(current_activity, 'N/A')
                label_text = f' {activity_num}'
                self.ax.text(start_time, text_y_pos, label_text, color='black',
                             fontweight='bold', verticalalignment='center', zorder=4,
                             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
        last_row_activity = df['manual_labels_activity'].iloc[-1]
        if last_row_activity != "None":
             self.ax.axvline(x=df['Timestamp_pd'].iloc[-1], color='r', linestyle='--', linewidth=1.2, zorder=3)

if __name__ == "__main__":
    app = ConcatApp()
    app.mainloop()