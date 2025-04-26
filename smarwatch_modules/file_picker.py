import tkinter as tk
from tkinter import filedialog

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

# Ask the user to pick the first folder
folder1 = filedialog.askdirectory(title="Select the first folder")

# Ask the user to pick the second folder
folder2 = filedialog.askdirectory(title="Select the second folder")

# Print them to check
print(f"First folder selected: {folder1}")
print(f"Second folder selected: {folder2}")
