import pdfplumber
import pandas as pd

# Open the PDF file
with pdfplumber.open("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Labels_copy.pdf") as pdf:
    # Initialize lists to hold data
    activities = []
    starts = []
    stops = []

    # Iterate through each page
    for page in pdf.pages:
        # Extract text from the page
        text = page.extract_text()
        
        # Here you need to parse the text into activities, starts, and stops
        # This step is highly dependent on the structure of your PDF
        # For example, if each line contains "activity,start,stop" separated by commas:
        for line in text.splitlines():
            try:
                activity, start, stop = line.split(',')
                activities.append(activity.strip())
                starts.append(start.strip())
                stops.append(stop.strip())
            except ValueError:
                # Handle lines that do not match the expected format
                pass

# Create a DataFrame
df = pd.DataFrame({
    "activity": activities,
    "start": starts,
    "stop": stops
})

# Save to CSV
df.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/task_timeline.csv", index=False)
