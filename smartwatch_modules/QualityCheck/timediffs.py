from datetime import datetime
import pandas as pd
import numpy as np
import os
import glob
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# Two dictionaries for each activity column
json_time_difference_all_part_lables = {}
json_time_difference_all_part_activity = {}

def calculate_time_diff(df, activity_col_name):
    activities = [a for a in df[activity_col_name].dropna().unique() if a != "None"]
    json_time_difference = {}
    for activity in activities:
        activity_df = df[df[activity_col_name] == activity]
        first_col = activity_df['Timestamp_pd'].iloc[0]
        last_col = activity_df['Timestamp_pd'].iloc[-1]
        if isinstance(first_col, str) and isinstance(last_col, str):
            time1 = datetime.strptime(first_col, "%Y-%m-%d %H:%M:%S.%f")
            time2 = datetime.strptime(last_col, "%Y-%m-%d %H:%M:%S.%f")
            json_time_difference[activity] = abs((time1 - time2).total_seconds())
    return json_time_difference

if __name__ == "__main__":
    INPUT_DIR = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/merged_lables"
    # find all heart_rate_P*.csv
    pattern = os.path.join(INPUT_DIR, "heart_rate*.csv")
    for hr_path in sorted(glob.glob(pattern)):
        print("hr_path", hr_path)
        pid = hr_path.split("/")[-1].split('_')[2]
        print("pid: ", pid)
        df = pd.read_csv(hr_path)
        print(df.columns)
        # For Activity_Lables
        if "manual_labels_activity" in df.columns:
            json_time_difference_lables = calculate_time_diff(df, "manual_labels_activity")
            json_time_difference_all_part_lables[pid] = json_time_difference_lables
            print("Manual json_time_difference_all_part_lables = ",json_time_difference_all_part_lables)
        else:
            json_time_difference_all_part_lables[pid] = {}
        # For activity
        if "activity" in df.columns:
            json_time_difference_activity = calculate_time_diff(df, "activity")
            json_time_difference_all_part_activity[pid] = json_time_difference_activity
        else:
            json_time_difference_all_part_activity[pid] = {}

    print("Belt_Activity_Labels:", json_time_difference_all_part_lables)
    print("activity:", json_time_difference_all_part_activity)

    # Prepare participant list for consistent ordering (union of keys)
    participant_ids = sorted(
        set(json_time_difference_all_part_lables.keys()) | set(json_time_difference_all_part_activity.keys()),
        key=lambda x: int(x)
    )

    # Dash App
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H2("Grouped Bar Chart: Activities Passed per Participant"),
        html.Div([
            html.Label("Set threshold (seconds): "),
            dcc.Input(
                id='threshold-input',
                type='number',
                value=300,
                min=0,
                step=1,
                debounce=True,
                style={'marginRight': '10px'}
            ),
        ], style={'marginBottom': '20px'}),
        dcc.Graph(id='grouped-bar-plot')
    ])

    @app.callback(
        Output('grouped-bar-plot', 'figure'),
        Input('threshold-input', 'value')
    )
    def update_grouped_bar(threshold):
        if threshold is None:
            threshold = 300
        # Compute "passed" counts for each participant in both dictionaries
        passed_lables = []
        passed_activity = []
        for pid in participant_ids:
            acts_lables = json_time_difference_all_part_lables.get(pid, {})
            print("belt activiy == ",acts_lables)
            acts_activity = json_time_difference_all_part_activity.get(pid, {})
            print("watch activity== ",acts_activity)
            count_lables = sum(1 for v in acts_lables.values() if v >= threshold)
            count_activity = sum(1 for v in acts_activity.values() if v >= threshold)
            passed_lables.append(count_lables)
            passed_activity.append(count_activity)
        # Build grouped bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=participant_ids,
                    y=passed_lables,
                    name='manual_labels_activity',
                    marker_color='teal'
                ),
                go.Bar(
                    x=participant_ids,
                    y=passed_activity,
                    name='activity',
                    marker_color='orange'
                )
            ],
            layout=go.Layout(
                xaxis_title='Participant ID',
                yaxis_title=f'Number of Activities Passed (≥ {threshold}s)',
                yaxis=dict(dtick=1),
                barmode='group',
                bargap=0.3,
                plot_bgcolor='white'
            )
        )
        return fig

    app.run(debug=True)
