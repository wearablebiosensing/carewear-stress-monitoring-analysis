from dash import Output, Input
import plotly.graph_objs as go
import os
import pandas as pd
from utils.time_diff_grouped import calculate_time_diff

def register_tab3_callbacks(app, merged_lables_folder):
    # Prepare data at startup
    json_time_difference_all_part_lables = {}
    json_time_difference_all_part_activity = {}
    json_time_difference_all_part_manual = {}

    pattern = os.path.join(merged_lables_folder, "heart_rate*.csv")
    import glob
    for hr_path in sorted(glob.glob(pattern)):
        pid = hr_path.split("/")[-1].split('_')[2]
        df = pd.read_csv(hr_path)
        # For Activity_Lables
        if "Belt_Activity_Labels" in df.columns:
            json_time_difference_lables = calculate_time_diff(df, "Belt_Activity_Labels")
            json_time_difference_all_part_lables[pid] = json_time_difference_lables
        else:
            json_time_difference_all_part_lables[pid] = {}
        # For activity watch lables experimenter labels
        if "activity" in df.columns:
            json_time_difference_activity = calculate_time_diff(df, "activity")
            json_time_difference_all_part_activity[pid] = json_time_difference_activity
        else:
            json_time_difference_all_part_activity[pid] = {}
        # For manual_Activity 
        if "manual_labels_activity" in df.columns:
            json_time_difference_manual = calculate_time_diff(df, "manual_labels_activity")
            json_time_difference_all_part_manual[pid] = json_time_difference_manual
        else:
            json_time_difference_all_part_manual[pid] = {}

    participant_ids = sorted(
        set(json_time_difference_all_part_lables.keys()) |
        set(json_time_difference_all_part_activity.keys()) |
        set(json_time_difference_all_part_manual.keys()),
        key=lambda x: int(x)
    )

    @app.callback(
        Output('grouped-bar-plot-tab3', 'figure'),
        Input('threshold-input-tab3', 'value')
    )
    def update_grouped_bar(threshold):
        if threshold is None:
            threshold = 300
        passed_lables = []
        passed_activity = []
        passed_manual = []
        for pid in participant_ids:
            acts_lables = json_time_difference_all_part_lables.get(pid, {})
            acts_activity = json_time_difference_all_part_activity.get(pid, {})
            acts_manual = json_time_difference_all_part_manual.get(pid, {})
            count_lables = sum(1 for v in acts_lables.values() if v >= threshold)
            count_activity = sum(1 for v in acts_activity.values() if v >= threshold)
            count_manual = sum(1 for v in acts_manual.values() if v >= threshold)
            passed_lables.append(count_lables)
            passed_activity.append(count_activity)
            passed_manual.append(count_manual)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=participant_ids,
                    y=passed_lables,
                    name='Belt_Activity_Labels',
                    marker_color='teal'
                ),
                go.Bar(
                    x=participant_ids,
                    y=passed_activity,
                    name='activity',
                    marker_color='orange'
                ),
                go.Bar(
                    x=participant_ids,
                    y=passed_manual,
                    name='manual_Activity',
                    marker_color='purple'
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
