from dash import Output, Input
import plotly.graph_objs as go
import os
import pandas as pd
from utils.time_diff_grouped import calculate_time_diff
from utils.data_quality_check import *
def register_tab3_callbacks(app, merged_lables_folder):
    # Prepare data at startup
    json_time_difference_all_part_lables = {}
    json_time_difference_all_part_activity = {}
    json_time_difference_all_part_manual = {}
    # Store participant scores
    quality_scores = []
    # Store participant scores
    quality_scores = []
    outlier_flatline_scores = []

    pattern = os.path.join(merged_lables_folder, "heart_rate*.csv")
    import glob
    for hr_path in sorted(glob.glob(pattern)):
        pid = hr_path.split("/")[-1].split('_')[2]
        df = pd.read_csv(hr_path)

        # Step 2: Convert the 'Timestamp' column to datetime, coercing errors to NaT
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Step 3: Drop rows with invalid timestamps (NaT)
        df = df.dropna(subset=['Timestamp'])
        # Step 3: Drop rows with invalid timestamps
        df = df.dropna(subset=['Timestamp'])
        hr_clean = clean_hr_signal(df["HeartRate"])
        report = calculate_signal_quality(hr_clean,df["Timestamp"])
        quality_scores.append({
            "Participant": participant_id,
            "Quality Score": report["quality_score"]
        })
        # Collect outliers and flatlines
        outlier_flatline_scores.append({
            "Participant": participant_id,
            "Outliers": report["outliers"],
            "Flatlines": report["flatlines"]
        })

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
    print("json_time_difference_all_part_manual = \n",json_time_difference_all_part_manual)
    participant_ids = sorted(
        set(json_time_difference_all_part_lables.keys()) |
        set(json_time_difference_all_part_activity.keys()) |
        set(json_time_difference_all_part_manual.keys()),
        key=lambda x: str(x)
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
               # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=("Heart Rate Signal Quality Scores per Participant", "Number of Outliers and Flatlines per Participant","Number of lables")
        )

        # fig = go.Figure(
        #     data=[
        belt_trace = go.Bar(
                    x=participant_ids,
                    y=passed_lables,
                    name='Belt labels',
                    marker_color='teal'
                )
        watch_trace = go.Bar(
                    x=participant_ids,
                    y=passed_activity,
                    name='Watch labels',
                    marker_color='orange'
                )
        manual_trace = go.Bar(
                    x=participant_ids,
                    y=passed_manual,
                    name='Manual activity',
                    marker_color='purple'
                )
        #     ],
        #     layout=go.Layout(
        #         xaxis_title='Participant ID',
        #         yaxis_title=f'Number of Activities Passed (≥ {threshold}s)',
        #         yaxis=dict(dtick=1),
        #         barmode='group',
        #         bargap=0.3,
        #         plot_bgcolor='white'
        #     )
        # )
        # First plot: Quality Score per Participant
        trace1 = go.Bar(
            x=quality_df["Participant"],
            y=quality_df["Quality Score"],
            text=quality_df["Quality Score"].round(2),
            textposition='outside',
            marker_color='teal',
            name="Quality Score"
        )

        # Second plot: Outliers and Flatlines per Participant (grouped)
        outlier_trace = go.Bar(
            x=outlier_flatline_df["Participant"],
            y=outlier_flatline_df["Outliers"],
            name="Outliers",
            marker_color='royalblue',
            text=outlier_flatline_df["Outliers"],
            textposition='outside'
        )
        flatline_trace = go.Bar(
            x=outlier_flatline_df["Participant"],
            y=outlier_flatline_df["Flatlines"],
            name="Flatlines",
            marker_color='tomato',
            text=outlier_flatline_df["Flatlines"],
            textposition='outside'
        )

        # Add traces
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(outlier_trace, row=2, col=1)
        fig.add_trace(flatline_trace, row=2, col=1)

        fig.add_trace(belt_trace,row=3, col=1)
        fig.add_trace(watch_trace,row=3, col=1)
        fig.add_trace(manual_trace,row=3, col=1)


        # Update layout for clarity
        fig.update_layout(
            height=900,
            showlegend=True,
            barmode="group",
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        fig.update_yaxes(range=[0, 1.05], row=1, col=1, title="Quality Score (0–1)")
        fig.update_xaxes(title="Participant",showticklabels=True, row=1, col=1)

        fig.update_yaxes(title="Count", row=2, col=1)
        fig.update_xaxes(title="Participant", row=2, col=1)

        fig.show()

        return fig
