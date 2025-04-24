import os
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__)

# Path to the folder containing CSV files
folder_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Concat_File/"  # Replace with the actual path

# Get list of CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and "heart" in f]

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Participant Activity Dashboard", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': file, 'value': file} for file in csv_files],
        value=csv_files[0] if csv_files else None,
        placeholder="Select a Participant File"
    ),
    html.Div([
        html.Div(
            id='main-plot-container',
            style={'height': '300px'}
        ),
        html.Div(
            id='activity-plots-container',
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '10px'}
        )
    ], style={'display': 'flex', 'flexDirection': 'column'}),
    html.Div(
        id='time-difference-plot-container',
        style={'height': '400px', 'marginTop': '20px'}
    )
])

# Callback to update plots based on selected file
@app.callback(
    [Output('main-plot-container', 'children'),
     Output('activity-plots-container', 'children'),
     Output('time-difference-plot-container', 'children')],
    [Input('file-dropdown', 'value')]
)
def update_plots(selected_file):
    if not selected_file:
        return [html.Div("No file selected", style={'color': 'red'})], [], []

    # Load data from the selected file
    file_path = os.path.join(folder_path, selected_file)
    df = pd.read_csv(file_path)
    df = df.sort_values(by='Timestamp_pd')
    df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')

    # Ensure the data has the correct columns
    if 'activity' not in df.columns or 'Timestamp_pd' not in df.columns or 'HeartRate' not in df.columns:
        return [html.Div("Invalid data format", style={'color': 'red'})], [], []

    # Downsample data if necessary
    if len(df) > 1000:
        df_downsampled = df.sample(frac=0.1)
    else:
        df_downsampled = df
    df_downsampled = df_downsampled.sort_values(by='Timestamp_pd')

    # Get unique activities
    unique_activities = df_downsampled['activity'].dropna().unique()

    # Create a plot for overall heart rate data
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(
        x=df_downsampled.index,#['Timestamp_pd'],
        y=df_downsampled['HeartRate'],
        mode='markers+lines',
        marker=dict(size=8, color='#1d8cad'),
        line=dict(dash='dash', color='#1d8cad'),
        name="Overall Heart Rate"
    ))

    # Update layout
    fig_main.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        xaxis_title='Timestamp',
        yaxis_title='Heart Rate (bpm)',
        xaxis=dict(title=dict(font=dict(size=14))),
        yaxis=dict(title=dict(font=dict(size=14)))
    )

    # Create plots dynamically for each unique activity
    activity_plots = []
    json_time_difference = {}
    for activity in unique_activities:
        if activity != "None" and activity != "activity":
            activity_df = df_downsampled[df_downsampled['activity'] == activity]
            first_col = activity_df['Timestamp_pd'].iloc[0]
            last_col = activity_df['Timestamp_pd'].iloc[-1]
            time1 = datetime.strptime(first_col, "%Y-%m-%d %H:%M:%S.%f")
            time2 = datetime.strptime(last_col, "%Y-%m-%d %H:%M:%S.%f")
            difference_in_seconds = abs((time1 - time2).total_seconds())
            json_time_difference[activity] = difference_in_seconds

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=activity_df.index,
                y=activity_df['HeartRate'],
                mode='markers+lines',
                marker=dict(size=8, color='#1d8cad'),
                line=dict(dash='dash', color='#1d8cad'),
                name=f"activity: {activity}"
            ))

            fig.update_layout(
                title=f"activity: {activity}",
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor='white',
                xaxis_title='Timestamp',
                yaxis_title='Heart Rate',
                xaxis=dict(title=dict(font=dict(size=14))),
                yaxis=dict(title=dict(font=dict(size=14)))
            )
            activity_plots.append(dcc.Graph(figure=fig))

    # Create bar plot for time differences
    fig_time_diff = go.Figure()
    fig_time_diff.add_trace(go.Bar(
        x=list(json_time_difference.keys()),
        y=list(json_time_difference.values()),
        name="Time Differences",
        marker_color='#1dad84'
    ))

    annotations = []
    for i, (activity, value) in enumerate(json_time_difference.items()):
        annotations.append(dict(
            x=i,
            y=value,
            text=str(value),
            xref='x',
            yref='y',
            showarrow=False,
            yshift=10,
            font=dict(size=12)
        ))

    fig_time_diff.update_layout(
        title="Time Differences Between Activities",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        xaxis_title='Activity',
        yaxis_title='Time Difference (seconds)',
        xaxis=dict(title=dict(font=dict(size=14))),
        yaxis=dict(title=dict(font=dict(size=14))),
        annotations=annotations
    )

    return [dcc.Graph(figure=fig_main)], activity_plots, [dcc.Graph(figure=fig_time_diff)]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
