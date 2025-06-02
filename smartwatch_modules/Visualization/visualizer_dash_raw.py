import os
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import re
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
# Hide the main tkinter window
root = tk.Tk()
root.withdraw()
# Ask the user to pick the first folder
folder1 = filedialog.askdirectory(title="Select Concat_File folder")
# Ask the user to pick the second folder
folder2 = filedialog.askdirectory(title="Select the merged_lables folder")

# Print them to check
print(f"First folder selected: {folder1}")
print(f"Second folder selected: {folder2}")

# Path to the folder containing CSV files
folder_path = folder1 #"/DATASET/StudyData_Drive_2024/Concat_File/"  # Replace with your path
merged_lables_folder = folder2 #"DATASET/StudyData_Drive_2024/merged_lables"

# Define the name to filter by
filter_name = "heart"  # Replace with the desired keyword
# Get list of CSV files containing the filter name
csv_files_hr = [f for f in os.listdir(folder_path) if f.endswith('.csv') and filter_name in f]
print(csv_files_hr)
# timestamp,Respiration1,Respiration2,Respiration3,ECG,activity,datetime_est
# Get list of CSV files containing the filter name
csv_files_belt = [f for f in os.listdir(folder_path) if f.endswith('.csv') and "belt" in f]
print("csv_files_belt: ",csv_files_belt)

# Function to extract participant ID from file name
def extract_participant_id(file_name):
    match = re.search(r'_P(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')  # Use inf for files without IDs

merge_lables_hr = [f for f in os.listdir(merged_lables_folder) if f.endswith('.csv') and filter_name in f]
print("merge_lables_hr:",merge_lables_hr)


# Sort files by extracted participant ID
csv_files = sorted(csv_files_hr, key=extract_participant_id)


# Layout of the dashboard
app.layout = html.Div([
    html.Div([
        html.Nav([
            dcc.Link('Tab 1', href='/tab1', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
            dcc.Link('Tab 2', href='/tab2', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
            dcc.Link('Tab 3', href='/tab3', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
            dcc.Link('Tab 4', href='/tab4', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
            dcc.Link('Tab 5', href='/tab5', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'})
        ], style={'width': '200px', 'height': '100vh', 'position': 'fixed', 'background-color': '#f0f0f0'})
    ], style={'width': '200px'}),
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div(id='tabs-content', style={'margin-left': '220px'})
    ], style={'display': 'inline-block', 'width': 'calc(100% - 220px)'})
])

# Callback to render content based on URL path
@app.callback(
    Output('tabs-content', 'children'),
    [Input('url', 'pathname')]
)
def render_content(pathname):
    if pathname == '/tab1':
        return html.Div([
            html.H1("CareWear Dashboard: Tab 1", style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': file, 'value': file} for file in csv_files],
                value=csv_files[0] if csv_files else None,
                placeholder="Select a Participant File"
            ),
                html.Div(
                    id='main-plot-container',
                    style={
                        'height': '500px',
                        'backgroundColor': '#f0f8ff',  # Set background color (e.g., Alice Blue)
                        'border': '2px solid #0000ff'  # Adds a blue border with 2px thickness

                    }
                ),
                html.Div(
                    id='activity-plots-container',
                    style={ 'height': '500px',
                            'border': '2px solid #0000ff'}
                ),
            html.Div(
                id='time-difference-plot-container',
                style={'height': '400px',  'border': '2px solid #0000ff'}
            ),
            dcc.Dropdown(
                id='belt-file-dropdown',
                options=[{'label': file, 'value': file} for file in csv_files_belt],
                value=csv_files_belt[0] if csv_files_belt else None,
                placeholder="Select a Participant File"
            ),
             html.Div(
                id='belt-plot-container',
                style={'height': '400px',  'border': '2px solid #0000ff'}
            )
        ])
    elif pathname == '/tab2':
        return  html.Div([
            html.H1("CareWear Dashboard: Merged lables", style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='file-dropdown-tab2',
                options=[{'label': file_merge, 'value': file_merge} for file_merge in merge_lables_hr],
                value=merge_lables_hr[0] if merge_lables_hr else None,
                placeholder="Select a Participant File"
            ),
                html.Div(
                    id='main-plot-container-tab2',
                    style={
                        'height': '500px',
                        'backgroundColor': '#f0f8ff',  # Set background color (e.g., Alice Blue)
                        'border': '2px solid #0000ff'  # Adds a blue border with 2px thickness

                    }
                ),
            html.Div(
                    id='activity-plots-container-tab2',
                    style={
                        'height': '500px',
                        'backgroundColor': '#f0f8ff',  # Set background color (e.g., Alice Blue)
                        'border': '2px solid #0000ff'  # Adds a blue border with 2px thickness

                    }
                ),
            html.Div(
                id='time-difference-plot-container-tab2',
                style={'height': '400px',  'border': '2px solid #0000ff'}
            )
        ])
    elif pathname == '/tab3':
        return html.Div("Tab 3 content")
    elif pathname == '/tab4':
        return html.Div("Tab 4 content")
    elif pathname == '/tab5':
        return html.Div("Tab 5 content")
    else:
        return html.Div("Select a tab")
# activity_col_name="activity"
def create_activity_subplots(df,activity_col_name): 
    # Define the number of rows and columns for the grid
    rows, cols = 4, 4

    # Get unique activities (excluding "None")
    activities = [activity for activity in df[activity_col_name].dropna().unique() if activity != "None"]

    # Initialize the subplot figure
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Activity: {activity}" for activity in activities] + [""] * (rows * cols - len(activities))
    )
    json_time_difference = {}

    # Populate subplots with activity data
    current_row, current_col = 1, 1
    for activity in activities:
        activity_df = df[df[activity_col_name] == activity]
        if activity != "None":
            activity_df = df[df[activity_col_name] == activity]
            # Calculate time difference of start and end times in seconds.
            first_col = activity_df['Timestamp_pd'].iloc[0]
            last_col = activity_df['Timestamp_pd'].iloc[-1]
            print("first_col,last_col: ",first_col,last_col,type(first_col),type(last_col))
            if type(first_col) == str and type(last_col) == str:
                time1 = datetime.strptime(first_col, "%Y-%m-%d %H:%M:%S.%f")
                time2 = datetime.strptime(last_col, "%Y-%m-%d %H:%M:%S.%f")
                difference_in_seconds = abs((time1 - time2).total_seconds())
                json_time_difference[activity] = difference_in_seconds

        # Add scatter plot for each activity
        fig.add_trace(
            go.Scattergl(
                x=activity_df['Timestamp_pd'],
                y=activity_df['HeartRate'],
                mode='markers+lines',
                marker=dict(size=8, color='#1d8cad'),
                line=dict(dash='dash', color='#1d8cad'),
                name=f"Activity: {activity}"
            ),
            row=current_row,
            col=current_col
        )

        # Update row and column indices
        current_col += 1
        if current_col > cols:
            current_col = 1
            current_row += 1

        # Stop adding plots if we've filled all subplots
        if current_row > rows:
            break
            # Create bar plot for time differences
    fig_time_diff = go.Figure()
    fig_time_diff.add_trace(go.Bar(
        x=list(json_time_difference.keys()),
        y=list(json_time_difference.values()),
        name="Time Differences",
        marker_color='#1dad84'  # Change bar color to green (hex code)
    ))

    # Add annotations for bar values
    annotations = []
    for i, (activity, value) in enumerate(json_time_difference.items()):
        annotations.append(dict(
            x=i,  # Assuming x is categorical and starts from 0
            y=value,
            text=str(value),
            xref='x',
            yref='y',
            showarrow=False,
            yshift=10,  # Shift text slightly above the bar
            font=dict(size=12)
        ))

    # Update layout
    fig_time_diff.update_layout(
         autosize=True,
    title=dict(
        text="Time difference in each activity",
        x=0.5,  # Center title horizontally
        y=1,
        xanchor='center',  # Align title at the center
        font=dict(size=18)  # Set title font size
    ),
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',  # Set background color
        xaxis_title='Activity',  # Set X-axis label
        yaxis_title='Seconds',  # Set Y-axis label
        xaxis=dict(title=dict(font=dict(size=14))),  # Increase X-axis label size
        yaxis=dict(title=dict(font=dict(size=14))),  # Increase Y-axis label size
        annotations=annotations,
        shapes=[  # Add a horizontal dashed line at y=300
            dict(
                type='line',
                x0=-0.5,  # Start at the left edge of the plot
                y0=300,
                x1=len(json_time_difference) - 0.5,  # End at the right edge of the plot
                y1=300,
                line=dict(
                    color='black',
                    dash='dash',
                    width=1
                )
            )
        ]
    )
    fig_time_diff.update_yaxes(
        # text = "seconds",
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig_time_diff.update_xaxes(
        # text = "Activity",
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    # Update layout settings
    fig.update_layout(
        autosize=True,  # Enable automatic sizing of the graph
        # height=1000,  # Adjust overall height of the figure
        # width=1000,   # Adjust overall width of the figure
        title_text="Activity Subplots",
        showlegend=False,
        plot_bgcolor='white'
    )

    # Update axes styling for all subplots
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    return fig,fig_time_diff

def belt_subplots(df):
    # Initialize figure with subplots
    fig = make_subplots(
        rows=4, 
        cols=1, 
        subplot_titles=("Respiration1", "Respiration2", "Respiration3", "ECG")
    )
    # Add traces
    # # timestamp,Respiration1,Respiration2,Respiration3,ECG,activity,datetime_est
    fig.add_trace(go.Scatter(x=df.index, y=df["Respiration1"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Respiration2"]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Respiration3"]), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ECG"]), row=4, col=1)

    # Update layout
    fig.update_layout(height=900, width=3000, title_text="Belt data plots")
    return fig


# Callback to update plots based on selected file
@app.callback(
    [Output('main-plot-container', 'children'),
     Output('activity-plots-container', 'children'),
     Output('time-difference-plot-container', 'children')],
    [Input('file-dropdown', 'value')],
    prevent_initial_call=True
)
def update_plots(selected_file):
    if not selected_file:
        return [html.Div("No file selected", style={'color': 'red'})], [], []

    # Load data from the selected file
    file_path = os.path.join(folder_path, selected_file)
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if not {'Timestamp_pd', 'HeartRate', 'activity'}.issubset(df.columns):
        return [html.Div("Invalid data format", style={'color': 'red'})], [], []

    df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')
    df = df.sort_values(by='Timestamp_pd')

    # Create main plot for heart rate data
    fig_main = go.Figure()
    fig_main.add_trace(go.Scattergl(
        x=df['Timestamp_pd'],
        y=df['HeartRate'],
        mode='markers+lines',
        marker=dict(size=8, color='#1d8cad'),
        line=dict(dash='dash', color='#1d8cad'),
        name=""
    ))
    
    fig_main.update_layout(
    title=dict(
        text="Overall Heart Rate Data in bpm",
        x=0.5,  # Center title horizontally
        y=0.9,
        xanchor='center',  # Align title at the center
        font=dict(size=18)  # Set title font size
    ),
        autosize=True,  # Enable automatic sizing of the graph

    # height=600, width=900,
        plot_bgcolor='white',
        
        xaxis_title='Timestamp',
        yaxis_title='Heart Rate (bpm)',
            xaxis=dict(
        tickfont=dict(size=14),  # Increase X-axis tick font size
        title=dict(font=dict(size=16))  # Increase X-axis title font size
    ),
    yaxis=dict(
        tickfont=dict(size=14),  # Increase Y-axis tick font size
        title=dict(font=dict(size=16))  # Increase Y-axis title font size
    )
    )
    fig_main.update_yaxes(
        range=[df['HeartRate'].min(), df['HeartRate'].max()],

        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig_main.update_xaxes(
            # range=[df['Timestamp_pd'].min(), df['Timestamp_pd'].max()],
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    activity_fig,fig_time_diff = create_activity_subplots(df,"activity")

    # Return plots
    return [dcc.Graph(figure=fig_main, style={'width': '100%', 'height': '100%'})],[dcc.Graph(figure=activity_fig,style={'width': '100%', 'height': '100%'})] , [dcc.Graph(figure=fig_time_diff,style={'width': '100%', 'height': '100%'})]

# Callback to update plots based on selected file
@app.callback(
    [Output('main-plot-container-tab2', 'children'),
     Output('activity-plots-container-tab2', 'children'),
     Output('time-difference-plot-container-tab2', 'children')],
    [Input('file-dropdown-tab2', 'value')],
    prevent_initial_call=True
)
def update_plots_tab2(selected_file):
    if not selected_file:
        return [html.Div("No file selected", style={'color': 'red'})], [], []
    print("update_plots_tab2():/ ===== selected_file",selected_file)
    # Load data from the selected file
    file_path = os.path.join(merged_lables_folder, selected_file)
    df = pd.read_csv(file_path)

    df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')
    df = df.sort_values(by='Timestamp_pd')
    # Create main plot for heart rate data
    fig_main = go.Figure()
    fig_main.add_trace(go.Scattergl(
        x=df['Timestamp_pd'],
        y=df['HeartRate'],
        mode='markers+lines',
        marker=dict(size=8, color='#1d8cad'),
        line=dict(dash='dash', color='#1d8cad'),
        name=""
    ))
    fig_main.update_layout(
    title=dict(
        text="Overall Heart Rate Data in bpm",
        x=0.5,  # Center title horizontally
        y=0.9,
        xanchor='center',  # Align title at the center
        font=dict(size=18)  # Set title font size
    ),
        autosize=True,  # Enable automatic sizing of the graph

    # height=600, width=900,
        plot_bgcolor='white',
        
        xaxis_title='Timestamp',
        yaxis_title='Heart Rate (bpm)',
            xaxis=dict(
        tickfont=dict(size=14),  # Increase X-axis tick font size
        title=dict(font=dict(size=16))  # Increase X-axis title font size
    ),
    yaxis=dict(
        tickfont=dict(size=14),  # Increase Y-axis tick font size
        title=dict(font=dict(size=16))  # Increase Y-axis title font size
    )
    )
    fig_main.update_yaxes(
        range=[df['HeartRate'].min(), df['HeartRate'].max()],

        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig_main.update_xaxes(
            # range=[df['Timestamp_pd'].min(), df['Timestamp_pd'].max()],
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    activity_fig,fig_time_diff = create_activity_subplots(df,"Activity_Lables")

    return [dcc.Graph(figure=fig_main, style={'width': '100%', 'height': '100%'})],[dcc.Graph(figure=activity_fig,style={'width': '100%', 'height': '100%'})] , [dcc.Graph(figure=fig_time_diff,style={'width': '100%', 'height': '100%'})]

# Callback to update plots based on selected file
@app.callback(
    [Output('belt-plot-container', 'children')],
    [Input('belt-file-dropdown', 'value')],
    prevent_initial_call=True
)
def update_belt_plots(selected_file):
    if not selected_file:
            return [html.Div("No file selected", style={'color': 'red'})], [], []
    print("update_belt_plots():/ ===== selected_file",selected_file)
    # Load data from the selected file
    file_path = os.path.join(folder_path, selected_file)
    print("update_belt_plots(): ",file_path)
    df = pd.read_csv(file_path)
    belt_fig = belt_subplots(df)
    return [dcc.Graph(figure=belt_fig, style={'width': '100%', 'height': '100%'})]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)