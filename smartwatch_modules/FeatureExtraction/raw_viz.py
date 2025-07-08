import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

DATA_DIR = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024-25/Concat_File"

def list_participant_files(data_dir):
    return [f for f in os.listdir(data_dir) if f.endswith('.csv')]

participant_files = list_participant_files(DATA_DIR)

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Participant HR Data Visualization"),
    dcc.Dropdown(
        id='participant-dropdown',
        options=[{'label': f, 'value': f} for f in participant_files],
        value=participant_files[0] if participant_files else None,
        clearable=False,
        style={'width': '60%'}
    ),
    dcc.Graph(id='hr-graph'),
    dcc.Graph(id='timestamp-line-plot')
])

@app.callback(
    [Output('hr-graph', 'figure'),
     Output('timestamp-line-plot', 'figure')],
    Input('participant-dropdown', 'value')
)
def update_graphs(selected_file):
    if not selected_file:
        return go.Figure(), go.Figure()
    df = pd.read_csv(os.path.join(DATA_DIR, selected_file))
    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'])
    df = df.sort_values('Timestamp_pd')

    # Main HR scattergl plot
    trace1 = go.Scattergl(
        x=df['Timestamp_pd'],
        y=df['HeartRate'],
        mode='markers+lines',
        marker=dict(size=8, color='#1d8cad'),
        line=dict(dash='dash', color='#1d8cad'),
        name="Heart Rate"
    )
    layout1 = go.Layout(
        title=f"Heart Rate Over Time: {selected_file}",
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Heart Rate (bpm)'),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Second line plot: timestamp as y, index as x
    # For a line plot of timestamp values, plot index vs. timestamp string
    trace2 = go.Scatter(
        # x=df['Timestamp_pd'],
        y=df['Timestamp_pd'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        mode='lines',
        marker=dict(size=6, color='#e67e22'),
        line=dict(color='#e67e22'),
        name="Timestamp"
    )
    layout2 = go.Layout(
        title="Timestamp Sequence",
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Timestamp (formatted)'),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return go.Figure(data=[trace1], layout=layout1), go.Figure(data=[trace2], layout=layout2)

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
