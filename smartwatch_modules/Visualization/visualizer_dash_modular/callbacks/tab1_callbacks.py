from dash import dcc, html
from dash.dependencies import Input, Output
from utils.data_loader import load_csv
from utils.visualization import create_activity_subplots
import pandas as pd
import numpy as np

def register_tab1_callbacks(app, folder_path):
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
        df = load_csv(folder_path, selected_file)
        if not {'Timestamp_pd', 'HeartRate', 'activity'}.issubset(df.columns):
            return [html.Div("Invalid data format", style={'color': 'red'})], [], []
        df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')
        df = df.sort_values(by='Timestamp_pd')
        import plotly.graph_objects as go
        # fig_main = go.Figure()
        # fig_main.add_trace(go.Scattergl(
        #     x=df['Timestamp_pd'],
        #     y=df['HeartRate'],
        #     mode='markers+lines',
        #     marker=dict(size=8, color='#1d8cad'),
        #     line=dict(dash='dash', color='#1d8cad'),
        #     name=""
        # )
        # )
        # fig_main.update_layout(
        #     title=dict(text="Overall Heart Rate Data in bpm", x=0.5, y=0.9, xanchor='center', font=dict(size=18)),
        #     autosize=True, plot_bgcolor='white', xaxis_title='Timestamp', yaxis_title='Heart Rate (bpm)',
        #     xaxis=dict(tickfont=dict(size=14), title=dict(font=dict(size=16))),
        #     yaxis=dict(tickfont=dict(size=14), title=dict(font=dict(size=16)))
        # )
        # fig_main.update_yaxes(range=[df['HeartRate'].min(), df['HeartRate'].max()],
        #                       mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        # fig_main.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')

        # Add stems (vertical lines)
        for i, row in df.iterrows():
            fig_main.add_trace(go.Scatter(x=[row['Timestamp'], row['Timestamp']],
                                    y=[0, row['Amplitude']],
                                    mode='lines',
                                    line=dict(color='blue')))
        fig_main = go.Figure()
        # Add markers at sample points
        fig_main.add_trace(go.Scatter(x=df['Sample'], y=df['Amplitude'],
                                mode='markers',
                                marker=dict(color='blue', size=8)))

        fig_main.update_layout(title='Discrete Time Signal with Variable Sample Rate',
                        xaxis_title='Sample Points',
                        yaxis_title='Amplitude',
                        showlegend=False)

        fig_main.show()
        activity_fig, fig_time_diff = create_activity_subplots(df, "activity")
        return [dcc.Graph(figure=fig_main, style={'width': '100%', 'height': '100%'})], \
               [dcc.Graph(figure=activity_fig, style={'width': '100%', 'height': '100%'})], \
               [dcc.Graph(figure=fig_time_diff, style={'width': '100%', 'height': '100%'})]
