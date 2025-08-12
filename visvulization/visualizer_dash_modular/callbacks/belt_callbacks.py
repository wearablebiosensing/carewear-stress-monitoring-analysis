from dash import dcc, html
from dash.dependencies import Input, Output
from utils.data_loader import load_csv
from utils.visualization import belt_subplots
import pandas as pd
import numpy as np
def register_belt_callbacks(app, folder_path):
    @app.callback(
        [Output('belt-plot-container', 'children')],
        [Input('belt-file-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_belt_plots(selected_file):
        if not selected_file:
            return [html.Div("No file selected", style={'color': 'red'})]
        df = load_csv(folder_path, selected_file)
        belt_fig = belt_subplots(df)
        return [dcc.Graph(figure=belt_fig, style={'width': '100%', 'height': '100%'})]
