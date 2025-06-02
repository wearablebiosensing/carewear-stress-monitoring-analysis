from dash import dcc, html

def create_base_layout():
    return html.Div([
        html.Div([
            html.Nav([
                dcc.Link('Raw Data', href='/tab1', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
                dcc.Link('Merged Data', href='/tab2', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
                dcc.Link('Overall Qualitycheck ', href='/tab3', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
                dcc.Link('Tab 4', href='/tab4', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'}),
                dcc.Link('Tab 5', href='/tab5', style={'display': 'block', 'padding': '10px', 'background-color': '#d3d3d3'})
            ], style={'width': '200px', 'height': '100vh', 'position': 'fixed', 'background-color': '#f0f0f0'})
        ], style={'width': '200px'}),
        dcc.Location(id='url', refresh=False),
        html.Div([
            html.Div(id='tabs-content', style={'margin-left': '220px'})
        ], style={'display': 'inline-block', 'width': 'calc(100% - 220px)'})
    ])
