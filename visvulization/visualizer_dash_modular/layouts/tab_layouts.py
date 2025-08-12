from dash import dcc, html

def tab1_layout(csv_files, csv_files_belt):
    return html.Div([
        html.H1("CareWear Dashboard: Tab 1", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='file-dropdown',
            options=[{'label': file, 'value': file} for file in csv_files],
            value=csv_files[0] if csv_files else None,
            placeholder="Select a Participant File"
        ),
        html.Div(id='main-plot-container', style={'height': '500px', 'backgroundColor': '#f0f8ff', 'border': '2px solid #0000ff'}),
        html.Div(id='activity-plots-container', style={'height': '500px', 'border': '2px solid #0000ff'}),
        html.Div(id='time-difference-plot-container', style={'height': '400px', 'border': '2px solid #0000ff'}),
        dcc.Dropdown(
            id='belt-file-dropdown',
            options=[{'label': file, 'value': file} for file in csv_files_belt],
            value=csv_files_belt[0] if csv_files_belt else None,
            placeholder="Select a Participant File"
        ),
        html.Div(id='belt-plot-container', style={'height': '400px', 'border': '2px solid #0000ff'})
    ])

def tab2_layout(merge_lables_hr):
    return html.Div([
        html.H1("CareWear Dashboard: Merged labels", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='file-dropdown-tab2',
            options=[{'label': file_merge, 'value': file_merge} for file_merge in merge_lables_hr],
            value=merge_lables_hr[0] if merge_lables_hr else None,
            placeholder="Select a Participant File"
        ),
        html.Div(id='main-plot-container-tab2', style={'height': '500px', 'backgroundColor': '#f0f8ff', 'border': '2px solid #0000ff'}),
        html.Div(id='activity-plots-container-tab2', style={'height': '500px', 'backgroundColor': '#f0f8ff', 'border': '2px solid #0000ff'}),
        html.Div(id='time-difference-plot-container-tab2', style={'height': '400px', 'border': '2px solid #0000ff'})
    ])


def tab3_layout():
    return html.Div([
        html.H2("Grouped Bar Chart: Activities Passed per Participant"),
        html.Div([
            html.Label("Set threshold (seconds): "),
            dcc.Input(
                id='threshold-input-tab3',
                type='number',
                value=300,
                min=0,
                step=1,
                debounce=True,
                style={'marginRight': '10px'}
            ),
        ], style={'marginBottom': '20px'}),
        dcc.Graph(id='grouped-bar-plot-tab3')
    ])

