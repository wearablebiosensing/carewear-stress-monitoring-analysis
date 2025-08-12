import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def create_activity_subplots(df, activity_col_name):
    rows, cols = 4, 4
    activities = [a for a in df[activity_col_name].dropna().unique() if a != "None"]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Activity: {a}" for a in activities] + [""] * (rows * cols - len(activities))
    )
    json_time_difference = {}
    current_row, current_col = 1, 1
    for activity in activities:
        activity_df = df[df[activity_col_name] == activity]
        first_col = activity_df['Timestamp_pd'].iloc[0]
        last_col = activity_df['Timestamp_pd'].iloc[-1]
        if isinstance(first_col, str) and isinstance(last_col, str):
            time1 = datetime.strptime(first_col, "%Y-%m-%d %H:%M:%S.%f")
            time2 = datetime.strptime(last_col, "%Y-%m-%d %H:%M:%S.%f")
            json_time_difference[activity] = abs((time1 - time2).total_seconds())
        fig.add_trace(
            go.Scattergl(
                x=activity_df['Timestamp_pd'],
                y=activity_df['HeartRate'],
                mode='markers+lines',
                marker=dict(size=8, color='#1d8cad'),
                line=dict(dash='dash', color='#1d8cad'),
                name=f"Activity: {activity}"
            ),
            row=current_row, col=current_col
        )
        current_col += 1
        if current_col > cols:
            current_col = 1
            current_row += 1
        if current_row > rows:
            break
    # Bar plot for time differences
    fig_time_diff = go.Figure()
    fig_time_diff.add_trace(go.Bar(
        x=list(json_time_difference.keys()),
        y=list(json_time_difference.values()),
        name="Time Differences",
        marker_color='#1dad84'
    ))
    # Annotations
    annotations = []
    for i, (activity, value) in enumerate(json_time_difference.items()):
        annotations.append(dict(
            x=i, y=value, text=str(value), xref='x', yref='y',
            showarrow=False, yshift=10, font=dict(size=12)
        ))
    fig_time_diff.update_layout(
        autosize=True,
        title=dict(text="Time difference in each activity", x=0.5, y=1, xanchor='center', font=dict(size=18)),
        height=400, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='white',
        xaxis_title='Activity', yaxis_title='Seconds',
        xaxis=dict(title=dict(font=dict(size=14))),
        yaxis=dict(title=dict(font=dict(size=14))),
        annotations=annotations,
        shapes=[
            dict(type='line', x0=-0.5, y0=300, x1=len(json_time_difference) - 0.5, y1=300,
                 line=dict(color='black', dash='dash', width=1))
        ]
    )
    fig_time_diff.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig_time_diff.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_layout(
        autosize=True, title_text="Activity Subplots", showlegend=False, plot_bgcolor='white'
    )
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    return fig, fig_time_diff

def belt_subplots(df):
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("Respiration1", "Respiration2", "Respiration3", "ECG")
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Respiration1"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Respiration2"]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Respiration3"]), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ECG"]), row=4, col=1)
    fig.update_layout(height=900, width=3000, title_text="Belt data plots")
    return fig
