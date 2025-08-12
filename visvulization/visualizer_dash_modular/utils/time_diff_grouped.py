from datetime import datetime

def calculate_time_diff(df, activity_col_name):
    activities = [a for a in df[activity_col_name].dropna().unique() if a != "None"]
    json_time_difference = {}
    for activity in activities:
        activity_df = df[df[activity_col_name] == activity]
        first_col = activity_df['Timestamp_pd'].iloc[0]
        last_col = activity_df['Timestamp_pd'].iloc[-1]
        if isinstance(first_col, str) and isinstance(last_col, str):
            time1 = datetime.strptime(first_col, "%Y-%m-%d %H:%M:%S.%f")
            time2 = datetime.strptime(last_col, "%Y-%m-%d %H:%M:%S.%f")
            json_time_difference[activity] = abs((time1 - time2).total_seconds())
    return json_time_difference
