# Convert to DataFrame
df = pd.DataFrame(data).T.stack().reset_index()
df.columns = ['Participant', 'Activity', 'Samples']

# Add quality label
threshold = 500
df['Quality'] = df['Samples'].apply(lambda x: 'Good' if x >= threshold else 'Bad')

# Summary stats
total_activities = len(df)
good_activities = (df['Quality'] == 'Good').sum()
bad_activities = (df['Quality'] == 'Bad').sum()
percent_good = round(good_activities / total_activities * 100, 2)
percent_bad = round(bad_activities / total_activities * 100, 2)

# Per-activity quality summary
activity_summary = df.groupby(['Activity', 'Quality']).size().unstack(fill_value=0)
activity_summary['Total'] = activity_summary.sum(axis=1)
activity_summary['% Good'] = round(activity_summary['Good'] / activity_summary['Total'] * 100, 2)

# Per-participant quality summary
participant_summary = df.groupby(['Participant', 'Quality']).size().unstack(fill_value=0)
participant_summary['Total'] = participant_summary.sum(axis=1)
participant_summary['% Good'] = round(participant_summary['Good'] / participant_summary['Total'] * 100, 2)

(df.head(), activity_summary, participant_summary, percent_good, percent_bad)