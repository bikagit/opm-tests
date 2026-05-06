import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Data for first 10 days
numeric_data = [
    # Group 0
    [0.77107, 0.00000, 0], [0.77107, 0.00000, 0], [0.77107, 0.00000, 0],
    # Group 1
    [0.38044, 0.38424, 1], [0.40178, 0.33203, 1], [0.59814, 0.06252, 1],
    # Group 2
    [0.62194, 0.01549, 2], [0.62568, 0.01238, 2], [0.65114, 0.00267, 2],
    # Group 3
    [0.34950, 0.46391, 3], [0.38399, 0.37539, 3], [0.40062, 0.33480, 3],
    # Group 4
    [0.61533, 0.02296, 4], [0.61913, 0.01831, 4], [0.62411, 0.01361, 4],
    # Group 5
    [0.34622, 0.47255, 5], [0.38406, 0.37523, 5], [0.39994, 0.33644, 5],
    # Group 6
    [0.60846, 0.03441, 6], [0.61897, 0.01849, 6], [0.62181, 0.01561, 6],
    # Group 7
    [0.35329, 0.45397, 7], [0.38514, 0.37255, 7], [0.40888, 0.31531, 7],
    # Group 8
    [0.60654, 0.03852, 8], [0.61930, 0.01813, 8], [0.62141, 0.01598, 8],
    # Group 9
    [0.34398, 0.47845, 9], [0.38444, 0.37427, 9], [0.39482, 0.34884, 9]
]

actual_date_data = [
    [0.77107, 0.00000, '01.Jan 2015'], [0.77107, 0.00000, '01.Jan 2015'], [0.77107, 0.00000, '01.Jan 2015'],
    [0.38044, 0.38424, '02.Jan 2015'], [0.40178, 0.33203, '02.Jan 2015'], [0.59814, 0.06252, '02.Jan 2015'],
    [0.62194, 0.01549, '03.Jan 2015'], [0.62568, 0.01238, '03.Jan 2015'], [0.65114, 0.00267, '03.Jan 2015'],
    [0.34950, 0.46391, '04.Jan 2015'], [0.38399, 0.37539, '04.Jan 2015'], [0.40062, 0.33480, '04.Jan 2015'],
    [0.61533, 0.02296, '05.Jan 2015'], [0.61913, 0.01831, '05.Jan 2015'], [0.62411, 0.01361, '05.Jan 2015'],
    [0.34622, 0.47255, '06.Jan 2015'], [0.38406, 0.37523, '06.Jan 2015'], [0.39994, 0.33644, '06.Jan 2015'],
    [0.60846, 0.03441, '07.Jan 2015'], [0.61897, 0.01849, '07.Jan 2015'], [0.62181, 0.01561, '07.Jan 2015'],
    [0.35329, 0.45397, '08.Jan 2015'], [0.38514, 0.37255, '08.Jan 2015'], [0.40888, 0.31531, '08.Jan 2015'],
    [0.60654, 0.03852, '09.Jan 2015'], [0.61930, 0.01813, '09.Jan 2015'], [0.62141, 0.01598, '09.Jan 2015'],
    [0.34398, 0.47845, '10.Jan 2015'], [0.38444, 0.37427, '10.Jan 2015'], [0.39482, 0.34884, '10.Jan 2015']
]

# Convert to DataFrames
df_numeric = pd.DataFrame(numeric_data, columns=['SWAT', 'OILKR', 'Group'])
df_dates = pd.DataFrame(actual_date_data, columns=['SWAT', 'OILKR', 'Date'])

# Create subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Chronological Plot with Numeric Group Indices (First 10 Days)",
                                    "Chronological Plot with Actual Date Labels (First 10 Days)"))
colors = px.colors.qualitative.Set1

# Plot 1: Numeric group indices
for i, group in enumerate(sorted(df_numeric['Group'].unique())):
    day_data = df_numeric[df_numeric['Group'] == group]
    fig.add_trace(go.Scatter(x=day_data['SWAT'], y=day_data['OILKR'],
                             mode='lines+markers',
                             name=f'Group {group}',
                             line=dict(color=colors[i % len(colors)], width=2),
                             marker=dict(size=6)), row=1, col=1)

# Plot 2: Actual date labels
for i, date in enumerate(sorted(df_dates['Date'].unique())):
    day_data = df_dates[df_dates['Date'] == date]
    fig.add_trace(go.Scatter(x=day_data['SWAT'], y=day_data['OILKR'],
                             mode='lines+markers',
                             name=date,
                             line=dict(color=colors[i % len(colors)], width=2),
                             marker=dict(size=6)), row=1, col=2)

# Layout
fig.update_layout(title_text="Comparison of OILKR vs SWAT Representations for First 10 Days",
                  showlegend=False,
                  xaxis_title='SWAT', yaxis_title='OILKR')

# Show plot
fig.show()