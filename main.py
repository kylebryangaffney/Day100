import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

pd.options.display.float_format = '{:,.2f}'.format

def get_dataframe_info(df):
    shape = df.shape
    columns = df.columns.tolist()
    nan_values = df.isna().sum()
    duplicates = df.duplicated().sum()
    return shape, columns, nan_values, duplicates

df_hh_income = pd.read_csv('Median_Household_Income_2015.csv', encoding="windows-1252")
df_pct_poverty = pd.read_csv('Pct_People_Below_Poverty_Level.csv', encoding="windows-1252")
df_pct_completed_hs = pd.read_csv('Pct_Over_25_Completed_High_School.csv', encoding="windows-1252")
df_share_race_city = pd.read_csv('Share_of_Race_By_City.csv', encoding="windows-1252")
df_fatalities = pd.read_csv('Deaths_by_Police_US.csv', encoding="windows-1252")

info_hh_income = get_dataframe_info(df_hh_income)
info_pct_poverty = get_dataframe_info(df_pct_poverty)
info_pct_completed_hs = get_dataframe_info(df_pct_completed_hs)
info_share_race_city = get_dataframe_info(df_share_race_city)
info_fatalities = get_dataframe_info(df_fatalities)

# dfs_info = {
#     "df_hh_income": info_hh_income,
#     "df_pct_poverty": info_pct_poverty,
#     "df_pct_completed_hs": info_pct_completed_hs,
#     "df_share_race_city": info_share_race_city,
#     "df_fatalities": info_fatalities
# }

# for df_name, info in dfs_info.items():
#     print(f"DataFrame: {df_name}")
#     print(f"Shape: {info[0]}")
#     print(f"Columns: {info[1]}")
#     print(f"NaN Values:\n{info[2]}")
#     print(f"Duplicate Rows: {info[3]}\n")

# def clean_dataframe(df):
#     df.fillna(0, inplace=True)
#     df.drop_duplicates(inplace=True)
#     return df

# df_hh_income_cleaned = clean_dataframe(df_hh_income)
# df_pct_poverty_cleaned = clean_dataframe(df_pct_poverty)
# df_pct_completed_hs_cleaned = clean_dataframe(df_pct_completed_hs)
# df_share_race_city_cleaned = clean_dataframe(df_share_race_city)
# df_fatalities_cleaned = clean_dataframe(df_fatalities)


# df_pct_poverty_sorted = df_pct_poverty.sort_values(by='poverty_rate', ascending=False)
# highest_poverty_state = df_pct_poverty_sorted.iloc[0]
# lowest_poverty_state = df_pct_poverty_sorted.iloc[-1]

# print(f"State with the highest poverty rate: {highest_poverty_state['Geographic Area']} ({highest_poverty_state['poverty_rate']}%)")
# print(f"State with the lowest poverty rate: {lowest_poverty_state['Geographic Area']} ({lowest_poverty_state['poverty_rate']}%)")
# fig = px.bar(df_pct_poverty_sorted, x='Geographic Area', y='poverty_rate', 
#              title='Poverty Rate by US State (Highest to Lowest)',
#              labels={'poverty_rate': 'Poverty Rate (%)', 'Geographic Area': 'US State'},
#              height=600)

# fig.show()

# df_pct_completed_hs_sorted = df_pct_completed_hs.sort_values(by='percent_completed_hs', ascending=True)

# lowest_graduation_state = df_pct_completed_hs_sorted.iloc[0]
# highest_graduation_state = df_pct_completed_hs_sorted.iloc[-1]

# print(f"State with the lowest high school graduation rate: {lowest_graduation_state['Geographic Area']} ({lowest_graduation_state['percent_completed_hs']}%)")
# print(f"State with the highest high school graduation rate: {highest_graduation_state['Geographic Area']} ({highest_graduation_state['percent_completed_hs']}%)")

# fig = px.bar(df_pct_completed_hs_sorted, x='Geographic Area', y='percent_completed_hs', 
#              title='High School Graduation Rate by US State (Lowest to Highest)',
#              labels={'percent_completed_hs': 'High School Graduation Rate (%)', 'Geographic Area': 'US State'},
#              height=600)
# fig.show()

df_pct_poverty.fillna(0, inplace=True)
df_pct_poverty.drop_duplicates(inplace=True)
df_pct_completed_hs.fillna(0, inplace=True)
df_pct_completed_hs.drop_duplicates(inplace=True)

df_merged = pd.merge(df_pct_poverty, df_pct_completed_hs, on='Geographic Area')
df_long = df_merged.melt(id_vars='Geographic Area', value_vars=['poverty_rate', 'percent_completed_hs'],
                         var_name='indicator', value_name='value')

# Create dual-axis line chart using plotly.graph_objects
fig = go.Figure()

# Add Poverty Rate trace
fig.add_trace(go.Scatter(
    x=df_merged['Geographic Area'],
    y=df_merged['poverty_rate'],
    name='Poverty Rate',
    yaxis='y1',
    line=dict(color='firebrick')
))

# Add High School Graduation Rate trace
fig.add_trace(go.Scatter(
    x=df_merged['Geographic Area'],
    y=df_merged['percent_completed_hs'],
    name='High School Graduation Rate',
    yaxis='y2',
    line=dict(color='royalblue')
))

# Update layout for dual axes
fig.update_layout(
    title='Poverty Rate vs High School Graduation Rate by US State',
    xaxis=dict(title='US State'),
    yaxis=dict(
        title='Poverty Rate (%)',
        titlefont=dict(color='firebrick'),
        tickfont=dict(color='firebrick'),
        side='left'
    ),
    yaxis2=dict(
        title='High School Graduation Rate (%)',
        titlefont=dict(color='royalblue'),
        tickfont=dict(color='royalblue'),
        overlaying='y',
        side='right'
    )
)

fig.show()