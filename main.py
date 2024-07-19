import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# This might be helpful:
from collections import Counter

pd.options.display.float_format = '{:,.2f}'.format

df_hh_income = pd.read_csv('Median_Household_Income_2015.csv', encoding="windows-1252")
df_pct_poverty = pd.read_csv('Pct_People_Below_Poverty_Level.csv', encoding="windows-1252")
df_pct_completed_hs = pd.read_csv('Pct_Over_25_Completed_High_School.csv', encoding="windows-1252")
df_share_race_city = pd.read_csv('Share_of_Race_By_City.csv', encoding="windows-1252")
df_fatalities = pd.read_csv('Deaths_by_Police_US.csv', encoding="windows-1252")

def get_dataframe_info(df):
    shape = df.shape
    columns = df.columns.tolist()
    nan_values = df.isna().sum()
    duplicates = df.duplicated().sum()
    return shape, columns, nan_values, duplicates

info_hh_income = get_dataframe_info(df_hh_income)
info_pct_poverty = get_dataframe_info(df_pct_poverty)
info_pct_completed_hs = get_dataframe_info(df_pct_completed_hs)
info_share_race_city = get_dataframe_info(df_share_race_city)
info_fatalities = get_dataframe_info(df_fatalities)


dfs_info = {
    "df_hh_income": info_hh_income,
    "df_pct_poverty": info_pct_poverty,
    "df_pct_completed_hs": info_pct_completed_hs,
    "df_share_race_city": info_share_race_city,
    "df_fatalities": info_fatalities
}

for df_name, info in dfs_info.items():
    print(f"DataFrame: {df_name}")
    print(f"Shape: {info[0]}")
    print(f"Columns: {info[1]}")
    print(f"NaN Values:\n{info[2]}")
    print(f"Duplicate Rows: {info[3]}\n")

df_hh_income['Median Income'] = pd.to_numeric(df_hh_income['Median Income'], errors='coerce')
df_hh_income.dropna(inplace=True)

df_pct_poverty['poverty_rate'] = pd.to_numeric(df_pct_poverty['poverty_rate'], errors='coerce')
df_pct_poverty.dropna(inplace=True)
df_poverty_rate = df_pct_poverty.groupby('Geographic Area')['poverty_rate'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(20, 8))
plt.bar(df_poverty_rate['Geographic Area'], df_poverty_rate['poverty_rate'])
plt.xlabel("US State")
plt.ylabel("Poverty Rate")
plt.title("Poverty Rates in US States")
plt.grid(axis='y', alpha=0.8)
plt.show()

df_pct_completed_hs['percent_completed_hs'] = pd.to_numeric(df_pct_completed_hs['percent_completed_hs'], errors='coerce')
df_pct_completed_hs.dropna(inplace=True)
df_hs_graduation_rate = df_pct_completed_hs.groupby('Geographic Area')['percent_completed_hs'].mean().sort_values().reset_index()

plt.figure(figsize=(20, 8))
plt.plot(df_hs_graduation_rate['Geographic Area'], df_hs_graduation_rate['percent_completed_hs'], linestyle='-', marker='x', markersize=5, linewidth=1)
plt.xlabel("US State")
plt.ylabel("High School Graduation Rate")
plt.title("High School Graduation Rates in US State")
plt.grid(True)
plt.show()

df_merged_poverty_graduation = df_poverty_rate.merge(df_hs_graduation_rate, on='Geographic Area')
df_merged_poverty_graduation.sort_values('percent_completed_hs', ascending=False, inplace=True)

plt.figure(figsize=(20,8))
plt.title('Comparing State Poverty Rates and High School Graduation Rates')

ax = plt.gca()
ax2 = ax.twinx()

ax.plot(df_merged_poverty_graduation['Geographic Area'], df_merged_poverty_graduation['poverty_rate'], label='Poverty Rate', linestyle='-', marker='o', markersize=5, linewidth=1)

ax2.plot(df_merged_poverty_graduation['Geographic Area'], df_merged_poverty_graduation['percent_completed_hs'], color='crimson', label='High School Graduation Rate', linestyle='--', marker='x', markersize=5, linewidth=2)

ax.set_ylabel('Poverty Rate')
ax.set_xlabel('State')
ax2.set_ylabel('High School Graduation Rate')

plt.legend(loc='upper center')
ax.yaxis.grid(False)
ax2.grid(False)
plt.show()
plt.figure(figsize=(24, 10))
ax = sns.jointplot(df_merged_poverty_graduation, x="percent_completed_hs", y="poverty_rate", kind='kde', levels=6, height=10)
ax1 = ax.plot_joint(sns.scatterplot)

ax.fig.suptitle('Kernel Density Estimate (KDE) Comparing Poverty Rates and High School Graduation Rates', y=1.02)

for line in range(0, df_merged_poverty_graduation.shape[0]):
     ax.ax_joint.text(df_merged_poverty_graduation.percent_completed_hs[line], 
                      df_merged_poverty_graduation.poverty_rate[line], 
                      df_merged_poverty_graduation['Geographic Area'][line], 
                      horizontalalignment='left', 
                      size='medium', 
                      color='black', 
                      weight='semibold')

plt.xlabel("High School Graduation Rate")
plt.ylabel("Poverty Rate")

plt.show()


plt.figure(figsize=(24, 10))
sns.regplot(data=df_merged_poverty_graduation, x="percent_completed_hs", y="poverty_rate")

plt.xlabel("High School Graduation Rate", fontsize=18)
plt.ylabel("Poverty Rate", fontsize=18)
plt.title('Linear Regression Between Poverty Rates and High School Graduation Rates', fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

cols = ['share_white', 'share_black','share_native_american', 'share_asian', 'share_hispanic']
df_share_race_city[cols] = df_share_race_city[cols].apply(pd.to_numeric, errors='coerce')
df_share_race_city.dropna(inplace=True)
df_state_races = df_share_race_city.groupby('Geographic area')[cols].mean().reset_index()
df_state_races.rename(columns={'share_white': 'White', 'share_black': 'Black', 'share_native_american': 'Native American', 'share_asian': 'Asian', 'share_hispanic': 'Hispanic'}, inplace=True)

fig = px.bar(df_state_races,
    x="Geographic area",
    y=['White', 'Black', 'Native American', 'Asian', 'Hispanic'],
    title="Racial Makeup of Each US State",
    labels={"value": "Population Percentage by Race", 'variable': 'Race', 'Geographic area': 'State'},
    barmode="stack",
)
fig.update_xaxes(tickangle=0)

fig.show()


df_fatalities_by_race = df_fatalities[df_fatalities['race'].notna()]
df_fatalities_by_race = df_fatalities_by_race['race'].value_counts()

fig = px.pie(names=df_fatalities_by_race.index,
             values=df_fatalities_by_race.values,
             title="Deaths by Race",
             hole=0.4,)

fig.update_traces(textfont_size=15, labels=['White', 'Black', 'Native American', 'Asian', 'Hispanic'])

fig.show()

df_deaths_by_gender = df_fatalities['gender'].value_counts()

fig = px.bar(df_deaths_by_gender, x=df_deaths_by_gender.index,
             y=df_deaths_by_gender.values,
             color=df_deaths_by_gender.index,
             title="Deaths by Gender")
fig.update_xaxes(title_text='Gender', tickvals=[0, 1], ticktext=['MEN', 'WOMEN'])
fig.update_yaxes(title_text='Death Count')

fig.show()

df_age_manner_of_death = df_fatalities[df_fatalities['age'].notna()]
df_age_manner_of_death = df_age_manner_of_death.groupby('gender')[['age', 'manner_of_death']].value_counts().reset_index(name='count')
df_age_manner_of_death['gender'].replace({'M': 'Man', 'F': 'Woman'}, inplace=True)

fig = px.box(df_age_manner_of_death, x='manner_of_death', y='age', color='gender', color_discrete_map={'Woman': 'red', 'Man': 'blue'})
fig.update_layout(
    title="Manner of Death Compared to Gender and Age",
    xaxis_title="Manner of Death",
    yaxis_title="Age",
    legend_title="Gender",
)
fig.show()

df_armed = df_fatalities[df_fatalities['armed'].notna()]
unarmed_percentage = ((df_armed['armed'] == 'unarmed').sum() / df_armed['armed'].value_counts().sum()) * 100
df_people_armed = df_armed['armed'].value_counts()
print(f"Armed people killed: {round(100 - unarmed_percentage)}%")

fig = px.bar(df_people_armed, x=df_people_armed.index, y=df_people_armed.values, log_y=True, color=df_people_armed.values)

fig.update_layout(
    title="Type of Weapon",
    xaxis_title="Weapon",
    yaxis_title="Count",
)

fig.show()

people_with_guns = df_people_armed['gun']
people_unarmed = df_people_armed['unarmed']
print(f'People with guns killed by the police: {people_with_guns}')
print(f'People unarmed killed by the police: {people_unarmed}')