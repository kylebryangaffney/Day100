import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

pd.options.display.float_format = '{:,.2f}'.format

# Load the datasets
df_hh_income = pd.read_csv('Median_Household_Income_2015.csv', encoding="windows-1252")
df_pct_poverty = pd.read_csv('Pct_People_Below_Poverty_Level.csv', encoding="windows-1252")
df_pct_completed_hs = pd.read_csv('Pct_Over_25_Completed_High_School.csv', encoding="windows-1252")
df_share_race_city = pd.read_csv('Share_of_Race_By_City.csv', encoding="windows-1252")
df_fatalities = pd.read_csv('Deaths_by_Police_US.csv', encoding="windows-1252")

def get_dataframe_info(df):
    """
    Returns basic information about a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    tuple: Shape, columns, number of NaN values, and number of duplicate rows.
    """
    shape = df.shape
    columns = df.columns.tolist()
    nan_values = df.isna().sum()
    duplicates = df.duplicated().sum()
    return shape, columns, nan_values, duplicates

# Gather information about each DataFrame
info_hh_income = get_dataframe_info(df_hh_income)
info_pct_poverty = get_dataframe_info(df_pct_poverty)
info_pct_completed_hs = get_dataframe_info(df_pct_completed_hs)
info_share_race_city = get_dataframe_info(df_share_race_city)
info_fatalities = get_dataframe_info(df_fatalities)

# Store DataFrame info in a dictionary for easy access
dfs_info = {
    "df_hh_income": info_hh_income,
    "df_pct_poverty": info_pct_poverty,
    "df_pct_completed_hs": info_pct_completed_hs,
    "df_share_race_city": info_share_race_city,
    "df_fatalities": info_fatalities
}

# Print DataFrame information
for df_name, info in dfs_info.items():
    print(f"DataFrame: {df_name}")
    print(f"Shape: {info[0]}")
    print(f"Columns: {info[1]}")
    print(f"NaN Values:\n{info[2]}")
    print(f"Duplicate Rows: {info[3]}\n")

# Process and clean household income data
df_hh_income['Median Income'] = pd.to_numeric(df_hh_income['Median Income'], errors='coerce')
df_hh_income.dropna(inplace=True)

# Process and clean poverty data
df_pct_poverty['poverty_rate'] = pd.to_numeric(df_pct_poverty['poverty_rate'], errors='coerce')
df_pct_poverty.dropna(inplace=True)
df_poverty_rate = df_pct_poverty.groupby('Geographic Area')['poverty_rate'].mean().sort_values(ascending=False).reset_index()

# Plot poverty rates by US state
plt.figure(figsize=(20, 8))
plt.bar(df_poverty_rate['Geographic Area'], df_poverty_rate['poverty_rate'])
plt.xlabel("US State")
plt.ylabel("Poverty Rate")
plt.title("Poverty Rates in US States")
plt.grid(axis='y', alpha=0.8)
plt.show()

# Process and clean high school graduation data
df_pct_completed_hs['percent_completed_hs'] = pd.to_numeric(df_pct_completed_hs['percent_completed_hs'], errors='coerce')
df_pct_completed_hs.dropna(inplace=True)
df_hs_graduation_rate = df_pct_completed_hs.groupby('Geographic Area')['percent_completed_hs'].mean().sort_values().reset_index()

# Plot high school graduation rates by US state
plt.figure(figsize=(20, 8))
plt.plot(df_hs_graduation_rate['Geographic Area'], df_hs_graduation_rate['percent_completed_hs'], linestyle='-', marker='x', markersize=5, linewidth=1)
plt.xlabel("US State")
plt.ylabel("High School Graduation Rate")
plt.title("High School Graduation Rates in US State")
plt.grid(True)
plt.show()

# Merge poverty and high school graduation data
df_merged_poverty_graduation = df_poverty_rate.merge(df_hs_graduation_rate, on='Geographic Area')
df_merged_poverty_graduation.sort_values('percent_completed_hs', ascending=False, inplace=True)

# Plot comparison of poverty rates and high school graduation rates
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

# KDE plot comparing poverty rates and high school graduation rates
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

# Linear regression plot between poverty rates and high school graduation rates
plt.figure(figsize=(24, 10))
sns.regplot(data=df_merged_poverty_graduation, x="percent_completed_hs", y="poverty_rate")

plt.xlabel("High School Graduation Rate", fontsize=18)
plt.ylabel("Poverty Rate", fontsize=18)
plt.title('Linear Regression Between Poverty Rates and High School Graduation Rates', fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

# Process and clean racial composition data
cols = ['share_white', 'share_black','share_native_american', 'share_asian', 'share_hispanic']
df_share_race_city[cols] = df_share_race_city[cols].apply(pd.to_numeric, errors='coerce')
df_share_race_city.dropna(inplace=True)
df_state_races = df_share_race_city.groupby('Geographic area')[cols].mean().reset_index()
df_state_races.rename(columns={'share_white': 'White', 'share_black': 'Black', 'share_native_american': 'Native American', 'share_asian': 'Asian', 'share_hispanic': 'Hispanic'}, inplace=True)

# Plot racial makeup of each US state
fig = px.bar(df_state_races,
    x="Geographic area",
    y=['White', 'Black', 'Native American', 'Asian', 'Hispanic'],
    title="Racial Makeup of Each US State",
    labels={"value": "Population Percentage by Race", 'variable': 'Race', 'Geographic area': 'State'},
    barmode="stack",
)
fig.update_xaxes(tickangle=0)
fig.show()

# Process and plot fatalities by race
df_fatalities_by_race = df_fatalities[df_fatalities['race'].notna()]
df_fatalities_by_race = df_fatalities_by_race['race'].value_counts()

fig = px.pie(names=df_fatalities_by_race.index,
             values=df_fatalities_by_race.values,
             title="Deaths by Race",
             hole=0.4)

fig.update_traces(textfont_size=15, labels=['White', 'Black', 'Native American', 'Asian', 'Hispanic'])
fig.show()

# Plot fatalities by gender
df_deaths_by_gender = df_fatalities['gender'].value_counts()

fig = px.bar(df_deaths_by_gender, x=df_deaths_by_gender.index,
             y=df_deaths_by_gender.values,
             color=df_deaths_by_gender.index,
             title="Deaths by Gender")
fig.update_xaxes(title_text='Gender', tickvals=[0, 1], ticktext=['MEN', 'WOMEN'])
fig.update_yaxes(title_text='Death Count')
fig.show()

# Plot manner of death compared to gender and age
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

# Analyze and plot armed/unarmed data
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

# Process age data for fatalities
df_fatalities['age'] = pd.to_numeric(df_fatalities['age'], errors='coerce')
df_fatalities['race'].replace({'W': 'White', 'B': 'Black', 'N': 'Native American', 'A': 'Asian', 'H': 'Hispanic', 'O': 'Other'}, inplace=True)
df_deaths_by_age = df_fatalities.dropna(subset=['age'])

deaths_under_25 = df_deaths_by_age[df_deaths_by_age['age'] < 25].shape[0]
total_deaths = df_deaths_by_age.shape[0]
percentage_under_25 = round((deaths_under_25 / total_deaths) * 100)

print(f'People killed under 25 years old: {percentage_under_25}%')

# Plot KDE of age for fatalities
plt.figure(figsize=(10, 6))
sns.histplot(df_deaths_by_age, x='age', kde=True, color='blue', alpha=0.2)
plt.xlabel('Age')
plt.ylabel('Deaths')
plt.title('Kernel Density Estimation of Age')
plt.xticks(range(0, 101, 10))
plt.show()

# Plot KDE of age by race for fatalities
graph = sns.FacetGrid(df_deaths_by_age, col="race")
graph.map(sns.histplot, 'age', kde=True, color='blue', alpha=0.2)
graph.set_axis_labels('Age', 'Deaths')
graph.set_titles('KDE of Age - Race: {col_name}')
plt.show()

# Plot total number of deaths by race
deaths_by_race = df_fatalities[df_fatalities['race'].notna()]
deaths_by_race = deaths_by_race['race'].value_counts()
plt.figure(figsize=(20, 8))
plt.plot(deaths_by_race, linestyle='-', marker='x', markersize=5, linewidth=1)
plt.xlabel("Race")
plt.ylabel("Deaths")
plt.title("Total Number of Deaths by Race")
plt.grid(True)
plt.yticks(range(0, deaths_by_race.max() + 100, 100))
plt.show()

# Calculate and print percentage of deaths with signs of mental illness
deaths_with_mental_illness = df_fatalities['signs_of_mental_illness'].sum() 
total_deaths = df_fatalities['signs_of_mental_illness'].count()
death_percentage_with_mental_illness = deaths_with_mental_illness/total_deaths * 100
print(f"Percentage of deaths with signs of mental illness: {round(death_percentage_with_mental_illness)}%")

# Plot top 10 cities by police-caused deaths
top_10_cities_with_most_deaths = df_fatalities[['state', 'city']].value_counts().head(10).reset_index(name='count')

fig = px.bar(top_10_cities_with_most_deaths, x='city', y='count')
fig.update_layout(
    title="Top 10 Cities by Police Caused Deaths",
    xaxis_title="City",
    yaxis_title="Total Deaths in Each City",
)
fig.show()

# Analyze racial composition in top 10 cities with most deaths
merged_df = top_10_cities_with_most_deaths.merge(df_fatalities, on=['state', 'city'])
merged_df = merged_df.groupby(['state', 'city', 'count'])['race'].value_counts(dropna=False).reset_index(name='death_race')
merged_df['death_race'] = round((merged_df['death_race'] / merged_df['count']) * 100)
merged_df = merged_df[merged_df['race'].isin(["Asian", "Black", "Hispanic", "Native American", "White"])]
merged_df.dropna(subset=['death_race'], inplace=True)

# Extracting cities and their racial composition from df_share_race_city
cities = '|'.join(top_10_cities_with_most_deaths['city'].tolist())
top_10_cities_race = df_share_race_city[df_share_race_city['City'].str.contains(cities, case=False)]
for city in top_10_cities_with_most_deaths['city']:
    city_match_mask = top_10_cities_race['City'].str.contains(city, case=False)
    top_10_cities_race.loc[city_match_mask, 'City'] = city

top_10_cities_race = top_10_cities_race.groupby(['Geographic area', 'City']).mean().reset_index()
top_10_cities_race = top_10_cities_race.merge(top_10_cities_with_most_deaths, left_on=['Geographic area', 'City'], right_on=['state', 'city'])
top_10_cities_race.drop(['Geographic area', 'City', 'count'], axis=1, inplace=True)

# Melting the dataframe for easier plotting
long_df = top_10_cities_race.melt(id_vars=["state", "city"], var_name="race", value_name="race_share")
race_mapping = {
    "share_white": "White",
    "share_black": "Black",
    "share_native_american": "Native American",
    "share_asian": "Asian",
    "share_hispanic": "Hispanic"
}
long_df['race'] = long_df['race'].replace(race_mapping)

# Merging death_race information back into long_df
long_df = long_df.merge(merged_df[['state', 'city', 'race', 'death_race']], on=['state', 'city', 'race'], how='left')
long_df.dropna(subset=['race_share', 'death_race'], inplace=True)

# Plotting racial composition vs death rates in top 10 cities
for city in top_10_cities_with_most_deaths['city']:
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=long_df[long_df['city'] == city], x='race', y='race_share', ax=ax, color='skyblue', alpha=0.7)
    for i in range(long_df[long_df['city'] == city].shape[0]):
        race_share = long_df[long_df['city'] == city]['race_share'].iloc[i]
        if np.isfinite(race_share):
            ax.text(i, race_share + 1, f"{race_share:.1f}%", ha='center', color='blue')
    
    ax2 = ax.twinx()
    sns.lineplot(data=long_df[long_df['city'] == city], x='race', y='death_race', ax=ax2, color='red', alpha=0.7, marker='o', linewidth=2, label='Death Rate')
    for i in range(long_df[long_df['city'] == city].shape[0]):
        death_race = long_df[long_df['city'] == city]['death_race'].iloc[i]
        if np.isfinite(death_race):
            ax2.text(i, death_race + 1, f"{death_race:.1f}%", ha='center', color='red')
    
    ax2.grid(None)
    ax.set_xlabel("Race", fontsize=14)
    ax.set_ylabel("Race Share (%)", fontsize=14)
    ax2.set_ylabel("Death by Race (%)", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
    
    plt.title(f'City: {city}', fontsize=16)
    plt.tight_layout()
    plt.show()

# Analyze and plot police killings over time
deaths_over_time = df_fatalities.groupby('date').size().reset_index(name='count').sort_values('date')
deaths_over_time['year'] = deaths_over_time['date'].dt.year
deaths_over_time['month'] = deaths_over_time['date'].dt.month
deaths_over_time = deaths_over_time.groupby(['year', 'month'])['count'].sum().reset_index()

plt.figure(figsize=(20, 8), facecolor='white')
plt.plot(deaths_over_time.index, deaths_over_time['count'], marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=6)
plt.xlabel('Year-Month', fontsize=14)
plt.ylabel('Number of Deaths', fontsize=14)
plt.title('Number of Police Killings Over Time', fontsize=18, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(deaths_over_time.index, deaths_over_time.apply(lambda x: f"{x['year']}-{x['month']:02d}", axis=1), rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()

for i, row in deaths_over_time.iterrows():
    plt.text(i, row['count'] + 0.2, row['count'], ha='center', va='bottom', fontsize=10, color='black')

plt.show()
