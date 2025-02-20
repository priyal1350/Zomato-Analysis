#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')

# ### Loading Data
df = pd.read_csv('zomato.csv')
df.head()

# ### Checking Data Types
df.dtypes

# ### Shape of the Dataset
df.shape

# ### Column Names
df.columns

# ### Dropping Unnecessary Columns
df = df.drop(['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list'], axis=1)
df.head()

# ### Data Info
df.info()

# ### Dropping Duplicates
df.drop_duplicates(inplace=True)
df.shape

# ### Cleaning Rate Column
df['rate'].unique()

# ### Removing "NEW", "-" and "/5" from Rate Column
def handlerate(value):
    if value == 'NEW' or value == '-':
        return np.nan
    else:
        return float(str(value).split('/')[0])

df['rate'] = df['rate'].apply(handlerate)
df['rate'].head()

# ### Filling Null Values in Rate Column with Mean
df['rate'].fillna(df['rate'].mean(), inplace=True)

# ### Dropping Null Values
df.dropna(inplace=True)

# ### Renaming Columns
df.rename(columns={'approx_cost(for two people)': 'Cost2plates', 'listed_in(type)': 'Type'}, inplace=True)

# ### Dropping Redundant Column
df = df.drop(['listed_in(city)'], axis=1)

# ### Cleaning Cost2plates Column
def handlecomma(value):
    return float(str(value).replace(',', '')) if ',' in str(value) else float(value)

df['Cost2plates'] = df['Cost2plates'].apply(handlecomma)

# ### Cleaning Rest Type Column
rest_types = df['rest_type'].value_counts()
rest_types_lessthan1000 = rest_types[rest_types < 1000]

df['rest_type'] = df['rest_type'].apply(lambda x: 'others' if x in rest_types_lessthan1000 else x)

# ### Cleaning Location Column
location = df['location'].value_counts()
location_lessthan300 = location[location < 300]

df['location'] = df['location'].apply(lambda x: 'others' if x in location_lessthan300 else x)

# ### Cleaning Cuisines Column
cuisines = df['cuisines'].value_counts()
cuisines_lessthan100 = cuisines[cuisines < 100]

df['cuisines'] = df['cuisines'].apply(lambda x: 'others' if x in cuisines_lessthan100 else x)

# ## Data is Clean, Let's Jump to Visualization

# ### Count Plot of Various Locations
plt.figure(figsize=(16,10))
sns.countplot(x=df['location'])
plt.xticks(rotation=90)
plt.title("Number of Restaurants Across Locations")
plt.xlabel("Location")
plt.ylabel("Count")
plt.show()

# ### Visualizing Online Order
plt.figure(figsize=(6,6))
sns.countplot(x=df['online_order'], palette='inferno')
plt.title("Online Order Availability")
plt.show()

# ### Visualizing Book Table
plt.figure(figsize=(6,6))
sns.countplot(x=df['book_table'], palette='rainbow')
plt.title("Book Table Availability")
plt.show()

# ### Visualizing Online Order vs Rate
plt.figure(figsize=(6,6))
sns.boxplot(x='online_order', y='rate', data=df)
plt.title("Online Order vs Rating")
plt.show()

# ### Visualizing Book Table vs Rate
plt.figure(figsize=(6,6))
sns.boxplot(x='book_table', y='rate', data=df)
plt.title("Book Table vs Rating")
plt.show()

# ### Visualizing Online Order Facility, Location Wise
df1 = df.groupby(['location', 'online_order'])['name'].count().unstack()
df1.plot(kind='bar', figsize=(15,8))
plt.title("Online Order Availability Across Locations")
plt.xlabel("Location")
plt.ylabel("Count")
plt.show()

# ### Visualizing Book Table Facility, Location Wise
df2 = df.groupby(['location', 'book_table'])['name'].count().unstack()
df2.plot(kind='bar', figsize=(15,8))
plt.title("Book Table Facility Across Locations")
plt.xlabel("Location")
plt.ylabel("Count")
plt.show()

# ### Visualizing Types of Restaurants vs Rate
plt.figure(figsize=(14, 8))
sns.boxplot(x='Type', y='rate', data=df, palette='inferno')
plt.title("Types of Restaurants vs Rating")
plt.xticks(rotation=90)
plt.show()

# ### 📌 Grouping Types of Restaurants, Location Wise
df3 = df.groupby(['location', 'Type'])['name'].count().unstack()

plt.figure(figsize=(15, 8))
df3.plot(kind='bar', stacked=True, figsize=(15,8), colormap='coolwarm')
plt.title("Types of Restaurants Across Locations", fontsize=14)
plt.xlabel("Location", fontsize=12)
plt.ylabel("Number of Restaurants", fontsize=12)
plt.xticks(rotation=90)
plt.legend(title="Restaurant Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### 📌 Visualizing No. of Votes, Location Wise
plt.figure(figsize=(15,8))
sns.boxplot(x="location", y="votes", data=df, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Number of Votes Distribution Across Locations", fontsize=14)
plt.xlabel("Location", fontsize=12)
plt.ylabel("No. of Votes", fontsize=12)
plt.show()

# ### 📌 Visualizing Top Cuisines
top_cuisines = df['cuisines'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
plt.title("Top 10 Most Popular Cuisines", fontsize=14)
plt.xlabel("Number of Restaurants", fontsize=12)
plt.ylabel("Cuisines", fontsize=12)
plt.show()

# ### 📌 Correlation Heatmap of Features
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt=".2f", linewidths=0.5, annot_kws={"size": 12})
plt.title("Correlation Heatmap of Features", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# ### 📌 Top 10 Restaurants Based on Rating & Votes
top_restaurants = df[['name', 'rate', 'votes']].sort_values(by=['rate', 'votes'], ascending=[False, False]).head(10)
print("Top 10 Restaurants Based on Rating & Votes:\n", top_restaurants)