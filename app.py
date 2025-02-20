import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set style for plots
plt.style.use('dark_background')

# Load Dataset
st.title("🍽️ Zomato Data Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv('zomato.csv')
    return df

df = load_data()
st.write("### Preview of Dataset:")
st.dataframe(df.head())

# Cleaning Data
drop_cols = ['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

def handlerate(value):
    if isinstance(value, str):
        if 'NEW' in value or '-' in value:
            return np.nan
        return float(value.split('/')[0])
    return np.nan

df['rate'] = df['rate'].apply(handlerate)
df['rate'].fillna(df['rate'].mean(), inplace=True)

df.rename(columns={'approx_cost(for two people)': 'Cost2plates', 'listed_in(type)': 'Type'}, inplace=True)
df['Cost2plates'] = df['Cost2plates'].astype(str).apply(lambda x: float(x.replace(',', '')) if ',' in x else float(x))
df.drop(columns=['listed_in(city)'], inplace=True, errors='ignore')

if 'location' in df.columns:
    df['location'].fillna('Unknown', inplace=True)  # Fill NaN values
    location_counts = df['location'].value_counts()
    locations_lessthan300 = set(location_counts[location_counts < 300].index)
    df['location'] = df['location'].apply(lambda x: 'others' if x in locations_lessthan300 else x)
else:
    st.warning("⚠️ 'location' column not found.")


# Sidebar Selection
option = st.sidebar.selectbox("Choose Analysis:", [
    "Count Plot of Various Locations",
    "Visualizing Online Order",
    "Visualizing Book Table",
    "Visualizing Online Order vs Rate",
    "Visualizing Book Table vs Rate",
    "Visualizing Online Order Facility, Location Wise",
    "Visualizing Book Table Facility, Location Wise",
    "Visualizing Types of Restaurants vs Rate",
    "Grouping Types of Restaurants, Location Wise",
    "Visualizing No. of Votes, Location Wise",
    "Visualizing Top Cuisines",
    "Correlation Heatmap of Features",
    "Top 10 Restaurants Based on Rating & Votes"
])

# Visualizations
if option == "Count Plot of Various Locations":
    st.subheader("📍 Count of Restaurants Across Locations")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(y=df['location'], order=df['location'].value_counts().index, palette='coolwarm', ax=ax)
    ax.set_title("Number of Restaurants by Location")
    st.pyplot(fig)

elif option == "Visualizing Online Order":
    st.subheader("📦 Online Order Availability")
    fig, ax = plt.subplots()
    sns.countplot(x=df['online_order'], palette='coolwarm', ax=ax)
    ax.set_title("Online Order Availability")
    st.pyplot(fig)

elif option == "Visualizing Book Table":
    st.subheader("📅 Table Booking Availability")
    fig, ax = plt.subplots()
    sns.countplot(x=df['book_table'], palette='coolwarm', ax=ax)
    ax.set_title("Table Booking Availability")
    st.pyplot(fig)

elif option == "Visualizing Online Order vs Rate":
    st.subheader("📊 Online Order vs Rating")
    fig, ax = plt.subplots()
    sns.boxplot(x='online_order', y='rate', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Impact of Online Ordering on Rating")
    st.pyplot(fig)

elif option == "Visualizing Book Table vs Rate":
    st.subheader("📊 Book Table vs Rating")
    fig, ax = plt.subplots()
    sns.boxplot(x='book_table', y='rate', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Impact of Table Booking on Rating")
    st.pyplot(fig)

elif option == "Visualizing Online Order Facility, Location Wise":
    st.subheader("🏙️ Online Order Facility Across Locations")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(y='location', hue='online_order', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Online Order Availability by Location")
    st.pyplot(fig)

elif option == "Visualizing Book Table Facility, Location Wise":
    st.subheader("🏙️ Book Table Facility Across Locations")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(y='location', hue='book_table', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Table Booking Availability by Location")
    st.pyplot(fig)

elif option == "Visualizing Types of Restaurants vs Rate":
    st.subheader("🍽️ Types of Restaurants vs Rating")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Type', y='rate', data=df, palette='coolwarm', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

elif option == "Grouping Types of Restaurants, Location Wise":
    st.subheader("🍽️ Types of Restaurants by Location")
    st.dataframe(df.groupby(['location', 'Type']).size().reset_index(name='count'))

elif option == "Visualizing No. of Votes, Location Wise":
    st.subheader("🏙️ Number of Votes Across Locations")
    top_votes = df.groupby('location')['votes'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=top_votes.index, x=top_votes.values, palette='coolwarm', ax=ax)
    ax.set_title("Top 10 Locations by Votes")
    st.pyplot(fig)

elif option == "Visualizing Top Cuisines":
    st.subheader("🍜 Top 10 Most Popular Cuisines")
    top_cuisines = df['cuisines'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(y=top_cuisines.index, x=top_cuisines.values, palette='coolwarm', ax=ax)
    ax.set_title("Top 10 Cuisines")
    st.pyplot(fig)

elif option == "Correlation Heatmap of Features":
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

elif option == "Top 10 Restaurants Based on Rating & Votes":
    st.subheader("🏆 Top 10 Restaurants by Rating & Votes")
    top_restaurants = df.nlargest(10, ['rate', 'votes'])[['name', 'rate', 'votes']]
    st.dataframe(top_restaurants)

st.success("✅ Data Analysis Completed!")
