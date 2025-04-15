
# app.py — Streamlit Dashboard for NYC Green Taxi (May 2022)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="NYC Green Taxi Dashboard - May 2022", layout="wide")
st.title("🚖 NYC Green Taxi Data - May 2022 Dashboard")

# Data Loader
@st.cache_data
def load_data():
    df = pd.read_parquet("green_tripdata_2022-05.parquet")
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[df['trip_duration'] > 0]
    df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
    df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("📌 Filter Data")
selected_weekday = st.sidebar.selectbox("Select Weekday", ['All'] + sorted(df['weekday'].unique()))
distance_range = st.sidebar.slider("Trip Distance (miles)", 0.0, float(df['trip_distance'].max()), (0.0, 10.0))

# Optional filter
if 'payment_type' in df.columns:
    selected_payment = st.sidebar.selectbox("Select Payment Type", ['All'] + sorted(df['payment_type'].dropna().unique()))
else:
    selected_payment = "All"

# Apply Filters
filtered_df = df.copy()
if selected_weekday != "All":
    filtered_df = filtered_df[filtered_df['weekday'] == selected_weekday]
if selected_payment != "All" and 'payment_type' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['payment_type'] == selected_payment]
filtered_df = filtered_df[(filtered_df['trip_distance'] >= distance_range[0]) & 
                          (filtered_df['trip_distance'] <= distance_range[1])]

st.markdown(f"### Showing {len(filtered_df)} filtered trips")

# 1. Trip Duration Histogram
st.subheader("⏱️ Trip Duration Distribution")
fig1 = px.histogram(filtered_df, x="trip_duration", nbins=40, color_discrete_sequence=["#00cc96"],
                    title="Distribution of Trip Duration (in Minutes)")
st.plotly_chart(fig1, use_container_width=True)

# 2. Trip Count by Weekday
st.subheader("📅 Trip Count by Weekday")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.countplot(data=filtered_df, x='weekday', order=filtered_df['weekday'].value_counts().index, ax=ax2)
ax2.set_title("Number of Trips per Weekday")
ax2.set_ylabel("Trip Count")
ax2.set_xlabel("Weekday")
plt.xticks(rotation=45)
st.pyplot(fig2)

# 3. Trip Count by Hour of Day
st.subheader("🕐 Trip Count by Hour of Day")
fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.countplot(data=filtered_df, x='hourofday', order=sorted(filtered_df['hourofday'].unique()), ax=ax3)
ax3.set_title("Number of Trips per Hour")
ax3.set_ylabel("Trip Count")
ax3.set_xlabel("Hour (0-23)")
st.pyplot(fig3)

# 4. Pie Charts: Payment Type and Trip Type
if 'payment_type' in filtered_df.columns or 'trip_type' in filtered_df.columns:
    st.subheader("🥧 Payment and Trip Type Distributions")
    cols = st.columns(2)
    if 'payment_type' in filtered_df.columns:
        with cols[0]:
            st.plotly_chart(px.pie(filtered_df, names='payment_type', title='Payment Type Distribution'), use_container_width=True)
    if 'trip_type' in filtered_df.columns:
        with cols[1]:
            st.plotly_chart(px.pie(filtered_df, names='trip_type', title='Trip Type Distribution'), use_container_width=True)

# 5. Correlation Heatmap
st.subheader("📉 Correlation Heatmap")
numeric_cols = ['trip_distance', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                'improvement_surcharge', 'congestion_surcharge', 'trip_duration', 
                'passenger_count', 'total_amount']
corr = filtered_df[numeric_cols].corr()
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

# 6. Average Total Amount by Weekday
st.subheader("💵 Average Total Amount by Weekday")
if 'total_amount' in filtered_df.columns:
    avg_total = filtered_df.groupby('weekday')['total_amount'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    st.plotly_chart(px.bar(avg_total, title="Avg Total Amount by Weekday", labels={'value': 'Avg Total ($)', 'index': 'Weekday'}), use_container_width=True)

# 7. Model Training and Evaluation
st.subheader("📈 Predictive Modeling")

features = ['trip_distance', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 
            'improvement_surcharge', 'congestion_surcharge', 'passenger_count']
target = 'total_amount'

# Only include rows without NA in selected features
model_df = filtered_df.dropna(subset=features + [target])
X = model_df[features]
y = model_df[target]

model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest Regressor"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_type == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | NYC Green Taxi Trip Analysis - May 2022")
