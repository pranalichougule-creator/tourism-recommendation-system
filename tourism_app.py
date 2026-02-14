import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

st.title("Tourism Recommendation System")

st.write("Select your preferences to get top 5 destination recommendations")

# Load dataset
df = pd.read_excel("model_data_with_activity_type.xlsx")

# Select features
features = df[["Activity_Type", "Best_Season", "cost in category"]].copy()

# Encode categorical data
enc_activity = LabelEncoder()
enc_season = LabelEncoder()
enc_budget = LabelEncoder()

features["Activity_Type"] = enc_activity.fit_transform(features["Activity_Type"])
features["Best_Season"] = enc_season.fit_transform(features["Best_Season"])
features["cost in category"] = enc_budget.fit_transform(features["cost in category"])

# KNN model
knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(features)

# User input
activity = st.selectbox("Select Activity Type", df["Activity_Type"].unique())
season = st.selectbox("Select Season", df["Best_Season"].unique())
budget = st.selectbox("Select Budget", df["cost in category"].unique())

# Button
if st.button("Get Recommendations"):
    user_input = [[
        enc_activity.transform([activity])[0],
        enc_season.transform([season])[0],
        enc_budget.transform([budget])[0]
    ]]
    
    distances, indices = knn.kneighbors(user_input)
    result = df.iloc[indices[0]]
    
    st.subheader("Top 5 Recommended Destinations")
    st.dataframe(result[[
        "Destination",
        "State",
        "Activity_Type",
        "Best_Season",
        "Avg_Cost (₹)"
    ]])
