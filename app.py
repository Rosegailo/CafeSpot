import os
import pickle
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# 1. Page Configuration
st.set_page_config(
    page_title="CafeSpot - Café Recommendation Engine",
    page_icon="☕",
    layout="wide",
)


# 2. Asset Loading with Caching
@st.cache_resource
def load_assets():
  # Paths relative to execution directory
  data_path = "data/processed/cleaned_cafes.csv"
  model_path = "models/kmeans_model.pkl"
  scaler_path = "models/scaler.pkl"

  # Fallback for flat directory structures during testing
  if not os.path.exists(data_path):
    data_path = "cleaned_cafes.csv"
    model_path = "kmeans_model.pkl"
    scaler_path = "scaler.pkl"

  df = pd.read_csv(data_path)
  with open(model_path, "rb") as f:
    model = pickle.load(f)
  with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

  # Compute LogReviews and scale features to assign Cluster labels to the data
  df["LogReviews"] = np.log1p(df["NumReview"])
  X = df[["Latitude", "Longitude", "Rating", "LogReviews"]]
  X_scaled = scaler.transform(X)
  df["Cluster"] = model.predict(X_scaled)

  return df, model, scaler


try:
  df_cafes, kmeans_model, scaler = load_assets()
except Exception as e:
  st.error(
      f"Error loading model artifacts: {e}. Please ensure model files exist."
  )
  st.stop()

# 3. Sidebar Inputs & Validation
st.sidebar.header("📍 Preferences & Filters")

user_lat = st.sidebar.number_input(
    "Your Latitude",
    min_value=12.7000,
    max_value=13.3000,
    value=12.9716,
    format="%.6f",
)
user_lon = st.sidebar.number_input(
    "Your Longitude",
    min_value=77.4000,
    max_value=77.8000,
    value=77.5946,
    format="%.6f",
)

min_rating = st.sidebar.slider(
    "Minimum Rating Threshold", min_value=3.0, max_value=5.0, value=4.0, step=0.1
)
max_results = st.sidebar.slider(
    "Max Recommendations to Display",
    min_value=3,
    max_value=20,
    value=5,
)

# 4. Header & Overview
st.title("☕ CafeSpot Recommendation Engine")
st.caption(
    "Unsupervised K-Means Clustering model for optimal café selection based on"
    " spatial proximity, ratings, and popularity."
)

# 5. Model Inference Pipeline
# Map user input to model cluster space
user_log_reviews = np.log1p(df_cafes["NumReview"].median())  # Baseline median
user_features = np.array(
    [[user_lat, user_lon, min_rating, user_log_reviews]]
)
user_scaled = scaler.transform(user_features)
predicted_cluster = kmeans_model.predict(user_scaled)[0]

# Filter candidate cafes
filtered_df = df_cafes[
    (df_cafes["Cluster"] == predicted_cluster)
    & (df_cafes["Rating"] >= min_rating)
].copy()

# Calculate direct distance (Euclidean proxy for ranking)
filtered_df["Distance_Score"] = np.sqrt(
    (filtered_df["Latitude"] - user_lat) ** 2
    + (filtered_df["Longitude"] - user_lon) ** 2
)
recommendations = filtered_df.sort_values(by="Distance_Score").head(
    max_results
)

# 6. Display Dashboard Layout
col1, col2 = st.columns([3, 2])

with col1:
  st.subheader("Interactive Map & Recommended Clusters")

  # Center map at user location
  m = folium.Map(location=[user_lat, user_lon], zoom_start=13)

  # User location pin
  folium.Marker(
      [user_lat, user_lon],
      popup="<b>Your Location</b>",
      icon=folium.Icon(color="red", icon="user", prefix="fa"),
  ).add_to(m)

  # Recommended café pins
  for _, row in recommendations.iterrows():
    folium.Marker(
        [row["Latitude"], row["Longitude"]],
        popup=f"<b>{row['Company Name']}</b><br>Rating: {row['Rating']} ⭐<br>Reviews: {row['NumReview']}",
        tooltip=row["Company Name"],
        icon=folium.Icon(color="blue", icon="coffee", prefix="fa"),
    ).add_to(m)

  st_folium(m, width="100%", height=450)

with col2:
  st.subheader("Model Diagnostic & Summary")
  st.metric(label="Assigned Spatial Cluster", value=f"Cluster #{predicted_cluster}")
  st.metric(
      label="Total Matching Cafés Found", value=len(filtered_df)
  )

  st.write("**Selected Cluster Stats:**")
  cluster_stats = (
      df_cafes[df_cafes["Cluster"] == predicted_cluster][
          ["Rating", "NumReview"]
      ]
      .mean()
      .to_dict()
  )
  st.json(
      {
          "Average Cluster Rating": round(cluster_stats["Rating"], 2),
          "Average Review Count": int(cluster_stats["NumReview"]),
      }
  )

# 7. Data Table View
st.markdown("---")
st.subheader("Top Recommended Locations")
if not recommendations.empty:
  st.dataframe(
      recommendations[
          ["Company Name", "Rating", "NumReview", "Address", "Pin Code"]
      ],
      use_container_width=True,
  )
else:
  st.warning(
      "No cafes matched your strict criteria. Try lowering the rating"
      " threshold."
  )