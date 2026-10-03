import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")

# 1. Load Data
data_path = "data/original/Data Test - Sheet1.csv"
df = pd.read_csv(data_path)

# Create output directory for plots if it doesn't exist
plots_dir = "documentation/plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# OUTLIER REMOVAL: Filter out the data typos (Latitude > 20 or Longitude < 70 are wrong for Bangalore)
df = df[(df['Latitude'] >= 12.0) & (df['Latitude'] <= 14.0)]
df = df[(df['Longitude'] >= 77.0) & (df['Longitude'] <= 78.0)]

print("--- 1. Dataset Overview After Clean Spatial Filtering ---")
print(f"Dimensions: {df.shape}")

# Define Target Variable for Classification
df['Target'] = (df['Rating'] >= 4.3).astype(int)

# Visualization 5: Clean Geographic Distribution by Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Longitude',
    y='Latitude',
    hue='Target',
    size='NumReview',
    sizes=(20, 200),
    palette='coolwarm',
    alpha=0.7
)
plt.title('Geographic Distribution of Cafes in Bangalore (Cleaned Spatial Data)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{plots_dir}/5_geo_distribution.png", dpi=300)
plt.close()

# Save clean processed data
df.to_csv("data/processed/cleaned_cafes.csv", index=False)
print("Processed data saved cleanly to data/processed/cleaned_cafes.csv")
