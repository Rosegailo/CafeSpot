import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Load Original Dataset
data_path = "data/original/Data Test - Sheet1.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit()

df = pd.read_csv(data_path)

# Ensure plots output directory exists
plots_dir = "documentation/plots"
os.makedirs(plots_dir, exist_ok=True)

print("--- 1. Raw Dataset Overview ---")
print(f"Original Dimensions: {df.shape}")

# 2. Preprocessing & Outlier Filtering (Bengaluru Spatial Coordinates)
df = df[(df['Latitude'] >= 12.0) & (df['Latitude'] <= 14.0)]
df = df[(df['Longitude'] >= 77.0) & (df['Longitude'] <= 78.0)]

# Drop duplicates
df = df.drop_duplicates(subset=['Company Name', 'Address']).copy()

# Discretize Rating Target (Class 0: <4.05, Class 1: 4.05-4.45, Class 2: >=4.45)
df['Target'] = 0
df.loc[df['Rating'] >= 4.05, 'Target'] = 1
df.loc[df['Rating'] >= 4.45, 'Target'] = 2

print(f"Cleaned Dataset Dimensions: {df.shape}")

# ---------------------------------------------------------
# Plot 1: Rating Distribution Histogram
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], kde=True, bins=15, color='teal')
plt.title('1. Rating Distribution Histogram')
plt.xlabel('Rating (1.0 - 5.0)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{plots_dir}/1_rating_distribution.png", dpi=300)
plt.close()
print("Saved 1_rating_distribution.png")

# ---------------------------------------------------------
# Plot 2: Class Distribution Bar Chart
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x='Target', hue='Target', palette='viridis', legend=False)
plt.title('2. Operational Rating Target Class Distribution')
plt.xlabel('Target Class (0: <4.05 | 1: 4.05-4.45 | 2: >=4.45)')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig(f"{plots_dir}/2_class_distribution.png", dpi=300)
plt.close()
print("Saved 2_class_distribution.png")

# ---------------------------------------------------------
# Plot 3: Reviews vs Rating Scatter Plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='NumReview', y='Rating', hue='Target', palette='deep', alpha=0.8, s=70)
plt.title('3. Customer Reviews vs Rating')
plt.xlabel('Number of Reviews (NumReview)')
plt.ylabel('Rating')
plt.tight_layout()
plt.savefig(f"{plots_dir}/3_reviews_vs_rating.png", dpi=300)
plt.close()
print("Saved 3_reviews_vs_rating.png")

# ---------------------------------------------------------
# Plot 4: Feature Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_cols = ['Rating', 'NumReview', 'Pin Code', 'Latitude', 'Longitude', 'Target']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title('4. Feature Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(f"{plots_dir}/4_correlation_heatmap.png", dpi=300)
plt.close()
print("Saved 4_correlation_heatmap.png")

# ---------------------------------------------------------
# Plot 5: Clean Geographic Distribution Plot
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x='Longitude',
    y='Latitude',
    hue='Target',
    size='NumReview',
    sizes=(30, 250),
    palette='Set2',
    alpha=0.85
)
plt.title('5. Geographic Distribution of Cafes in Bengaluru')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{plots_dir}/5_geo_distribution.png", dpi=300)
plt.close()
print("Saved 5_geo_distribution.png")

# ---------------------------------------------------------
# Save Clean Processed Dataset
# ---------------------------------------------------------
df.to_csv("data/processed/cleaned_cafes.csv", index=False)
print("✅ Cleaned processed dataset saved to data/processed/cleaned_cafes.csv")
