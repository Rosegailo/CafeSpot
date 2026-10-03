# Byte & Brew: CafeSpot
### A Clustering-Based Recommender for Optimal Café Selection

Unsupervised K-Means Clustering model for optimal café selection based on spatial proximity, ratings, and popularity.

## Project Structure
```text
GROUPNAME_PROJECTTITLE/
├── app/                  # Streamlit application files
│   └── app.py
├── data/
│   ├── original/         # Original dataset and source note
│   └── processed/        # Cleaned dataset
│       └── cleaned_cafes.csv
├── notebooks/            # EDA and model development notebooks
├── src/                  # Reusable preprocessing and modeling code
├── models/               # Saved best model or complete pipeline
│   ├── kmeans_model.pkl
│   └── scaler.pkl
├── documentation/        # Technical documentation and screenshots
├── paper/                # Final DOCX and PDF
├── README.md             # Setup and run instructions
└── requirements.txt      # Exact software dependencies
```

## Setup & Running Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run EDA & Preprocessing:**
   This generates visualizations in `documentation/plots/` and cleaned data.
   ```bash
   python notebooks/eda_and_preprocessing.py
   ```

3. **Train & Compare Models:**
   This evaluates 3 algorithms (Logistic Regression, KNN, Random Forest) and saves the best model.
   ```bash
   python src/model_training.py
   ```

4. **Run Streamlit App locally:**
   ```bash
   streamlit run app/app.py
   ```

## Model Details
- **Task:** Binary Classification (Predicting 'Top Rated' vs 'Standard')
- **Target:** Rating ≥ 4.3
- **Features:** Latitude, Longitude, Number of Reviews
- **Algorithms Compared:** Logistic Regression, K-Nearest Neighbors, Random Forest Classifier.
- **Best Model:** Random Forest (based on F1-score).
