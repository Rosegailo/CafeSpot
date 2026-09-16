# CafeSpot - Café Recommendation Engine

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

2. **Run Streamlit App locally:**
   ```bash
   streamlit run app/app.py
   ```
