# Byte & Brew: CafeSpot Recommender

**Course:** Machine Learning / Data Science  
**Instructor:** Dr. Armida P. Salazar  
**Author:** Rosemarie Gailo  
**Live Deployed Application:** [https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app](https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app)  

---

## ☕ Project Overview
**Byte & Brew: CafeSpot Recommender** is an end-to-end machine learning-driven decision support system designed to evaluate prospective urban café location suitability. By analyzing geographic coordinates (Latitude, Longitude), postal codes (Pin Code), and customer review density (`NumReview`), the system predicts whether a proposed site will achieve a **Top Rated (≥ 4.45)**, **Moderate Rated (4.05 - 4.45)**, or **Standard/Low Rated (< 4.05)** operational outcome.

The project incorporates Exploratory Data Analysis (EDA), 5-fold stratified cross-validation model comparison in **Orange Data Mining** across three permitted algorithms (**Support Vector Machine (SVM)**, **k-Nearest Neighbors (kNN)**, and **Random Forest**), and an interactive **Streamlit + Folium GIS** web application.

---

## 📁 Repository Structure
```text
Byte_and_Brew_CafeSpot_Recommender/
├── app/                  # Streamlit web application files
│   └── app.py            # Main interactive dashboard script
├── data/
│   ├── original/         # Original raw dataset (340 records)
│   │   └── Data Test - Sheet1.csv
│   └── processed/        # Cleaned and discretized dataset (332 records)
│       └── cleaned_cafes.csv
├── notebooks/            # Python scripts for EDA & preprocessing
│   └── eda_and_preprocessing.py
├── src/                  # Model training and load testing scripts
│   ├── model_training.py
│   └── test_load_models.py
├── models/               # Saved model artifacts (.pkcls & .pkl)
│   ├── best_model.pkcls   # Top Orange SVM Model
│   ├── KNN_model.pkcls    # Orange kNN Model
│   └── Random_forest_model.pkcls # Orange Random Forest Model
├── documentation/        # Generated EDA plots and figures
│   └── plots/
├── paper/                # Final academic documentation paper
│   └── documentation.md  # Complete IMRaD IEEE research report
├── README.md             # Setup, running instructions & project guide
└── requirements.txt      # Exact Python dependencies
```

---

## 🚀 Setup and Running Instructions

### **1. Clone the Repository & Prepare Virtual Environment**
```bash
git clone https://github.com/Rosegailo/CafeSpot.git
cd CafeSpot
python -m venv venv
```

Activate the virtual environment:
* **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
* **Mac/Linux:** `source venv/bin/activate`

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run Exploratory Data Analysis & Preprocessing**
```bash
python notebooks/eda_and_preprocessing.py
```

### **4. Verify Model Artifacts**
Run the automated verification script to test loading and predictions across all saved Orange `.pkcls` models:
```bash
python src/test_load_models.py
```

### **5. Launch the Interactive Streamlit Web App**
```bash
python -m streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Model Performance & Comparison (5-Fold Cross-Validation)

| Algorithm | Model File | AUC | CA (Accuracy) | Precision | F1-Score | MCC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Support Vector Machine (SVM)** | `best_model.pkcls` | **0.657** 🏆 | **0.468** 🏆 | **0.515** 🏆 | 0.384 | **0.238** 🏆 | 🏆 **Selected Best Model** |
| **k-Nearest Neighbors (kNN)** | `KNN_model.pkcls` | 0.649 | 0.441 | 0.442 | **0.441** | 0.161 | Evaluated & Saved |
| **Random Forest Classifier** | `Random_forest_model.pkcls` | 0.621 | 0.441 | 0.438 | 0.437 | 0.162 | Evaluated & Saved |

---

## 💡 Application User Guide

1. **Model Configuration (Sidebar):** Select between **Orange SVM**, **Orange kNN**, and **Orange Random Forest** from the dropdown menu.
2. **Café Parameters (Sidebar):**
   * `Latitude` & `Longitude`: Candidate site geographical coordinates (Default: `12.971600`, `77.594600` - Bengaluru).
   * `Expected Number of Reviews`: Projected review traffic count (e.g., `150`).
   * `Pin Code`: Area postal code (e.g., `560001`).
3. **Prediction Dashboard:** Displays the predicted rating bracket, confidence level percentage, interactive Folium GIS map marker, and top 5 nearest reference venues from the historical dataset.

---

## ⚠️ Model Limitations & Troubleshooting

* **Geographical Scope:** The model is calibrated specifically for urban metropolitan commercial zones in Bengaluru, India.
* **Feature Boundaries:** Predictor variables are constrained to spatial coordinates, postal code, and review density. Financial overhead, rent costs, and indoor capacity are not included.
* **Troubleshooting Command Not Found:** If running `streamlit` in PowerShell yields a `CommandNotFoundException`, use `python -m streamlit run app/app.py`.
