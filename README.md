# CafeSpot Recommender

**Course:** Introduction to Machine Learning / Data Science  
**Instructor:** Dr. Armida P. Salazar  
**Group Name:** Byte & Brew  
**Authors & Roles:**
* **Rosemarie Ann P. Gailo** — Author & Developer
* **Moira N. Mercado** — Author & Researcher
* **Angel Mhae D. Pardillo** — Author & Researcher
* **Pauleen Mae C. Rodriguez** — Author & Data Analyst

**Live Deployed Application:** [https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app](https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app)  
**GitHub Repository:** [https://github.com/Rosegailo/CafeSpot.git](https://github.com/Rosegailo/CafeSpot.git)  

---

## ☕ Project Overview
**CafeSpot Recommender** is an end-to-end machine learning-driven decision support system designed to evaluate prospective urban café location suitability. By analyzing geographic coordinates (`Latitude`, `Longitude`) and customer review density (`NumReview`), the system predicts whether a proposed site will achieve a **Top Rated (≥ 4.45)**, **Moderate Rated (4.05 - 4.45)**, or **Standard/Low Rated (< 4.05)** operational outcome.

The project incorporates Exploratory Data Analysis (EDA), 5-fold stratified cross-validation model comparison in **Orange Data Mining** across three permitted algorithms (**Support Vector Machine [SVM]**, **k-Nearest Neighbors [kNN]**, and **Random Forest**), and an interactive **Streamlit + Folium GIS** web application.

---

## 📁 Repository Structure
```text
Byte_and_Brew_CafeSpot_Recommender/
├── app/                  # Streamlit web application files
│   └── app.py            # Main interactive dashboard script with input validation
├── data/
│   ├── original/         # Original raw dataset (340 records)
│   │   └── Data Test - Sheet1.csv
│   └── processed/        # Cleaned and discretized dataset (326 records)
│       └── cleaned_cafes.csv
├── notebooks/            # Python scripts for EDA & preprocessing
│   └── eda_and_preprocessing.py
├── src/                  # Model training and load testing scripts
│   ├── model_training.py
│   └── test_load_models.py
├── models/               # Saved clean Orange model artifacts (.pkcls)
│   ├── clean_SVM_model.pkcls          # Top Clean Orange SVM Model
│   ├── clean_KNN_model.pkcls          # Clean Orange kNN Model
│   └── clean_RandomForest_model.pkcls # Clean Orange Random Forest Model
├── documentation/        # Generated EDA plots and figures
│   └── plots/
├── paper/                # Final academic research paper & deliverables
│   ├── Byte & Brew CafeSpot Recommender.docx # Editable Word Document Paper
│   └── documentation.md                      # Full IMRaD IEEE research report
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
Run the automated verification script to test loading and predictions across all clean saved Orange `.pkcls` models:
```bash
python src/test_load_models.py
```

### **5. Launch the Interactive Streamlit Web App**
```bash
python -m streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Model Performance & Comparison (5-Fold Stratified Cross-Validation)

| Algorithm | Model File | AUC | CA (Accuracy) | Precision | F1-Score | MCC | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Support Vector Machine (SVM)** | `clean_SVM_model.pkcls` | **0.635** 🏆 | **0.442** 🏆 | **0.431** 🏆 | 0.399 | **0.174** 🏆 | 🏆 **Selected Best Model** |
| **k-Nearest Neighbors (kNN)** | `clean_KNN_model.pkcls` | 0.619 | 0.405 | 0.408 | **0.405** 🏆 | 0.108 | Evaluated & Saved |
| **Random Forest Classifier** | `clean_RandomForest_model.pkcls` | 0.595 | 0.387 | 0.388 | 0.386 | 0.081 | Evaluated & Saved |

---

## 💡 Application User Guide & Input Validation

1. **Model Configuration (Sidebar):** Select between **Clean Orange SVM**, **Clean Orange kNN**, and **Clean Orange Random Forest** from the dropdown menu.
2. **Café Parameters (Sidebar):**
   * `Latitude` & `Longitude`: Candidate site geographical coordinates (Default: `12.971600`, `77.594600` - Bengaluru).
   * `Expected Number of Reviews`: Projected review traffic count (e.g., `150`).
3. **Input Over-typing & Boundary Indicators:**
   * The sidebar and main dashboard feature live validation indicators. If latitude is outside $12.0 - 14.0$, longitude is outside $77.0 - 78.0$, or reviews exceed expected dataset max ($>30,000$), an automatic warning badge alerts the user.
4. **Prediction Dashboard:** Displays the predicted rating bracket, confidence level percentage, interactive Folium GIS map marker, and top 5 nearest reference venues from the historical dataset.

---

## ⚠️ Model Limitations & Troubleshooting

* **Geographical Scope:** The model is calibrated specifically for urban metropolitan commercial zones in Bengaluru, India.
* **Feature Boundaries:** Predictor variables are constrained to spatial coordinates, postal code, and review density. Financial overhead, rent costs, and indoor capacity are not included.
* **Troubleshooting Command Not Found:** If running `streamlit` in PowerShell yields a `CommandNotFoundException`, use `python -m streamlit run app/app.py`.
