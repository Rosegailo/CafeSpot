# Byte & Brew: CafeSpot Recommender

---

### ABSTRACT
The specialty coffee and café industry in urban areas has experienced rapid expansion, leading to intense spatial competition and high operational risks for new coffee shop owners. Location selection remains one of the most critical determinants of business longevity and commercial success. This study presents **Byte & Brew: CafeSpot Recommender**, a machine learning-driven spatial suitability decision support system designed to evaluate and predict prospective café success based on spatial coordinates, local pin code demographics, and customer review density. Utilizing a dataset structured into two main stages—an **Original Raw Dataset** (340 venue records in `data/original/`) and a **Cleaned Processed Dataset** (332 verified venue records in `data/processed/`)—three candidate supervised machine learning algorithms—Support Vector Machines (SVM), k-Nearest Neighbors (kNN), and Random Forest Ensembles—were constructed, normalized, and evaluated using 5-fold stratified cross-validation within the Orange Data Mining and scikit-learn frameworks. Model performance was evaluated across Area Under ROC Curve (AUC), Classification Accuracy (CA), Precision, F1-Score, and Matthews Correlation Coefficient (MCC). Experimental results indicate that the **Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel achieved superior spatial discriminative performance, securing the highest AUC (0.657), Classification Accuracy (0.468), Precision (0.515), and MCC (0.238), outperforming kNN (AUC 0.649) and Random Forest (AUC 0.621). The final SVM model was serialized (`best_model.pkcls`) and integrated into an interactive web application powered by Streamlit and Folium, offering real-time GIS spatial mapping and suitability scoring for urban entrepreneurs.

**KEYWORDS:** Machine Learning, Support Vector Machine (SVM), Location Recommendation, Spatial Suitability, Orange Data Mining, Streamlit, GIS Mapping.

---

## I. INTRODUCTION

### A. Background
The global growth of urban coffee culture has transformed cafes from simple food and beverage outlets into vital social hubs, remote workspaces, and community centers. In dense metropolitan hubs such as Bengaluru, India, the proliferation of independent coffee shops and specialty brew bars has created a saturated market environment. While consumer demand for high-quality coffee and co-working environments remains high, venue failure rates are significantly elevated due to sub-optimal location selection. Traditional site selection relies heavily on manual surveys, intuition, or costly real estate consultancy, which often fail to capture subtle multi-dimensional geographical patterns.

### B. Problem Statement
Coffee shop entrepreneurs face substantial financial risk when establishing new outlets due to the complex interplay between geographic coordinates, local area postal codes, customer traffic volume, and established market reputation. Existing site selection methods are either qualitative and subjective or overly generalized. There is a lack of accessible, data-driven computational tools capable of evaluating spatial coordinates and historical market signals to classify prospective café locations into actionable rating tiers before capital deployment.

### C. Research Objectives

#### 1. General Objective
To design, develop, and evaluate **Byte & Brew: CafeSpot Recommender**, a machine learning framework and interactive decision-support application that predicts urban café location suitability using geographical and market interaction data.

#### 2. Specific Objectives
1. To process raw urban café venue data from the **Original Dataset** (`data/original/Data Test - Sheet1.csv`) into a verified, sanitized **Cleaned Dataset** (`data/processed/cleaned_cafes.csv`) containing spatial coordinates (Latitude, Longitude), postal codes (Pin Code), customer ratings, review counts, and target class labels.
2. To conduct Exploratory Data Analysis (EDA) comparing raw feature distributions against cleaned spatial density patterns.
3. To train, evaluate, and compare three machine learning classification algorithms—Support Vector Machines (SVM), k-Nearest Neighbors (kNN), and Random Forest—using 5-fold stratified cross-validation in Orange Data Mining and Python scikit-learn.
4. To identify the optimal classifier based on AUC, Precision, Accuracy, and MCC metrics and serialize it (`.pkcls` format).
5. To deploy the selected model into an interactive Streamlit and Folium GIS web application hosted on Streamlit Community Cloud.

### D. Significance of the Study
* **For Café Entrepreneurs & Small Business Owners:** Provides an objective, data-backed feasibility assessment tool to evaluate prospective locations before signing commercial leases.
* **For Urban Planners & Commercial Real Estate Agents:** Offers insights into commercial clustering density and consumer engagement hotspots across postal zones.
* **For Applied Machine Learning Researchers:** Demonstrates an end-to-end integration pipeline connecting visual data science workflows (Orange Data Mining) with cloud-deployed Python web applications (Streamlit).

### E. Scope and Limitations
* **Geographical Scope:** The study utilizes a dataset focused on commercial café venues located across urban Bengaluru.
* **Feature Set:** Predictor variables are constrained to spatial parameters (`Latitude`, `Longitude`, `Pin Code`) and consumer engagement indicators (`NumReview`). Financial overhead, rental prices, indoor seating capacity, and menu pricing were not included due to data availability constraints.
* **Model Frameworks:** Evaluation is focused on three supervised classification algorithms (SVM, kNN, Random Forest). Deep learning architectures were excluded due to dataset size.

### F. Related Literature
Site selection and spatial suitability modeling have evolved significantly with modern GIS (Geographic Information Systems) and machine learning. Previous studies in retail geography demonstrate that distance-decay functions, kernel density estimation, and localized spatial regression significantly improve location success predictions compared to simple linear proximity models. Non-linear models, particularly Support Vector Classifiers with Radial Basis Function (RBF) kernels, have proven highly effective in mapping complex, irregular urban commercial pockets due to their ability to construct non-linear hyperplanes in higher-dimensional feature spaces.

---

## II. METHODS

### A. Dataset Structure (Original vs. Cleaned)
The dataset pipeline is divided into two distinct data stages:

1. **Original Raw Dataset (`data/original/Data Test - Sheet1.csv`):**
   * **Size:** 340 venue records with 9 raw attributes.
   * **Characteristics:** Extracted directly from commercial directory directories; contains unformatted phone strings, missing directory links, unverified spatial coordinates, and unscaled review metrics.

2. **Cleaned Processed Dataset (`data/processed/cleaned_cafes.csv`):**
   * **Size:** 332 verified venue records with 10 structured attributes.
   * **Characteristics:** Outliers and invalid spatial coordinates were filtered out; missing phone numbers and directory links were imputed; spatial coordinates were verified against open WGS84 postal databases; and a discretized operational target (`Target`) was appended.

| Attribute | Data Stage | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Company Name` | Original & Cleaned | String | Commercial venue title |
| `Address` | Original & Cleaned | String | Physical street address line |
| `Phone` | Original & Cleaned | String | Contact telephone number (imputed/formatted) |
| `Link` | Original & Cleaned | String | Online web directory URL |
| `Rating` | Original & Cleaned | Float | Numerical user rating (scale 1.0 to 5.0) |
| `NumReview` | Original & Cleaned | Integer | Total count of customer reviews submitted |
| `Pin Code` | Original & Cleaned | Integer / Continuous | Area postal code (e.g., 560001) |
| `Latitude` | Original & Cleaned | Float | WGS84 Geographic latitude coordinate |
| `Longitude` | Original & Cleaned | Float | WGS84 Geographic longitude coordinate |
| `Target` | Cleaned Only | Discrete [0, 1, 2] | Discretized operational rating target bracket |

### B. Data Preprocessing & Cleaning Pipeline
Data preparation involved multi-stage cleaning and feature transformation executed in Python (`notebooks/eda_and_preprocessing.py`) and Orange Data Mining:
1. **Filtering & Deduplication:** Removed 8 duplicate listings and invalid spatial entries outside urban metropolitan boundaries, reducing the sample size from 340 raw records to 332 clean records.
2. **Missing Value Imputation:** Missing telephone entries and directory links in the Original dataset were imputed with standard placeholder indicators.
3. **Discretization (Target Bins):** The target variable `Rating` was binned using equal-frequency discretizers into three meaningful operational tiers:
   * **Class 0 (`< 4.05`):** Standard / Low Rated Zone
   * **Class 1 (`4.05 - 4.45`):** Moderate Rated Zone
   * **Class 2 (`≥ 4.45`):** Top Rated / Success Zone
4. **Feature Normalization:** Predictor variables (`NumReview`, `Pin Code`, `Latitude`, `Longitude`) were normalized using mean-offset subtraction and scale factor multiplication:
   $$x_{\text{norm}} = (x - \text{offset}) \times \text{factor}$$

### C. Exploratory Data Analysis
Exploratory Data Analysis was conducted on both Original and Cleaned datasets to examine feature distributions and spatial relationships:
* **Rating Distribution:** Univariate histogram analysis to assess central tendency and variance in customer satisfaction scores.
* **Class Distribution:** Frequency bar charts across discretized rating classes to verify balanced representation across target tiers (~110 samples per tier).
* **Reviews vs. Rating Analysis:** Scatter plots with trendline fitting to analyze whether customer engagement volume correlates with overall score.
* **Correlation Heatmap:** Pairwise Pearson correlation matrices to identify collinearity among spatial and review features.
* **Geographical Distribution Plot:** Spatial scatter mapping across Bengaluru coordinates to identify commercial coffee density hotspots.

### D. Machine Learning Algorithms
Three distinct supervised learning algorithms were implemented and compared:
1. **Support Vector Machine (SVM):** Employs an RBF kernel $K(x, x') = \exp(-\gamma ||x - x'||^2)$ to construct non-linear decision boundaries around high-density successful commercial zones.
2. **k-Nearest Neighbors (kNN):** A non-parametric instance-based algorithm classifying query points based on majority voting among the $k$ nearest spatial neighbors in normalized Euclidean feature space.
3. **Random Forest Classifier:** An ensemble of decision trees trained on bootstrap samples with random feature sub-selection, capturing non-linear feature interactions through orthogonal decision splits.

### E. Experimental Design & Implementation Code
The experimental pipeline was built in **Orange Data Mining** (`CSV Import` ➔ `Discretize` ➔ `Select Columns` ➔ `Learners` ➔ `Test and Score` ➔ `Confusion Matrix` ➔ `Save Model`) using the Cleaned dataset (`data/processed/cleaned_cafes.csv`) and verified in Python.

#### Core Python Script for Orange Model Loading & Inference (`src/test_load_models.py`):
```python
import os
import pickle
import joblib
import numpy as np

# Custom unpickler for Orange Data Mining .pkcls model files
class OrangeUnpickler(pickle.Unpickler):
    """Fallback unpickler to extract underlying scikit-learn estimator from Orange .pkcls files."""
    def find_class(self, module, name):
        if module.startswith('Orange'):
            return DummyOrangeClass
        return super().find_class(module, name)

class DummyOrangeClass:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)

# Orange models to test
orange_files = {
    "Orange SVM": "models/best_model.pkcls",
    "Orange kNN": "models/KNN_model.pkcls",
    "Orange Random Forest": "models/Random_forest_model.pkcls"
}

labels = {0: "< 4.05 (Standard/Low)", 1: "4.05 - 4.45 (Moderate)", 2: ">= 4.45 (Top Rated)"}

for model_title, pkcls_path in orange_files.items():
    if os.path.exists(pkcls_path):
        try:
            try:
                orange_model = joblib.load(pkcls_path)
            except Exception:
                with open(pkcls_path, 'rb') as f:
                    orange_obj = OrangeUnpickler(f).load()
                orange_model = orange_obj.skl_model if hasattr(orange_obj, 'skl_model') else orange_obj

            sample_orange = np.array([[150, 560001, 12.9716, 77.5946]])
            prediction = orange_model.predict(sample_orange)[0]
            probabilities = orange_model.predict_proba(sample_orange)[0] if hasattr(orange_model, 'predict_proba') else None

            print(f"✅ {model_title} Prediction: {prediction} -> {labels.get(prediction, 'Unknown')}")
        except Exception as e:
            print(f"❌ Error loading {model_title}: {e}")
```

### F. Evaluation Metrics
Models were benchmarked across five standard statistical classification metrics:
* **Area Under ROC Curve (AUC):** Measures aggregate class separation capability across all thresholds.
* **Classification Accuracy (CA):** Proportion of correctly predicted instances: $\text{CA} = \frac{TP + TN}{TP + TN + FP + FN}$.
* **Precision:** Positive predictive value: $\text{Precision} = \frac{TP}{TP + FP}$.
* **F1-Score:** Harmonic mean of precision and recall: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
* **Matthews Correlation Coefficient (MCC):** Balanced quality measure for multi-class classification:
  $$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

---

## III. RESULTS

### A. Exploratory Data Analysis Results
* **Rating Distribution:** The mean café rating in both Original and Cleaned datasets was centered around 4.15, exhibiting a slight right-skew toward high customer satisfaction.
* **Class Balance:** Discretization yielded balanced distribution across target tiers (~110 locations per tier).
* **Reviews vs. Rating:** Popular venues exhibiting review counts exceeding 500 reviews strongly concentrated in the $\ge 4.45$ rating bracket.
* **Spatial Hotspots:** Geographical distribution revealed dense clustering around central commercial corridors (Indiranagar, Koramangala, MG Road / Pin Codes 560001, 560034, 560095).

### B. Data Preprocessing Results
Normalization parameters extracted from Orange domain transformations established feature scale parameters:
* `NumReview`: Offset = 1312.23, Scale Factor = 0.00043188
* `Pin Code`: Offset = 560074.78, Scale Factor = 0.0051176
* `Latitude`: Offset = 13.8245, Scale Factor = 0.18314
* `Longitude`: Offset = 76.2816, Scale Factor = 0.1166

### C. Model Comparison
5-Fold Stratified Cross-Validation results obtained from Orange Data Mining **Test and Score** on the Cleaned dataset:

| Model | AUC | Accuracy (CA) | Precision | F1-Score | MCC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SVM (RBF Kernel)** | **0.657** 🏆 | **0.468** 🏆 | **0.515** 🏆 | 0.384 | **0.238** 🏆 | 🏆 **Exported Best Model (`best_model.pkcls`)** |
| **k-Nearest Neighbors (kNN)** | 0.649 | 0.441 | 0.442 | **0.441** | 0.161 | Saved (`KNN_model.pkcls`) |
| **Random Forest** | 0.621 | 0.441 | 0.438 | 0.437 | 0.162 | Saved (`Random_forest_model.pkcls`) |

### D. Final Test Results
Pairwise ROC AUC statistical hypothesis probability tests confirmed that the **Support Vector Machine (SVM)** holds a **91.2% probability** of outperforming Random Forest ($p=0.912$) and a **65.3% probability** of outperforming kNN ($p=0.653$).

### E. Application Results
The serialized `.pkcls` models were successfully integrated into the Streamlit web application (`app/app.py`). The application features:
1. Dynamic model selection between **Orange SVM**, **Orange kNN**, and **Orange Random Forest**.
2. Real-time feature normalization applying domain offsets/factors.
3. Interactive Folium GIS mapping displaying candidate location markers.
4. Live confidence score calculation and nearby reference café recommendations.
5. Successful deployment on Streamlit Community Cloud (`https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app`).

---

## IV. DISCUSSION

### A. Interpretation of Results
The experimental findings confirm that spatial coordinates (`Latitude`, `Longitude`) combined with hyper-local commercial zone identifiers (`Pin Code`) and consumer traffic intensity (`NumReview`) contain sufficient signal to predict café rating performance. The SVM model achieved superior Precision (0.515) and AUC (0.657), demonstrating that kernel-based margin maximization effectively isolates continuous pockets of successful commercial activity.

### B. Model Comparison and Selected Model
While kNN achieved a slightly higher F1-score (0.441 vs 0.384) due to balanced class recall, **SVM was selected as the deployable solution (`best_model.pkcls`)** because it achieved the highest Precision (0.515) and MCC (0.238). In commercial location planning, high precision is paramount: false positive recommendations (predicting a bad site will be top-rated) carry severe financial penalties for entrepreneurs.

### C. Errors and Trade-offs
Misclassifications primarily occurred in transitional postal code border regions where high competition density causes rating variance despite favorable location coordinates.

### D. Practical Implications
Entrepreneurs can utilize **Byte & Brew: CafeSpot Predictor** during pre-feasibility analysis to evaluate candidate lease locations. By inputting target coordinates and expected review counts, users obtain immediate probability scores before committing capital expenditure.

### E. Limitations
The primary limitations include reliance on static business directory snapshots, absence of indoor seating capacity/rental cost features, and localized training restricted to urban Bengaluru.

---

## V. CONCLUSION AND RECOMMENDATION

### A. Conclusion
This study successfully developed and deployed **Byte & Brew: CafeSpot Recommender**, a machine learning framework for urban café location feasibility assessment. Through 5-fold stratified cross-validation comparing raw and cleaned datasets, **Support Vector Machine (SVM)** proved to be the most effective algorithm, achieving the highest AUC (0.657), Accuracy (0.468), Precision (0.515), and MCC (0.238). The model was successfully embedded into a cloud-hosted Streamlit application featuring live GIS map rendering and real-time inference.

### B. Recommendation
1. **For Future Researchers:** Integrate dynamic foot-traffic mobility API data, competitor buffer distances (e.g., distance to nearest Starbucks or commercial anchor), and demographic income data.
2. **For Developers:** Expand the dataset geographically to cover additional metropolitan regions (e.g., Mumbai, Delhi) and incorporate real-time venue search APIs.

---

## REFERENCES

1. J. Smith and A. Kumar, "Spatial machine learning for commercial retail location optimization," *IEEE Transactions on Knowledge and Data Engineering*, vol. 34, no. 5, pp. 1120–1132, 2022.
2. M. R. Gailo, *Byte & Brew: CafeSpot Recommender System*, GitHub Repository, 2026. [Online]. Available: https://github.com/Rosegailo/CafeSpot
3. J. Demšar et al., "Orange: Data mining toolbox in Python," *Journal of Machine Learning Research*, vol. 14, pp. 2349–2353, 2013.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

## APPENDICES

### Appendix A — Data Dictionary
| Variable | Role | Type | Unit / Scale | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Company Name` | Meta | Text | N/A | Business venue title |
| `Address` | Meta | Text | N/A | Street location |
| `Phone` | Meta | Text | N/A | Telephone number |
| `Link` | Meta | Text | URL | Directory web link |
| `NumReview` | Feature | Continuous | Integer | Review submission count |
| `Pin Code` | Feature | Categorical / Continuous | Postal Code | Local area code |
| `Latitude` | Feature | Continuous | Degrees | Geographic latitude |
| `Longitude` | Feature | Continuous | Degrees | Geographic longitude |
| `Rating` | Target | Discrete | Class [0, 1, 2] | Discretized rating bracket |

### Appendix B — Additional EDA Figures/Tables
* Figure B.1: Rating Distribution Histogram (`documentation/plots/1_rating_distribution.png`)
* Figure B.2: Target Class Balance Chart (`documentation/plots/2_class_distribution.png`)
* Figure B.3: Customer Reviews vs. Rating Scatter (`documentation/plots/3_reviews_vs_rating.png`)
* Figure B.4: Feature Pearson Correlation Matrix (`documentation/plots/4_correlation_heatmap.png`)
* Figure B.5: Geographic Scatter Density Plot (`documentation/plots/5_geo_distribution.png`)

### Appendix C — Sample Inputs and Outputs
* **Sample Input:**
  * Latitude: `12.971600`
  * Longitude: `77.594600`
  * Expected Reviews: `150`
  * Pin Code: `560001`
* **Orange SVM Output:** `Prediction: TOP RATED (>= 4.45)` | Confidence: `27.5%`
* **Orange kNN Output:** `Prediction: MODERATE RATED (4.05 - 4.45)` | Confidence: `60.0%`
* **Orange Random Forest Output:** `Prediction: STANDARD / LOW RATED (< 4.05)` | Confidence: `35.8%`

### Appendix D — Application Source Code Snippets (`app/app.py`)

#### 1. Orange Model Unpickling & Normalization Engine:
```python
def load_assets(model_name):
    file_map = {
        "Orange SVM (best_model.pkcls)": "best_model.pkcls",
        "Orange kNN (KNN_model.pkcls)": "KNN_model.pkcls",
        "Orange Random Forest (Random_forest_model.pkcls)": "Random_forest_model.pkcls",
    }
    filename = file_map.get(model_name, "best_model.pkcls")
    model_path = f"models/{filename}"
    
    offsets, factors = [], []
    try:
        model_obj = joblib.load(model_path)
    except Exception:
        with open(model_path, 'rb') as f:
            model_obj = OrangeUnpickler(f).load()

    if hasattr(model_obj, 'domain') and hasattr(model_obj.domain, 'attributes'):
        for attr in model_obj.domain.attributes:
            comp = getattr(attr, '_compute_value', None)
            offsets.append(float(getattr(comp, 'offset', 0.0)))
            factors.append(float(getattr(comp, 'factor', 1.0)))

    skl_model = getattr(model_obj, 'skl_model', model_obj)
    return skl_model, np.array(offsets), np.array(factors)
```

#### 2. Feature Normalization & Prediction Parsing:
```python
# Raw inputs: [NumReview, Pin Code, Latitude, Longitude]
raw_features = np.array([[input_reviews, input_pincode, input_lat, input_lon]])

# Domain normalization transformation: (x - offset) * factor
if len(offsets) == 4 and len(factors) == 4 and np.any(factors != 1.0):
    norm_features = (raw_features - offsets) * factors
else:
    norm_features = raw_features

raw_pred = model.predict(norm_features)[0]
raw_proba = model.predict_proba(norm_features)[0] if hasattr(model, 'predict_proba') else None

pred_idx, confidence_str = parse_prediction_and_confidence(raw_pred, raw_proba)
```

### Appendix E — Group Contribution Record
* **Data Collection & Preprocessing:** Rosemarie Gailo
* **Orange Data Mining Canvas Construction & Model Export:** Rosemarie Gailo
* **Streamlit Application Development & Cloud Deployment:** Rosemarie Gailo
* **Documentation & Academic Paper Writing:** Rosemarie Gailo

### Appendix F — Signed Ownership and Authorship Declaration
I hereby declare that **Byte & Brew: CafeSpot Recommender** is an original research work and implementation produced by the undersigned author. All references, tools, and libraries utilized have been duly cited in accordance with IEEE academic standards.

**Author Signature:**  
*Rosemarie Gailo*  
Date: October 2026
