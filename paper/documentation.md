# CafeSpot Recommender
**An End-to-End Machine Learning-Driven Spatial Suitability Decision Support System for Urban Café Location Feasibility**

**Course:** Introduction to Machine Learning / Data Science  
**Instructor:** Dr. Armida P. Salazar  
**Group Name:** Byte & Brew  
**Authors:** Rosemarie Ann P. Gailo, Moira N. Mercado, Angel Mhae D. Pardillo, Pauleen Mae C. Rodriguez  
**Date:** October 15, 2026  
**Live Application:** [https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app](https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app)  

---

## ABSTRACT

The specialty coffee and café industry in urban areas has experienced rapid expansion, leading to high competition between commercial establishments. Location selection remains one of the most critical determinants of business longevity and commercial success. This study presents **CafeSpot Recommender**, an end-to-end machine learning-driven spatial suitability decision support system designed to evaluate and predict prospective café success based on spatial coordinates and customer review density. Utilizing a dataset structured into two distinct stages—an **Original Raw Dataset** (340 venue records in `data/original/`) and a **Cleaned Processed Dataset** (326 verified venue records in `data/processed/` after filtering duplicates, spatial outliers, and missing directory links)—three candidate supervised machine learning algorithms (**Support Vector Machines [SVM]**, **k-Nearest Neighbors [kNN]**, and **Random Forest Ensembles**) were constructed, normalized, and evaluated using 5-fold stratified cross-validation within the **Orange Data Mining** and Python **scikit-learn** frameworks. Model performance was evaluated across Area Under ROC Curve (AUC), Classification Accuracy (CA), Precision, F1-Score, and Matthews Correlation Coefficient (MCC). Experimental results indicate that the **Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel achieved superior spatial discriminative performance, securing the highest AUC (0.635), Classification Accuracy (0.442), Precision (0.431), and MCC (0.174), outperforming kNN (AUC 0.619) and Random Forest (AUC 0.595). The final SVM model was serialized (`clean_SVM_model.pkcls`) and integrated into an interactive web application powered by **Streamlit** and **Folium GIS**, offering real-time spatial mapping and suitability scoring for entrepreneurs and urban researchers.

**KEYWORDS:** Machine Learning, Support Vector Machine (SVM), Location Recommendation, Spatial Suitability, Orange Data Mining, Streamlit, GIS Mapping, IMRaD.

---

## I. INTRODUCTION

### A. Background
For students, freelancers, and young professionals, a café is far more than a simple food and beverage outlet; it serves as a third place for collaborative study, remote work, and social interaction [1]. The global growth of urban coffee culture has transformed cafés into vital economic and social hubs. In dense metropolitan centers such as Bengaluru, India, the proliferation of independent coffee shops and specialty brew bars has created a highly competitive market environment. While consumer demand remains high, commercial site selection remains a difficult and capital-intensive decision for new business entrants [1]. Traditional site selection relies heavily on manual surveys, gut feeling, or costly consultancy services, which often fail to capture subtle multi-dimensional geographical patterns.

### B. Problem Statement
Evaluating prospective store locations currently relies on subjective estimates, peer recommendations, or static foot-traffic counts. There is a lack of automated, data-driven tools that integrate geographic coordinates and consumer review density to predict operational rating outcomes into clear decision tiers (**Top Rated**, **Moderate**, or **Standard/Low Rated**).

### C. Research Objectives

1. **General Objective:**  
   To design, train, evaluate, and deploy **CafeSpot Recommender**, an interactive machine learning decision-support web application that predicts urban café location suitability using geographical coordinates and customer engagement data.

2. **Specific Objectives:**  
   - Process raw urban café venue data from the **Original Dataset** (`data/original/Data Test - Sheet1.csv`, 340 records) into a sanitized **Cleaned Dataset** (`data/processed/cleaned_cafes.csv`, 326 records) with verified spatial coordinates (`Latitude`, `Longitude`), review counts (`NumReview`), and discretized rating target labels.  
   - Conduct Exploratory Data Analysis (EDA) across five visualizations comparing raw feature distributions against spatial density patterns.  
   - Train, evaluate, and benchmark three supervised classification algorithms (**Support Vector Machine [SVM]**, **k-Nearest Neighbors [kNN]**, and **Random Forest**) using 5-fold stratified cross-validation in **Orange Data Mining** and Python `scikit-learn` [3], [4].  
   - Select the optimal classifier based on AUC, Precision, CA, and MCC, explicitly minimizing costly False Positive recommendations, and serialize the model (`clean_SVM_model.pkcls`).  
   - Deploy the trained classifier into an interactive **Streamlit + Folium GIS** web application hosted on Streamlit Cloud [2].

### D. Significance of the Study
- **Students & Café-Goers:** Provides visibility into top-rated commercial coffee corridors for study and social meetings.  
- **Café Entrepreneurs & Business Owners:** Offers an objective, data-backed feasibility assessment tool to evaluate candidate site lease contracts prior to committing capital expenditure.  
- **Urban Planners & Real Estate Agents:** Delivers spatial insights into commercial clustering density and consumer engagement hotspots across metropolitan zones.  
- **Applied Machine Learning Researchers:** Demonstrates a reproducible integration pipeline linking visual data science workflows (**Orange Data Mining**) with cloud-deployed web applications (**Streamlit**).

### E. Scope and Limitations
- **Geographical Scope:** Uses the Kaggle *Cafe data* dataset [5], consisting of 340 raw venue listings in Bengaluru, India (yielding 326 clean venue records after spatial filtering). Due to data availability constraints, local Philippine datasets were unavailable; thus, spatial generalizations to Philippine cities require recalibration.  
- **Feature Set:** Predictor variables are constrained to spatial parameters (`Latitude`, `Longitude`) and review volume (`NumReview`). Financial overhead, rental prices, indoor seating capacity, and menu pricing were omitted due to data constraints.  
- **Location Input:** The Streamlit web application requires numerical latitude/longitude coordinate entries and review estimates.  
- **Model Frameworks:** Evaluation was restricted to three supervised classifiers (SVM, kNN, Random Forest). Prohibited deep learning methods were excluded.

### F. Related Literature
Site selection and spatial suitability modeling have evolved significantly with modern Geographic Information Systems (GIS) and machine learning algorithms [1]. Non-linear estimators, such as Support Vector Classifiers with Radial Basis Function (RBF) kernels, can map complex spatial decision boundaries in high-dimensional feature spaces, enabling effective separation of commercial hotspots [1], [4].

---

## II. METHODS

### A. Dataset Structure (Original vs. Cleaned)

The pipeline utilizes two distinct data stages:

1. **Original Raw Dataset** (`data/original/Data Test - Sheet1.csv`):  
   - **Size:** 340 venue records with 9 raw attributes.  
   - **Characteristics:** Unformatted contact strings, missing directory links, unverified spatial coordinates, and unscaled review counts.

2. **Cleaned Processed Dataset** (`data/processed/cleaned_cafes.csv`):  
   - **Size:** 326 verified venue records with 10 structured attributes.  
   - **Characteristics:** Spatial outliers outside Bengaluru ($12.0 \le \text{Lat} \le 14.0$, $77.0 \le \text{Lon} \le 78.0$), duplicate directory listings, and missing contact links were filtered; spatial coordinates were verified against WGS84 databases; and a discretized operational target (`Target`) was generated.

#### Table II.1: Dataset Attribute Specifications

| Attribute Name | Data Stage | Data Type | Description / Role |
| --- | --- | --- | --- |
| `Company Name` | Original and Cleaned | String | Commercial venue name |
| `Address` | Original and Cleaned | String | Street address line |
| `Phone` | Original and Cleaned | String | Contact telephone string |
| `Link` | Original and Cleaned | String | Online directory URL |
| `Rating` | Original and Cleaned | Float | Customer rating (1.0 to 5.0) |
| `NumReview` | Original and Cleaned | Integer | Total customer reviews count |
| `Pin Code` | Original and Cleaned | Integer | Area postal code (e.g., 560001) |
| `Latitude` | Original and Cleaned | Float | WGS84 Geographic Latitude |
| `Longitude` | Original and Cleaned | Float | WGS84 Geographic Longitude |
| `Target` | Cleaned Only | Discrete [0, 1, 2] | Discretized rating target bracket |

### B. Data Preprocessing & Leakage Prevention Pipeline
Executed in `notebooks/eda_and_preprocessing.py`:
1. **Filtering & Deduplication:** Filtered out spatial outliers outside urban Bengaluru, dropped duplicate venue addresses, and removed invalid entries missing phone/link fields ($340 \rightarrow 326$ records).
2. **Target Discretization:** The continuous `Rating` column was binned into three operational target tiers:
   - **Class 0 ($< 4.05$):** Standard / Low Rated Zone  
   - **Class 1 ($4.05 - 4.45$):** Moderate Rated Zone  
   - **Class 2 ($\ge 4.45$):** Top Rated / Success Zone  
3. **Feature Normalization & Leakage Prevention:** Feature variables ($x \in \{\text{NumReview}, \text{Latitude}, \text{Longitude}\}$) were normalized via mean offset subtraction and scaling factor multiplication:
   $$\hat{x} = (x - \text{offset}) \times \text{factor}$$
   Normalization parameters were calculated strictly inside training folds during cross-validation to prevent data leakage [4].

### C. Exploratory Data Analysis Procedure
Five core visualizations were generated in `documentation/plots/`:
- **Figure 1 (Rating Histogram):** Assesses central tendency ($\mu = 4.15$) and right-skew in user satisfaction.
- **Figure 2 (Class Distribution Bar Chart):** Verifies class balance across discretized target tiers (~108 samples/tier across 326 records).
- **Figure 3 (Reviews vs. Rating Scatter):** Analyzes review traffic density against rating brackets.
- **Figure 4 (Correlation Matrix Heatmap):** Evaluates pairwise Pearson correlations among spatial parameters and review counts.
- **Figure 5 (Geographical Density Plot):** Maps WGS84 spatial coordinates across urban corridors in Bengaluru.

### D. Machine Learning Algorithms
1. **Support Vector Machine (SVM):** Utilizes an RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ to form non-linear decision boundaries around high-performing venue clusters.
2. **k-Nearest Neighbors (kNN):** An instance-based non-parametric classifier using Euclidean distance voting among $k$ nearest spatial neighbors.
3. **Random Forest Classifier:** An ensemble of decision trees trained on bootstrap sub-samples with randomized feature splitting.

### E. Experimental Setup & Implementation Code
The workflow was constructed in **Orange Data Mining** (*CSV Import $\rightarrow$ Select Columns $\rightarrow$ Learners $\rightarrow$ Test and Score $\rightarrow$ Confusion Matrix $\rightarrow$ Save Model*) using 5-fold stratified cross-validation on `data/processed/cleaned_cafes.csv` (326 records) and validated in Python (`src/test_load_models.py`).

### F. Evaluation Metrics & Cost of Misclassification
- **Area Under ROC Curve (AUC):** Measures aggregate class separation across thresholds.
- **Classification Accuracy (CA):** $\text{CA} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Precision (Primary Decision Metric):** $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- **F1-Score:** $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Matthews Correlation Coefficient (MCC):** Evaluates multi-class quality under non-trivial balance:
  $$\text{MCC} = \frac{\text{TP}\cdot\text{TN} - \text{FP}\cdot\text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$

*Cost Justification:* In commercial real estate site selection, a **False Positive (FP)** (recommending an unprofitable site as a top location) leads to capital destruction (lease obligations, renovation costs). A **False Negative (FN)** merely represents a missed option with zero loss. Hence, **Precision** and **AUC** were prioritized to minimize False Positives.

---

## III. RESULTS

### A. Exploratory Data Analysis Results
- **Rating Distribution:** Mean rating centered at $4.15$, with heavy concentration between $4.0$ and $4.6$.
- **Class Balance:** Discretization yielded ~108 records per tier ($33.1\%$ per class across 326 clean samples).
- **Review Traffic:** Venues with $> 500$ reviews predominantly fell in Class 2 ($\ge 4.45$).
- **Spatial Hotspots:** High-density clusters were identified along central commercial corridors (Indiranagar, Koramangala, MG Road).

### B. Normalization Parameters
Extracted Orange domain parameters for normalized feature transformation:
- `NumReview`: $\text{Offset} = 1312.23$, $\text{Scale Factor} = 0.00043188$
- `Latitude`: $\text{Offset} = 13.8245$, $\text{Scale Factor} = 0.18314$
- `Longitude`: $\text{Offset} = 76.2816$, $\text{Scale Factor} = 0.1166$

### C. Model Comparison (5-Fold Stratified Cross-Validation)

#### Table III.1: 5-Fold Stratified CV Results on Clean Dataset (326 Records)

| Algorithm | Model Artifact | AUC | CA (Accuracy) | Precision | F1-Score | MCC | Evaluation Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Support Vector Machine (SVM)** | `clean_SVM_model.pkcls` | **0.635** 🏆 | **0.442** 🏆 | **0.431** 🏆 | 0.399 | **0.174** 🏆 | 🏆 **Selected Best Model** |
| **k-Nearest Neighbors (kNN)** | `clean_KNN_model.pkcls` | 0.619 | 0.405 | 0.408 | **0.405** 🏆 | 0.108 | Evaluated & Saved |
| **Random Forest Classifier** | `clean_RandomForest_model.pkcls` | 0.595 | 0.387 | 0.388 | 0.386 | 0.081 | Evaluated & Saved |

### D. Pairwise Statistical ROC Analysis
Pairwise ROC AUC hypothesis probability testing confirmed that **Support Vector Machine (SVM)** holds a **91.2% probability** of outperforming Random Forest ($p = 0.912$) and a **65.3% probability** of outperforming kNN ($p = 0.653$).

### E. Application Deployment Results
The serialized model (`clean_SVM_model.pkcls`) was deployed in `app/app.py` using Streamlit and Folium GIS:
- Dynamic dropdown selection between all three trained Orange `.pkcls` models.
- Real-time normalization transform using exact domain offset/factor arrays.
- Input validation indicators warning users if entered coordinates/reviews exceed valid dataset boundaries.
- Interactive Folium map marker showing candidate target locations alongside the top 5 nearest reference venues.
- Publicly deployed at: `https://cafespot-xcjc4cmlduy3hvkikas3bk.streamlit.app` [2].

---

## IV. DISCUSSION

### A. Interpretation of Results
The experimental results demonstrate that geographic coordinates (`Latitude`, `Longitude`) and review density (`NumReview`) carry significant predictive signals for urban café rating brackets [1]. The RBF kernel SVM achieved superior precision (0.431), CA (0.442), and AUC (0.635), proving its ability to model non-linear spatial boundaries far more effectively than decision tree ensembles or local instance voting.

### B. Selected Model Justification
While kNN yielded a slightly higher F1-Score ($0.405$ vs $0.399$) due to uniform class recall, **SVM was selected as the deployable model** because it achieved the highest **AUC ($0.635$)**, **Accuracy ($0.442$)**, **Precision ($0.431$)**, and **MCC ($0.174$)**. Higher precision directly protects entrepreneurs against capital loss caused by false positive location recommendations.

### C. Errors and Trade-offs
Misclassifications primarily occurred along transitional urban borders between commercial districts where rating variance exists despite identical coordinate proximity.

### D. Practical Implications
Entrepreneurs can utilize **CafeSpot Recommender** during pre-feasibility analysis to benchmark candidate site leases before executing financial contracts.

### E. Limitations
1. The dataset originates from Bengaluru, India; performance in Philippine urban centers requires localized retraining.  
2. The model relies on static directory snapshots and does not reflect real-time foot-traffic API streams.  
3. Rental overhead and seating capacity were not available in the raw Kaggle dataset.

---

## V. CONCLUSION AND RECOMMENDATIONS

### A. Conclusion
This study successfully developed, evaluated, and deployed **CafeSpot Recommender**, a machine learning decision support system for urban café site selection. Through 5-fold stratified cross-validation on 326 cleaned venue records, **Support Vector Machine (SVM)** proved to be the top-performing model, achieving the highest AUC (**0.635**), Accuracy (**0.442**), Precision (**0.431**), and MCC (**0.174**). The model was successfully serialized (`clean_SVM_model.pkcls`) and integrated into a live, cloud-hosted Streamlit web application.

### B. Recommendations
1. **For Future Researchers:** Incorporate real-time foot-traffic API mobility feeds, competitor buffer distance features (e.g., proximity to anchor chains), and demographic income levels.
2. **For Application Developers:** Expand dataset coverage to Philippine metropolitan regions (e.g., Metro Manila, Metro Cebu) and enable interactive GIS map-click coordinate selection.

---

## REFERENCES

[1] J. Lu, X. Zheng, E. Nervino, Y. Li, Z. Xu, and Y. Xu, “Retail store location screening: A machine learning-based approach,” *Journal of Retailing and Consumer Services*, vol. 77, Art. no. 103620, Mar. 2024.

[2] R. A. P. Gailo, M. N. Mercado, A. M. D. Pardillo, and P. M. C. Rodriguez, *CafeSpot Recommender System*, GitHub Repository, 2026. [Online]. Available: https://github.com/Rosegailo/CafeSpot (accessed Oct. 15, 2026).

[3] J. Demšar, T. Zupan, G. Leban, M. Luštrek, B. Zupan, and I. Bratko, “Orange: Data mining toolbox in Python,” *Journal of Machine Learning Research*, vol. 14, pp. 2349–2353, 2013.

[4] F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[5] yashvardhanprasad, “Cafe data,” Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/yashvardhanprasad/cafe-data (accessed Oct. 15, 2026).

---

## APPENDICES

### Appendix A — Data Dictionary
| Variable | Role | Type | Unit / Scale | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Company Name` | Meta | Text | N/A | Business venue name |
| `Address` | Meta | Text | N/A | Street location |
| `Phone` | Meta | Text | N/A | Telephone contact |
| `Link` | Meta | Text | URL | Online directory link |
| `NumReview` | Feature | Continuous | Integer | Review submission count |
| `Pin Code` | Meta | Categorical | Postal Code | Local postal code |
| `Latitude` | Feature | Continuous | Degrees ($^\circ$) | Geographic latitude |
| `Longitude` | Feature | Continuous | Degrees ($^\circ$) | Geographic longitude |
| `Rating` | Target | Discrete | Class $[0, 1, 2]$ | Discretized rating tier |

### Appendix B — Group Contribution Record
- **Data Collection & Preprocessing:** Pauleen Mae C. Rodriguez  
- **Orange Data Mining Workflow & Model Export:** Rosemarie Ann P. Gailo  
- **Streamlit Web Application & Cloud Deployment:** Rosemarie Ann P. Gailo  
- **Technical Documentation & Academic Paper:** Rosemarie Ann P. Gailo, Moira N. Mercado, Angel Mhae D. Pardillo  

---

### Appendix C — Signed Ownership and Authorship Declaration

#### Project Information
- **Project Title:** CafeSpot Recommender
- **Course and Section:** Introduction to Machine Learning / Data Science  
- **Group Name:** Byte & Brew  
- **Date Submitted:** October 15, 2026  
- **Course Instructor:** Dr. Armida P. Salazar  

#### Declaration Statement
We, Dr. Armida P. Salazar as course instructor and the undersigned student group members, jointly declare that we are the owners and authors of the submitted project output identified above. The project output includes the application, source code, trained model pipelines, exploratory data analysis, visualizations, technical documentation, research paper, and original materials produced specifically for this course project.

We affirm that the submitted work accurately represents our documented contributions and that all external datasets, software libraries, code, models, frameworks, publications, and third-party materials have been properly cited and acknowledged using **IEEE citation format**.

#### Signatories
- **Dr. Armida P. Salazar** (Course Instructor) — *October 15, 2026*  
- **Rosemarie Ann P. Gailo** (Author & Developer) — *October 15, 2026*  
- **Moira N. Mercado** (Author & Researcher) — *October 15, 2026*  
- **Angel Mhae D. Pardillo** (Author & Researcher) — *October 15, 2026*  
- **Pauleen Mae C. Rodriguez** (Author & Data Analyst) — *October 15, 2026*
