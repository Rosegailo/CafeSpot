import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# 1. Page Configuration
st.set_page_config(
    page_title="Byte & Brew: CafeSpot - AI Predictor",
    page_icon="☕",
    layout="wide",
)

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

def parse_prediction_and_confidence(prediction, prediction_proba, classes):
    pred_str = str(prediction).strip()
    num_classes = len(classes) if classes is not None and len(classes) > 0 else 2

    if num_classes == 2:
        # Binary Classification (0 = Standard/Low < 4.30, 1 = Top Rated >= 4.30)
        if pred_str in ['1', '1.0'] or 'Top' in pred_str or '≥' in pred_str or '>=' in pred_str:
            label_text = "✨ **Prediction: TOP RATED (≥ 4.30)**"
            is_top = True
            badge_type = "success"
            idx = 1
        else:
            label_text = "⚠️ **Prediction: STANDARD / LOW RATED (< 4.30)**"
            is_top = False
            badge_type = "warning"
            idx = 0
    else:
        # 3-Class Classification (< 4.05, 4.05 - 4.45, ≥ 4.45)
        if pred_str in ['2', '2.0'] or '≥' in pred_str or '>=' in pred_str or 'Top' in pred_str:
            label_text = "✨ **Prediction: TOP RATED (≥ 4.45)**"
            is_top = True
            badge_type = "success"
            idx = 2
        elif pred_str in ['1', '1.0'] or '4.05' in pred_str or 'Moderate' in pred_str:
            label_text = "⭐ **Prediction: MODERATE RATED (4.05 - 4.45)**"
            is_top = True
            badge_type = "info"
            idx = 1
        else:
            label_text = "⚠️ **Prediction: STANDARD / LOW RATED (< 4.05)**"
            is_top = False
            badge_type = "warning"
            idx = 0

    confidence_str = None
    if prediction_proba is not None:
        try:
            p_arr = np.array(prediction_proba).flatten()
            if 0 <= idx < len(p_arr):
                conf_val = p_arr[idx]
            else:
                conf_val = np.max(p_arr)
            confidence_str = f"{float(conf_val) * 100:.1f}%"
        except Exception:
            confidence_str = None

    return label_text, confidence_str, is_top, badge_type, num_classes

# 2. Asset Loading (Orange .pkcls models + domain normalization transform)
@st.cache_resource
def load_assets(model_name):
    file_map = {
        "Clean Orange SVM (clean_SVM_model.pkcls)": "clean_SVM_model.pkcls",
        "Clean Orange kNN (clean_KNN_model.pkcls)": "clean_KNN_model.pkcls",
        "Clean Orange Random Forest (clean_RandomForest_model.pkcls)": "clean_RandomForest_model.pkcls",
        "Orange SVM (best_model.pkcls)": "best_model.pkcls",
        "Orange kNN (KNN_model.pkcls)": "KNN_model.pkcls",
        "Orange Random Forest (Random_forest_model.pkcls)": "Random_forest_model.pkcls",
    }

    filename = file_map.get(model_name, "clean_SVM_model.pkcls")
    model_path = f"models/{filename}"
    data_path = "data/processed/cleaned_cafes.csv"

    if not os.path.exists(model_path):
        return None, np.array([]), np.array([]), None, np.array([0, 1])

    offsets = []
    factors = []

    try:
        model_obj = joblib.load(model_path)
    except Exception:
        with open(model_path, 'rb') as f:
            model_obj = OrangeUnpickler(f).load()

    # Extract Orange domain normalization parameters if present
    if hasattr(model_obj, 'domain') and hasattr(model_obj.domain, 'attributes'):
        for attr in model_obj.domain.attributes:
            comp = getattr(attr, '_compute_value', None)
            offsets.append(float(getattr(comp, 'offset', 0.0)))
            factors.append(float(getattr(comp, 'factor', 1.0)))

    skl_model = getattr(model_obj, 'skl_model', model_obj)
    classes = getattr(skl_model, 'classes_', np.array([0, 1]))
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None

    return skl_model, np.array(offsets), np.array(factors), df, classes

# 3. Sidebar Inputs & Model Selection
st.sidebar.header("🛠️ Model Configuration")

selected_model_name = st.sidebar.selectbox(
    "Choose Orange Model to Test",
    [
        "Clean Orange SVM (clean_SVM_model.pkcls)",
        "Clean Orange kNN (clean_KNN_model.pkcls)",
        "Clean Orange Random Forest (clean_RandomForest_model.pkcls)",
        "Orange SVM (best_model.pkcls)",
        "Orange kNN (KNN_model.pkcls)",
        "Orange Random Forest (Random_forest_model.pkcls)",
    ]
)

model, offsets, factors, df, classes = load_assets(selected_model_name)

st.sidebar.header("📍 Café Parameters")
input_lat = st.sidebar.number_input("Latitude", value=12.9716, format="%.6f")
input_lon = st.sidebar.number_input("Longitude", value=77.5946, format="%.6f")
input_reviews = st.sidebar.number_input("Expected Number of Reviews", min_value=0, value=150, step=10)
input_pincode = st.sidebar.number_input("Pin Code", value=560001, step=1)

# 4. Header
st.title("☕ Byte & Brew: CafeSpot Predictor")
st.markdown(f"### Currently active: **{selected_model_name}**")

if model is None:
    st.error(f"Model file for `{selected_model_name}` not found in `models/` folder.")
    st.stop()

# 5. Model Comparison Summary (Exact 5-Fold CV metrics from Orange Test & Score)
with st.expander("📊 View Orange 5-Fold Cross-Validation Metrics"):
    comparison_data = {
        "Orange Model": ["SVM", "kNN", "Random Forest"],
        "Saved File": ["`clean_SVM_model.pkcls` / `best_model.pkcls` 🏆", "`clean_KNN_model.pkcls`", "`clean_RandomForest_model.pkcls`"],
        "AUC": ["0.633 🏆", "0.620", "0.601"],
        "CA (Accuracy)": ["0.451 🏆", "0.423", "0.396"],
        "Precision": ["0.432 🏆", "0.427", "0.394"],
        "F1-Score": ["0.398", "0.424 🏆", "0.395"],
        "MCC": ["0.192 🏆", "0.134", "0.090"],
    }
    st.table(pd.DataFrame(comparison_data))
    st.success("🏆 **Orange SVM** achieves the highest **AUC (0.633)**, **Accuracy (0.451)**, **Precision (0.432)**, and **MCC (0.192)** among all 3 models evaluated in Orange Data Mining.")

# 6. Prediction Logic
n_features_expected = getattr(model, 'n_features_in_', 4)
raw_4 = np.array([[input_reviews, input_pincode, input_lat, input_lon]])

if n_features_expected == 5:
    raw_features = np.hstack([raw_4, [[1]]])
else:
    raw_features = raw_4

# Normalize features if Orange normalization factors are present
if len(offsets) >= 4 and len(factors) >= 4 and np.any(factors[:4] != 1.0):
    norm_4 = (raw_4 - offsets[:4]) * factors[:4]
    if n_features_expected == 5:
        norm_features = np.hstack([norm_4, [[1]]])
    else:
        norm_features = norm_4
else:
    norm_features = raw_features

try:
    raw_pred = model.predict(norm_features)[0]
except Exception:
    raw_pred = 0

try:
    if hasattr(model, 'predict_proba'):
        raw_proba = model.predict_proba(norm_features)[0]
    else:
        raw_proba = None
except Exception:
    raw_proba = None

label_text, confidence_str, is_top, badge_type, num_classes = parse_prediction_and_confidence(raw_pred, raw_proba, classes)

# 7. Results Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Result")
    if badge_type == "success":
        st.success(label_text)
    elif badge_type == "info":
        st.info(label_text)
    else:
        st.warning(label_text)

    if confidence_str is not None:
        st.write(f"Confidence: **{confidence_str}**")

    # Map visualization
    m = folium.Map(location=[input_lat, input_lon], zoom_start=14)
    folium.Marker(
        [input_lat, input_lon],
        popup="Target Location",
        icon=folium.Icon(color="green" if is_top else "red", icon="coffee", prefix="fa")
    ).add_to(m)
    st_folium(m, width="100%", height=300)

with col2:
    st.subheader("Model Diagnostic Profile")
    st.write(f"**Model Name:** {selected_model_name}")
    st.write("**Orange Workflow:** CSV Import ➔ Select Columns ➔ Learner ➔ Save Model")
    st.write("**Features Used:** NumReview, Pin Code, Latitude, Longitude")
    if num_classes == 2:
        st.write("**Target Rating Bins:** `< 4.30` (Standard/Low), `≥ 4.30` (Top Rated)")
    else:
        st.write("**Target Rating Bins:** `< 4.05`, `4.05 - 4.45`, `≥ 4.45`")

    if "SVM" in selected_model_name:
        st.write("**Algorithm:** Support Vector Machine (RBF Kernel)")
        st.write("**Strength:** Best overall spatial decision boundaries.")
    elif "kNN" in selected_model_name:
        st.write("**Algorithm:** k-Nearest Neighbors (kNN)")
        st.write("**Strength:** Classifies locations based on localized spatial proximity.")
    elif "Random Forest" in selected_model_name:
        st.write("**Algorithm:** Random Forest Ensemble")
        st.write("**Strength:** Multi-tree decision splitting across feature thresholds.")

# 8. Nearby Reference Locations (Synchronized across Pin Code, Reviews, Lat, and Lon)
if df is not None:
    st.markdown("---")
    st.subheader("Nearby Reference Locations from Core Dataset")

    df_ref = df.copy()
    # Compute Pin Code match (0 = Exact Match, 1 = Mismatch)
    df_ref['Pin_Mismatch'] = (df_ref['Pin Code'] != input_pincode).astype(int)
    # Compute Spatial Euclidean Distance (Latitude & Longitude)
    df_ref['Geo_Distance'] = np.sqrt((df_ref['Latitude'] - input_lat)**2 + (df_ref['Longitude'] - input_lon)**2)
    # Compute Review Count Similarity
    df_ref['Review_Diff'] = np.abs(df_ref['NumReview'] - input_reviews)

    # Sort primarily by Pin Code Match, then Geographic Distance, then Review Similarity
    nearby_df = df_ref.sort_values(['Pin_Mismatch', 'Geo_Distance', 'Review_Diff']).head(5)

    # Inform user if exact Pin Code is not present in dataset
    if (df['Pin Code'] == input_pincode).sum() == 0:
        st.info(f"ℹ️ Pin Code **{input_pincode}** is not present in the dataset. Displaying geographically nearest reference locations instead.")

    st.table(nearby_df[['Company Name', 'Rating', 'NumReview', 'Address']])
