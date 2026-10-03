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

# 2. Asset Loading (Orange .pkcls models)
@st.cache_resource
def load_assets(model_name):
    file_map = {
        "Orange SVM (best_model.pkcls)": "best_model.pkcls",
        "Orange kNN (KNN_model.pkcls)": "KNN_model.pkcls",
        "Orange Random Forest (Random_forest_model.pkcls)": "Random_forest_model.pkcls",
    }

    filename = file_map.get(model_name, "best_model.pkcls")
    model_path = f"models/{filename}"
    data_path = "data/processed/cleaned_cafes.csv"

    if not os.path.exists(model_path):
        return None, None

    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, 'rb') as f:
            orange_obj = OrangeUnpickler(f).load()
        if hasattr(orange_obj, 'skl_model'):
            model = orange_obj.skl_model
        else:
            model = orange_obj

    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    return model, df

# 3. Sidebar Inputs & Model Selection
st.sidebar.header("🛠️ Model Configuration")

selected_model_name = st.sidebar.selectbox(
    "Choose Orange Model to Test",
    [
        "Orange SVM (best_model.pkcls)",
        "Orange kNN (KNN_model.pkcls)",
        "Orange Random Forest (Random_forest_model.pkcls)",
    ]
)

model, df = load_assets(selected_model_name)

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
        "Saved File": ["`best_model.pkcls` 🏆", "`KNN_model.pkcls`", "`Random_forest_model.pkcls`"],
        "AUC": ["0.657 🏆", "0.649", "0.621"],
        "CA (Accuracy)": ["0.468 🏆", "0.441", "0.441"],
        "Precision": ["0.515 🏆", "0.442", "0.438"],
        "F1-Score": ["0.384", "0.441 🏆", "0.437"],
        "MCC": ["0.238 🏆", "0.161", "0.162"],
    }
    st.table(pd.DataFrame(comparison_data))
    st.success("🏆 **Orange SVM** achieves the highest **AUC (0.657)**, **Accuracy (0.468)**, **Precision (0.515)**, and **MCC (0.238)** among all 3 models evaluated in Orange Data Mining.")

# 6. Prediction Logic (4 features: [NumReview, Pin Code, Latitude, Longitude])
features = np.array([[input_reviews, input_pincode, input_lat, input_lon]])
prediction = model.predict(features)[0]
prediction_proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None

# 7. Results Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Result")
    if prediction == 2:
        st.success("✨ **Prediction: TOP RATED (>= 4.45)**")
        if prediction_proba is not None:
            st.write(f"Confidence: **{prediction_proba[int(prediction)]*100:.1f}%**")
    elif prediction == 1:
        st.info("⭐ **Prediction: MODERATE RATED (4.05 - 4.45)**")
        if prediction_proba is not None:
            st.write(f"Confidence: **{prediction_proba[int(prediction)]*100:.1f}%**")
    else:
        st.warning("⚠️ **Prediction: STANDARD / LOW RATED (< 4.05)**")
        if prediction_proba is not None:
            st.write(f"Confidence: **{prediction_proba[int(prediction)]*100:.1f}%**")

    # Map visualization
    m = folium.Map(location=[input_lat, input_lon], zoom_start=14)
    is_top = (prediction >= 1)
    folium.Marker(
        [input_lat, input_lon],
        popup="Target Location",
        icon=folium.Icon(color="green" if is_top else "red", icon="coffee", prefix="fa")
    ).add_to(m)
    st_folium(m, width="100%", height=300)

with col2:
    st.subheader("Model Diagnostic Profile")
    st.write(f"**Model Name:** {selected_model_name}")
    st.write("**Orange Workflow:** CSV Import ➔ Discretize ➔ Select Columns ➔ Learner ➔ Save Model")
    st.write("**Features Used:** NumReview, Pin Code, Latitude, Longitude")
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

# 8. Nearby Reference Locations
if df is not None:
    st.markdown("---")
    st.subheader("Nearby Reference Locations from Core Dataset")
    df['Distance'] = np.sqrt((df['Latitude'] - input_lat)**2 + (df['Longitude'] - input_lon)**2)
    st.table(df.sort_values('Distance').head(5)[['Company Name', 'Rating', 'NumReview', 'Address']])
