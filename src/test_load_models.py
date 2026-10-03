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

print("🔄 Testing Orange Model Loading (3 Saved Models)...")

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

            print(f"\n✅ {model_title} Loaded Successfully!")

            sample_orange = np.array([[150, 560001, 12.9716, 77.5946]])
            prediction = orange_model.predict(sample_orange)[0]
            probabilities = orange_model.predict_proba(sample_orange)[0] if hasattr(orange_model, 'predict_proba') else None

            print(f"--- {model_title} Test Results ---")
            print(f"Input [NumReview, Pin Code, Lat, Lon]: {sample_orange[0]}")
            print(f"Prediction Class: {prediction} -> {labels.get(prediction, 'Unknown')}")
            if probabilities is not None:
                print(f"Confidence Level: {probabilities[int(prediction)]*100:.2f}%")
        except Exception as e:
            print(f"❌ Error loading {model_title}: {e}")
    else:
        print(f"⚠️ {model_title} file (`{pkcls_path}`) not found.")
