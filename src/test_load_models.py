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

print("🔄 Testing Clean Orange Models (3 Saved Models from Cleaned Dataset)...")

orange_files = {
    "Clean Orange SVM": "models/clean_SVM_model.pkcls",
    "Clean Orange kNN": "models/clean_KNN_model.pkcls",
    "Clean Orange Random Forest": "models/clean_RandomForest_model.pkcls"
}

labels_3class = {0: "< 4.05 (Standard/Low)", 1: "4.05 - 4.45 (Moderate)", 2: ">= 4.45 (Top Rated)"}
labels_2class = {0: "< 4.30 (Standard/Low)", 1: ">= 4.30 (Top Rated)"}

for model_title, pkcls_path in orange_files.items():
    if os.path.exists(pkcls_path):
        try:
            try:
                model_obj = joblib.load(pkcls_path)
            except Exception:
                with open(pkcls_path, 'rb') as f:
                    model_obj = OrangeUnpickler(f).load()

            offsets, factors = [], []
            if hasattr(model_obj, 'domain') and hasattr(model_obj.domain, 'attributes'):
                for attr in model_obj.domain.attributes:
                    comp = getattr(attr, '_compute_value', None)
                    offsets.append(float(getattr(comp, 'offset', 0.0)))
                    factors.append(float(getattr(comp, 'factor', 1.0)))

            orange_model = getattr(model_obj, 'skl_model', model_obj)
            classes = getattr(orange_model, 'classes_', [0, 1])
            n_exp = getattr(orange_model, 'n_features_in_', 4)

            raw_4 = np.array([[150, 560001, 12.9716, 77.5946]])
            raw_feat = np.hstack([raw_4, [[1]]]) if n_exp == 5 else raw_4

            offsets, factors = np.array(offsets), np.array(factors)
            if len(offsets) >= 4 and len(factors) >= 4 and np.any(factors[:4] != 1.0):
                norm_4 = (raw_4 - offsets[:4]) * factors[:4]
                norm_feat = np.hstack([norm_4, [[1]]]) if n_exp == 5 else norm_4
            else:
                norm_feat = raw_feat

            prediction = orange_model.predict(norm_feat)[0]
            probabilities = orange_model.predict_proba(norm_feat)[0] if hasattr(orange_model, 'predict_proba') else None

            label_dict = labels_2class if len(classes) == 2 else labels_3class

            print(f"\n✅ {model_title} Loaded Successfully!")
            print(f"--- {model_title} Test Results ---")
            print(f"Input [NumReview, Pin Code, Lat, Lon]: {raw_4[0]}")
            print(f"Prediction Class: {prediction} -> {label_dict.get(int(prediction), 'Unknown')}")
            if probabilities is not None:
                idx = int(prediction) if 0 <= int(prediction) < len(probabilities) else 0
                print(f"Confidence Level: {probabilities[idx]*100:.2f}%")
        except Exception as e:
            print(f"❌ Error loading {model_title}: {e}")
    else:
        print(f"⚠️ {model_title} file (`{pkcls_path}`) not found.")
