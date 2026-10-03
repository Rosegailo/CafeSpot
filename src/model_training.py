import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib
import os

# 1. Load Processed Data
data_path = "data/processed/cleaned_cafes.csv"
if not os.path.exists(data_path):
    print("Error: data/processed/cleaned_cafes.csv not found. Run notebooks/eda_and_preprocessing.py first.")
    exit()

df = pd.read_csv(data_path)

# 2. Define Features and Target
features = ['Latitude', 'Longitude', 'NumReview']
X = df[features]
y = df['Target']

# 3. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

# 5. Initialize Models (Exactly 3 as required)
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 6. Model Comparison using 5-Fold Cross-Validation
print("--- Model Comparison (5-Fold CV F1-Score) ---")
cv_results = {}
best_f1 = -1
best_model_name = ""

for name, model in models.items():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='f1')
    mean_f1 = scores.mean()
    cv_results[name] = mean_f1
    print(f"{name}: Mean F1 = {mean_f1:.4f}")

    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_model_name = name

# 7. Final Training & Saving
print(f"\n🏆 Best Model Identified: {best_model_name}")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f'models/{filename}')

best_model = models[best_model_name]
joblib.dump(best_model, 'models/best_model.pkl')

# 8. Evaluation on Test Set
y_pred = best_model.predict(X_test_scaled)
print(f"\n--- Final Test Set Evaluation ({best_model_name}) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
