import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os

# Clean dataset load කරන්න
df = pd.read_csv(r'D:\Project\Bone_Gender_Age_Project\dataset\cleaned_data_augmented_fixed.csv')

print("Dataset shape:", df.shape)

# Feature columns identify කරන්න
exclude_cols = ['Sex', 'Age', 'Age_Group']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]

# Gender Model - Random Forest Classifier
y_gender = df['Sex']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_gender, test_size=0.2, random_state=42)

gender_model = RandomForestClassifier(n_estimators=100, random_state=42)
gender_model.fit(X_train, y_train)

y_pred_gender = gender_model.predict(X_test)
print("\n=== Gender Model ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_gender), 2))
print(classification_report(y_test, y_pred_gender,
      target_names=['Male', 'Female'], zero_division=0))

# Age Group Model - XGBoost Classifier
le = LabelEncoder()
y_age = le.fit_transform(df['Age_Group'])
X_train, X_test, y_train, y_test = train_test_split(X, y_age, test_size=0.2, random_state=42, stratify=y_age)

# SMOTE apply කරන්න
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Balanced class distribution:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for u, c in zip(le.classes_, counts):
    print(f"  {u}: {c}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

age_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
age_model.fit(X_train_balanced, y_train_balanced)
y_pred_age = age_model.predict(X_test)

print("\n=== Age Group Model ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_age), 2))
print(classification_report(y_test, y_pred_age,
      target_names=le.classes_, zero_division=0))

# Models Save කරන්න - pickle files ලෙස save කරන්න
save_dir = os.path.dirname(os.path.abspath(__file__))

# Gender prediction model 
with open(os.path.join(save_dir, 'gender_model.pkl'), 'wb') as f:
    pickle.dump(gender_model, f)

# Age group prediction model
with open(os.path.join(save_dir, 'age_model.pkl'), 'wb') as f:
    pickle.dump(age_model, f)

# Age label encoder — numbers → age group strings convert කරන්න
with open(os.path.join(save_dir, 'age_label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

# Scaler — new input data normalize කරන්න
with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("\nModels saved successfully!")