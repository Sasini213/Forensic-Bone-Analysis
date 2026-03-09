import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle, os

df = pd.read_csv(r'D:\Project\Bone_Gender_Age_Project\dataset\cleaned_data_augmented_fixed.csv')

save_dir = os.path.dirname(os.path.abspath(__file__))

BONE_FEATURES = {
    'Humerus':  ['LHML','LHEB','LHHD','LHMLD','LHAPD','RHML','RHEB','RHHD','RHMLD','RHAPD','AVG_HML','AVG_HEB','AVG_HHD','AVG_HMLD','AVG_HAPD'],
    'Radius':   ['LRML','LRMLD','LRAPD','RRML','RRMLD','RRAPD','AVG_RML','AVG_RMLD','AVG_RAPD'],
    'Femur':    ['LFML','LFBL','LFEB','LFAB','LFHD','LFMLD','LFAPD','RFML','RFBL','RFEB','RFAB','RFHD','RFMLD','RFAPD','AVG_FML','AVG_FBL','AVG_FEB','AVG_FAB','AVG_FHD','AVG_FMLD','AVG_FAPD'],
    'Tibia':    ['LTML','LTPB','LTMLD','LTAPD','RTML','RTPB','RTMLD','RTAPD','AVG_TML','AVG_TPB','AVG_TMLD','AVG_TAPD'],
    'Os Coxae': ['BIB','LIBL','RIBL','LAcH','RAcH','AVG_IBL','AVG_AcH']
}

le = LabelEncoder()
le.fit(df['Age_Group'])

for bone, features in BONE_FEATURES.items():
    print(f"\n{'='*50}")
    print(f"Training models for: {bone}")
    print(f"{'='*50}")

    bone_df = df[features + ['Sex', 'Age_Group']].dropna()
    X = bone_df[features]
    
    # Gender model
    y_gender = bone_df['Sex']
    X_train, X_test, y_train, y_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
    
    try:
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
    except:
        X_bal, y_bal = X_train, y_train

    gender_model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
    gender_model.fit(X_bal, y_bal)
    gender_acc = accuracy_score(y_test, gender_model.predict(X_test))
    print(f"Gender Accuracy: {round(gender_acc*100, 1)}%")

    # Age model
    y_age = le.transform(bone_df['Age_Group'])
    X_train, X_test, y_train, y_test = train_test_split(X, y_age, test_size=0.2, random_state=42, stratify=y_age)
    
    try:
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
    except:
        X_bal, y_bal = X_train, y_train

    scaler = StandardScaler()
    X_bal_scaled = scaler.fit_transform(X_bal)
    X_test_scaled = scaler.transform(X_test)

    age_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42
    )
    age_model.fit(X_bal_scaled, y_bal)
    age_acc = accuracy_score(y_test, age_model.predict(X_test_scaled))
    print(f"Age Accuracy: {round(age_acc*100, 1)}%")

    # Save models
    bone_key = bone.replace(' ', '_').lower()
    with open(os.path.join(save_dir, f'gender_{bone_key}.pkl'), 'wb') as f:
        pickle.dump(gender_model, f)
    with open(os.path.join(save_dir, f'age_{bone_key}.pkl'), 'wb') as f:
        pickle.dump(age_model, f)
    with open(os.path.join(save_dir, f'scaler_{bone_key}.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, f'features_{bone_key}.pkl'), 'wb') as f:
        pickle.dump(features, f)

# Save label encoder
with open(os.path.join(save_dir, 'age_label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print("\n✅ All per-bone models saved!")