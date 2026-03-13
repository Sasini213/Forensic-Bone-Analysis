# 🦴 Forensic Bone Analysis System

A machine learning web application for predicting biological sex and age range from skeletal bone measurements.

## Features
- Gender prediction (98% accuracy)
- Age range estimation (84% accuracy)
- Per-bone individual models
- Case management with history
- Secure user authentication

## Tech Stack
- Python, Flask, SQLite
- scikit-learn, XGBoost
- HTML, CSS, JavaScript

## Setup Instructions

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Preprocess data
```
python ml_model/preprocessing.py
```

### 3. Train models
```
python ml_model/train_model.py
python ml_model/train_per_bone.py
```

### 4. Run application
```
python app.py
```

### 5. Open browser
```
http://127.0.0.1:5000
```