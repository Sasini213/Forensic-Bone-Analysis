# 🦴 Forensic Bone Analysis System

A machine learning web application for predicting biological sex and age range from skeletal bone measurements. Built as a Final Year Project for potential use by the Government Analyst's Department and Sri Lanka Police.

🌐 **Live Demo:** [https://forensic-bone-analysis.onrender.com](https://forensic-bone-analysis.onrender.com)

---

## ✨ Features

- **Biological sex prediction** — Random Forest classifier (~98% accuracy)
- **Age range estimation** — XGBoost classifier (~84% accuracy)
- **5 bone types supported** — Humerus, Radius, Femur, Tibia, Os Coxae
- **Per-bone individual models** with left/right measurement averaging
- **PDF forensic case report** generation (downloadable per prediction)
- **Prediction history** with search, filter, delete, and analyst notes
- **Dashboard** with summary stats and age range distribution charts
- **Secure user authentication** with session management and password hashing
- **AJAX username availability checker** on registration

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask 3.x |
| Database | PostgreSQL (via psycopg2) |
| ML Models | scikit-learn (Random Forest), XGBoost |
| Data Processing | pandas, numpy, SMOTE (imbalanced-learn) |
| PDF Generation | ReportLab |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js |
| Deployment | Render (with Gunicorn) |

---

## 📁 Project Structure

```
Forensic-Bone-Analysis/
├── app.py                  # Main Flask application
├── requirements.txt
├── Procfile                # Render/Gunicorn entry point
├── ml_model/
│   ├── database.py         # PostgreSQL DB functions (psycopg2)
│   ├── preprocessing.py    # Data cleaning & augmentation
│   ├── train_model.py      # Global model training
│   ├── train_per_bone.py   # Per-bone model training
│   ├── gender_*.pkl        # Trained Random Forest models
│   ├── age_*.pkl           # Trained XGBoost models
│   ├── scaler_*.pkl        # StandardScaler per bone
│   ├── features_*.pkl      # Feature lists per bone
│   └── age_label_encoder.pkl
├── templates/              # Jinja2 HTML templates
└── static/                 # CSS, JS, images
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sasini213/Forensic-Bone-Analysis.git
cd Forensic-Bone-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL

Create a local PostgreSQL database and set the connection URL as an environment variable:

```bash
# Windows (Command Prompt)
set DATABASE_URL=postgresql://username:password@localhost:5432/forensic_db

# Windows (PowerShell)
$env:DATABASE_URL="postgresql://username:password@localhost:5432/forensic_db"

# Mac/Linux
export DATABASE_URL=postgresql://username:password@localhost:5432/forensic_db
```

### 4. Preprocess data & train models

```bash
python ml_model/preprocessing.py
python ml_model/train_model.py
python ml_model/train_per_bone.py
```

## 🚀 Deployment (Render)

This project is deployed on [Render](https://render.com) using **Gunicorn**.

### Render setup steps:
1. Create a **PostgreSQL** database on Render
2. Create a new **Web Service** linked to this GitHub repo
3. Set the environment variable: `DATABASE_URL` = *(your Render PostgreSQL internal URL)*
4. Render auto-detects the `Procfile` and deploys

The `Procfile` contains:
```
web: gunicorn app:app
```

The live application auto-deploys from the `main` branch on every push.

**Live URL:** [https://forensic-bone-analysis.onrender.com](https://forensic-bone-analysis.onrender.com)

> ⚠️ Render free-tier instances spin down after inactivity. The first request may take 30–60 seconds.

---

## 🧠 ML Model Details

| Task | Model | Accuracy | Dataset Size |
|---|---|---|---|
| Biological Sex | Random Forest | ~98% | 4,501 records (augmented) |
| Age Range | XGBoost | ~84% | 4,501 records (augmented) |

**Preprocessing pipeline:**
- Median imputation for missing values
- Left/right measurement averaging (e.g. `LHML` + `RHML` → `AVG_HML`)
- StandardScaler feature normalisation
- SMOTE for class balancing
- LabelEncoder for categorical targets

**Age range categories:** 18–25, 26–35, 36–45, 46–55, 55+

---

## 📋 Supported Bone Measurements

| Bone | Measurements |
|---|---|
| Humerus | LHML, LHEB, LHHD, LHMLD, LHAPD (+ Right) |
| Radius | LRML, LRMLD, LRAPD (+ Right) |
| Femur | LFML, LFBL, LFEB, LFAB, LFHD, LFMLD, LFAPD (+ Right) |
| Tibia | LTML, LTPB, LTMLD, LTAPD (+ Right) |
| Os Coxae | BIB, LIBL, RIBL, LAcH, RAcH |

---

## ⚠️ Disclaimer

This system is an AI-assisted decision support tool. Predictions are based on statistical models and are intended to **assist — not replace** — the judgement of a qualified forensic expert. All conclusions must be verified by a certified forensic anthropologist before use in any legal or investigative proceeding.

---

## 👩‍💻 Author

Developed as a Final Year Project — BSc (Hons) Computer Science
