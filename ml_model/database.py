import sqlite3
import os
import hashlib
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database', 'forensic.db')

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            case_reference TEXT UNIQUE,
            sex_prediction TEXT,
            sex_confidence REAL,
            age_prediction TEXT,
            age_confidence REAL,
            bones_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Bone measurements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bone_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            bone_name TEXT,
            measurement_name TEXT,
            measurement_value REAL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, full_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, password, full_name)
            VALUES (?, ?, ?)
        ''', (username, hash_password(password), full_name))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, full_name FROM users 
        WHERE username = ? AND password = ?
    ''', (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    if user:
        return True, {"id": user[0], "full_name": user[1]}
    return False, "Invalid username or password!"

def save_prediction(user_id, sex_pred, sex_conf, age_pred, age_conf, bones_used, measurements):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    case_ref = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    cursor.execute('''
        INSERT INTO predictions 
        (user_id, case_reference, sex_prediction, sex_confidence, 
         age_prediction, age_confidence, bones_used)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, case_ref, sex_pred, sex_conf, age_pred, age_conf, ','.join(bones_used)))

    prediction_id = cursor.lastrowid

    for measure_name, measure_value in measurements.items():
        if measure_value:
            cursor.execute('''
                INSERT INTO bone_measurements 
                (prediction_id, bone_name, measurement_name, measurement_value)
                VALUES (?, ?, ?, ?)
            ''', (prediction_id, measure_name[:4], measure_name, float(measure_value)))

    conn.commit()
    conn.close()
    return case_ref

def get_user_predictions(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (user_id,))
    results = cursor.fetchall()
    conn.close()
    return results

if __name__ == "__main__":
    init_db()