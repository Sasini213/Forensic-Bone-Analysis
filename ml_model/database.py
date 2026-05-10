import os
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# =====================================================================
# PostgreSQL connection — Supabase
# YOUR_PASSWORD_HERE තැන ඔයාගේ Supabase password එක දාන්න
# =====================================================================
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id                SERIAL PRIMARY KEY,
        username          TEXT UNIQUE NOT NULL,
        password          TEXT NOT NULL,
        full_name         TEXT,
        security_question TEXT,
        security_answer   TEXT,
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id             SERIAL PRIMARY KEY,
    user_id        INTEGER,
    case_reference TEXT UNIQUE,
    sex_prediction TEXT,
    sex_confidence REAL,
    age_prediction TEXT,
    age_confidence REAL,
    bones_used     TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analyst_notes  TEXT DEFAULT '',
    FOREIGN KEY (user_id) REFERENCES users(id)
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS bone_measurements (
        id                SERIAL PRIMARY KEY,
        prediction_id     INTEGER,
        bone_name         TEXT,
        measurement_name  TEXT,
        measurement_value REAL,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )''')

    conn.commit()
    cursor.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS analyst_notes TEXT DEFAULT ''")
    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully!")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, full_name, security_question='', security_answer=''):
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, password, full_name, security_question, security_answer) VALUES (%s,%s,%s,%s,%s)',
            (username, hash_password(password), full_name, security_question, hash_password(security_answer))
        )
        conn.commit()
        return True, 'Registration successful!'
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, 'Username already exists! Please choose a different username.'
    finally:
        cursor.close()
        conn.close()

def login_user(username, password):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, full_name FROM users WHERE username=%s AND password=%s',
        (username, hash_password(password))
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user:
        return True, {'id': user[0], 'full_name': user[1]}
    return False, 'Invalid username or password!'

def reset_password(username, security_question, security_answer, new_password):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id FROM users WHERE username=%s AND security_question=%s AND security_answer=%s',
        (username, security_question, hash_password(security_answer))
    )
    user = cursor.fetchone()
    if user:
        cursor.execute(
            'UPDATE users SET password=%s WHERE username=%s',
            (hash_password(new_password), username)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True, 'Password reset successful!'
    cursor.close()
    conn.close()
    return False, 'Invalid username or security answer!'

def save_prediction(user_id, sex_pred, sex_conf, age_pred, age_conf, bones_used, measurements, notes=''):
    conn = get_conn()
    cursor = conn.cursor()
    case_ref = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    cursor.execute('''INSERT INTO predictions
        (user_id, case_reference, sex_prediction, sex_confidence,
         age_prediction, age_confidence, bones_used, analyst_notes)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
        (user_id, case_ref, sex_pred, sex_conf, age_pred, age_conf, ','.join(bones_used), notes)
    )
    prediction_id = cursor.fetchone()[0]
    for measure_name, measure_value in measurements.items():
        if measure_value:
            cursor.execute('''INSERT INTO bone_measurements
                (prediction_id, bone_name, measurement_name, measurement_value)
                VALUES (%s,%s,%s,%s)''',
                (prediction_id, measure_name[:4], measure_name, float(measure_value))
            )
    conn.commit()
    cursor.close()
    conn.close()
    return case_ref

def get_user_predictions(user_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC',
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    # created_at datetime object → string convert කරනවා
    results = []
    for row in rows:
        row = list(row)
        if row[8] and hasattr(row[8], 'strftime'):
            row[8] = row[8].strftime('%Y-%m-%d %H:%M:%S')
        results.append(tuple(row))
    return results

def delete_prediction(prediction_id, user_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM bone_measurements WHERE prediction_id=%s', (prediction_id,))
    cursor.execute('DELETE FROM predictions WHERE id=%s AND user_id=%s', (prediction_id, user_id))
    conn.commit()
    cursor.close()
    conn.close()

def get_prediction_measurements(prediction_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT measurement_name, measurement_value FROM bone_measurements WHERE prediction_id=%s ORDER BY id',
        (prediction_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {row[0]: row[1] for row in rows}

def update_analyst_notes(prediction_id, user_id, notes):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE predictions SET analyst_notes=%s WHERE id=%s AND user_id=%s',
        (notes, prediction_id, user_id)
    )
    conn.commit()
    cursor.close()
    conn.close() 

def get_user_info(user_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT username, full_name, created_at FROM users WHERE id=%s', (user_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row

def change_password(user_id, current_password, new_password):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE id=%s AND password=%s',
                   (user_id, hash_password(current_password)))
    user = cursor.fetchone()
    if user:
        cursor.execute('UPDATE users SET password=%s WHERE id=%s',
                       (hash_password(new_password), user_id))
        conn.commit()
        cursor.close()
        conn.close()
        return True, 'Password updated successfully!'
    cursor.close()
    conn.close()
    return False, 'Current password is incorrect!'

if __name__ == "__main__":
    init_db()