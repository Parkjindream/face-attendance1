# python_root/models.py
import sqlite3
import os
from config import DB_PATH
from datetime import datetime

# --- Database connection ---
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Initialize tables ---
def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        meta TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    # Encodings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS encodings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        encoding BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )''')

    # Attendance logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        timestamp TEXT NOT NULL,
        status TEXT,
        match_score REAL,
        camera_id TEXT,
        event_type TEXT DEFAULT 'IN',
        method TEXT DEFAULT 'image',
        raw_payload_meta TEXT,
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )''')

    # Audit logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        action TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        details TEXT
    )''')

    conn.commit()
    conn.close()

# --- Helper functions ---
def add_student(student_id, name, meta=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO students (student_id, name, meta) VALUES (?, ?, ?)",
                   (student_id, name, meta))
    conn.commit()
    conn.close()

# Initialize DB on first run
if __name__ == '__main__':
    init_db()
    print('Database initialized successfully.')


def init_db():
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.executescript(SCHEMA_SQL)
        conn.commit()

        # default settings
        cur.execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", ('attendance_start', '08:30'))
        cur.execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", ('late_threshold_minutes', '10'))
        conn.commit()

# Student operations

def add_student(student_id: str, name: str, meta: dict = None):
    now = datetime.utcnow().isoformat()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO students(student_id, name, meta, created_at) VALUES(?, ?, ?, ?)",
                    (student_id, name, json.dumps(meta or {}), now))
        conn.commit()


def get_student_by_id(student_id: str):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_students():
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT student_id, name, meta, created_at FROM students ORDER BY name")
        return [dict(r) for r in cur.fetchall()]

# Encoding operations

def save_encoding(student_id: str, encoding):
    enc_json = json.dumps([float(x) for x in encoding])
    now = datetime.utcnow().isoformat()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO encodings(student_id, encoding, created_at) VALUES(?, ?, ?)",
                    (student_id, enc_json, now))
        conn.commit()


def get_all_encodings():
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT student_id, encoding FROM encodings")
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append((r['student_id'], json.loads(r['encoding'])))
        return results

# Attendance logging

def log_attendance(student_id: str, timestamp: datetime, status: str, match_score: float = None, method: str = 'face'):
    ts = timestamp.isoformat()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO attendance_logs(student_id, timestamp, status, match_score, method) VALUES(?, ?, ?, ?, ?)",
                    (student_id, ts, status, match_score, method))
        conn.commit()


def has_logged_today(student_id: str, day: date = None):
    day = day or datetime.utcnow().date()
    start = datetime.combine(day, datetime.min.time()).isoformat()
    end = datetime.combine(day, datetime.max.time()).isoformat()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance_logs WHERE student_id = ? AND timestamp BETWEEN ? AND ?", (student_id, start, end))
        c = cur.fetchone()[0]
        return c > 0

# Settings helpers

def get_setting(key: str, default=None):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        r = cur.fetchone()
        return r[0] if r else default


def set_setting(key: str, value: str):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)", (key, value))
        conn.commit()

if __name__ == '__main__':
    init_db()
    print('Database initialized at', DATABASE_PATH)


