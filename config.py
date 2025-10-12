# python_root/config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ENCODINGS_DIR = os.path.join(DATA_DIR, 'encodings')
DB_PATH = os.path.join(DATA_DIR, 'db.sqlite3')
EXPORT_DIR = os.path.join(BASE_DIR, 'exports')

# Make sure directories exist
os.makedirs(ENCODINGS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Face recognition settings
FACE_MATCH_THRESHOLD = 0.5          # Euclidean distance threshold for recognition
CACHE_INTERVAL_SECONDS = 30         # Time interval to prevent duplicate logs

# Attendance settings
ATTENDANCE_START_TIME = "08:30"    # HH:MM, class start time
LATE_THRESHOLD_MINUTES = 10         # After this time, status = 'Late'
