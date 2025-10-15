# python_root/config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ENCODINGS_DIR = os.path.join(DATA_DIR, 'encodings')
DB_PATH = os.path.join(DATA_DIR, 'db.sqlite3')
EXPORT_DIR = os.path.join(BASE_DIR, 'exports')
