# python_root/face_utils.py
import face_recognition
import numpy as 
import os
from models import get_connection
from config import ENCODINGS_DIR, FACE_MATCH_THRESHOLD

# --- Load known encodings from database ---
def load_known_encodings():
    conn = get_connection()
  cursor = conn.cursor()
    cursor.execute("SELECT student_id, encoding FROM encodings")
