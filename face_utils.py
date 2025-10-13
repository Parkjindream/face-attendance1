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
 records = cursor.fetchall()
    conn.close()

    known_encodings = []
    known_ids = []

for row in records:
    student_id = row['student_id']
    encodeing = np.frombuffe(row['encoding'], dtype=np.float64)
    known_encodings.append(encoding)
    known_ids.append(student_id)
    
return known_encodings, known_ids

# --- Compute face encoding from image ---
def get_face_encoding(image):
    rgb_img = image[:, :, ::-1]  # BGR to RGB
    encodings = face_recognition.face_encodings(rgb_img)
    if encodings:
        return encodings[0]
    else:
        return None

  distances = face_recognition.face_distance(known_encodings, face_encoding)
    min_distance = np.min(distances)
    best_match_index = np.argmin(distances)

    if min_distance <= threshold:
        return known_ids[best_match_index], float(min_distance)
    else:
        return None, float(min_distance)
