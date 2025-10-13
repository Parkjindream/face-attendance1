import cv2
import numpy as np
import os
from models import add_student, get_connection
from face_utils import get_face_encoding
from config import ENCODINGS_DIR

def enroll_student(student_id, name, image_path):
  # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return False
  # Get encoding
    encoding = get_face_encoding(image)
    if encoding is None:
        print(f"No face found in image {image_path}")
        return False
  # Add student to DB
    add_student(student_id, name)
  # Save encoding to DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO encodings (student_id, encoding, source) VALUES (?, ?, ?)"
                   , (student_id, encoding.tobytes(), os.path.basename(image_path)))
    conn.commit()
    conn.close()
  # Optionally save encoding as npy file
    npy_path = os.path.join(ENCODINGS_DIR, f"{student_id}.npy")
    np.save(npy_path, encoding)
