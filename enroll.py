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
