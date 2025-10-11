from flask import Flask, request, jsonify, render_template, send_file
app = Flask(__name__)
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from models import get_connection, init_db
from face_utils import get_face_encoding, load_known_encodings, match_face
from config import FACE_MATCH_THRESHOLD, CACHE_INTERVAL_SECONDS, ATTENDANCE_START_TIME, LATE_THRESHOLD_MINUTES, EXPORT_DIR
import os
import pandas as pd
# Initialize DB
init_db()

# Cache to prevent duplicate recognition
recognition_cache = {}
