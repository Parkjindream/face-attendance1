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

# --- API: recognize face ---
@app.route('/api/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_b64 = data.get('image')
    timestamp = data.get('timestamp')
    camera_id = data.get('camera_id', 'unknown')

    if not image_b64 or not timestamp:
        return jsonify({'error': 'Missing image or timestamp'}), 400
        
    # Prevent duplicate recognition
    last_sent_time = recognition_cache.get(camera_id)
    now = datetime.fromisoformat(timestamp)
    if last_sent_time and (now - last_sent_time).total_seconds() < CACHE_INTERVAL_SECONDS:
        return jsonify({'status': 'skipped, cached'}), 200
    recognition_cache[camera_id] = now
