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

    # Decode image
    image_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Get face encoding
    encoding = get_face_encoding(img)
    if encoding is None:
        return jsonify({'status': 'No face detected'}), 200

    # Load known encodings
    known_encodings, known_ids = load_known_encodings()
 add-app


 main
    # Match face
    student_id, distance = match_face(encoding, known_encodings, known_ids, FACE_MATCH_THRESHOLD)

    status = None
    if student_id:
        # Determine attendance status
        start_time = datetime.strptime(ATTENDANCE_START_TIME, '%H:%M').time()
        late_time = (datetime.combine(now.date(), start_time) + timedelta(minutes=LATE_THRESHOLD_MINUTES)).time()
        if now.time() <= late_time:
            status = 'On Time'
        else:
            status = 'Late'
        # Save attendance log
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance_logs (student_id, timestamp, status, match_score, camera_id) VALUES (?, ?, ?, ?, ?)",
                       (student_id, now.isoformat(), status, distance, camera_id))
        conn.commit()
        conn.close()

        return jsonify({'status': status, 'student_id': student_id, 'distance': distance}), 200
    else:
        return jsonify({'status': 'Unknown', 'distance': distance}), 200

# --- API: enroll student via image upload ---
@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    return jsonify({'status': 'Not implemented'}), 501

# --- Dashboard ---
@app.route('/dashboard')
def dashboard():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT al.id, s.student_id, s.name, al.timestamp, al.status FROM attendance_logs al LEFT JOIN students s ON al.student_id = s.student_id ORDER BY al.timestamp DESC")
    logs = cursor.fetchall()
    conn.close()
    return render_template('dashboard.html', logs=logs)

# --- Export attendance logs to Excel ---
@app.route('/export')
def export():
    conn = get_connection()
    df = pd.read_sql_query("SELECT al.id, s.student_id, s.name, al.timestamp, al.status FROM attendance_logs al LEFT JOIN students s ON al.student_id = s.student_id ORDER BY al.timestamp", conn)
    conn.close()

    export_path = os.path.join(EXPORT_DIR, f'attendance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    df.to_excel(export_path, index=False)
    return send_file(export_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
