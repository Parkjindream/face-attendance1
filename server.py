from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request, Depends 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, FileResponse
import os, json, numpy as np, cv2, face_recognition, face_recognition_models
import datetime, threading, time, zipfile, io, glob, csv
from collections import defaultdict, deque
from typing import Optional, List, Dict, Any


# ---------- Export deps ----------
try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

try:
    import openpyxl
    from openpyxl import Workbook
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False

# --- Fix path โมเดล (ไม่จำเป็นเสมอไป แต่ไม่เป็นอันตราย) ---
os.environ["FACE_RECOGNITION_MODELS"] = face_recognition_models.__path__[0]

app = FastAPI(title="Face Recognition API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static & Templates ---
STATIC_DIR = "server/static"
TEMPLATES_DIR = "server/templates"
CONFIG_FILE = "attendance_config.json"
FACES_DB = "faces.json"
ATTEND_FILE = "attendance.json"
SESSIONS_FILE = "attendance_sessions.json"
LOG_FILE = "logs.json"   # ใช้เก็บ log แบบ real-time สำหรับหน้า Camera

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --------- (ตัวเลือก) API Key สำหรับ endpoint เสี่ยง/เปลี่ยนข้อมูล ---------
API_KEY = os.environ.get("API_KEY", "").strip()
def auth_guard(request: Request):
    """ถ้าไม่ได้ตั้ง API_KEY ไว้ → ปล่อยผ่านเพื่อความเข้ากันได้เดิม"""
    if not API_KEY:
        return
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- Utils ----------
def load_json_safe(path: str, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
            if not data:
                return default if default is not None else {}
            return json.loads(data)
    except Exception:
        return default if default is not None else {}

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def decode_upload_to_rgb(upload: UploadFile) -> np.ndarray:
    file_bytes = upload.file.read()
    upload.file.seek(0)
    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="ไม่สามารถอ่านไฟล์ภาพได้")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def now_datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def now_date_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def now_time_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

def _yyyymmdd_from_today() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")

def _normalize_date_token(s: str) -> str:
    # รับ "YYYY-MM-DD", "YYYY/MM/DD", "YYYYMMDD" => "YYYYMMDD"
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 8:
        return digits
    raise ValueError("รูปแบบวันที่ไม่ถูกต้อง (ต้องเป็น YYYYMMDD หรือ YYYY-MM-DD)")

def _hm_to_tuple(hm: str):
    # "HH:MM" -> (HH, MM)
    try:
        hh, mm = hm.split(":")
        return int(hh), int(mm)
    except Exception:
        return None

def _now_hm_tuple():
    n = datetime.datetime.now()
    return n.hour, n.minute

def _hm_lt(a, b):  # a < b
    return (a[0], a[1]) < (b[0], b[1])

def _is_valid_hhmm(s: str) -> bool:
    try:
        datetime.datetime.strptime(s, "%H:%M")
        return True
    except Exception:
        return False

# ---------- Config ----------
# เพิ่มพารามิเตอร์ที่ช่วย "ลื่น/ประหยัดทรัพยากร" และรองรับ "หลายช่วงเวลา"
_default_cfg = {
    "start": "0:00",
    "end": "23:59",
    "windows": [],  # ตัวเลือกใหม่: รองรับหลายช่วงเวลา [{"start":"08:00","end":"09:30"},{"start":"13:00","end":"15:00"}]
    # ค่าจับหลวม/แน่น + ตัวเลือกปรับความแม่นยำ/สมูท/ลดโหลด
    "det": {
        "strict": 0.62,    # match ทันทีเมื่อระยะ <= strict
        "loose": 0.75,     # โซนผ่อนคลาย ต้องโหวตซ้ำ
        "votes": 2,        # จำนวนครั้งที่ต้องพบภายใน window
        "window": 3.0,     # วินาทีที่นับโหวต
        "model": "hog",    # "hog" เร็วบน CPU, "cnn" แม่นยำกว่า (ต้องใช้ dlib CNN)
        "upsample": 1,     # จำนวนครั้งขยายภาพตอนหาใบหน้า (มากขึ้นแม่นขึ้น/ช้าลง)
        "jitters": 1,      # ปรับเพิ่ม (2-3) เพื่อความนิ่งของ encoding
        "margin": 0.03,    # ส่วนต่างระยะ (อันดับ1 - อันดับ2) อย่างน้อยเท่านี้จึงยอมรับในโซน loose
        "min_face": 50,    # กรองใบหน้าที่สั้น/กว้างน้อยกว่าพิกเซลนี้ทิ้ง
        "jpeg_quality": 80, # คุณภาพ JPEG ของสตรีม (60-90 แนะนำ)
        "scale": 0.5,       # ใหม่: ย่อภาพก่อนตรวจ/เข้ารหัส (0.3~1.0) ลดโหลดมาก
        "detect_stride": 3, # ใหม่: ตรวจทุก N เฟรม (เดิม fix 3)
        "alpha": 0.18,      # ใหม่: ค่าหนืดของการ smooth วงกลม (0.05~0.6) ยิ่งต่ำยิ่งนิ่ง
        "max_locs": 6       # ใหม่: จำกัดจำนวนหน้าต่อเฟรม (ช่วยกัน CPU พุ่งในภาพหมู่)
    }
}
config = load_json_safe(CONFIG_FILE, default=_default_cfg)
config.setdefault("det", _default_cfg["det"])
# รักษาค่าที่ขาด
for k, v in _default_cfg["det"].items():
    config["det"].setdefault(k, v)
config.setdefault("windows", [])
save_json(CONFIG_FILE, config)

def _ensure_det_sanity():
    det = config.get("det", {})
    # ความสัมพันธ์และขอบเขต
    if det["loose"] <= det["strict"]:
        det["loose"] = det["strict"] + 0.05
    det["upsample"] = max(0, int(det.get("upsample", 1)))
    det["jitters"] = max(1, int(det.get("jitters", 1)))
    det["margin"] = max(0.0, float(det.get("margin", 0.03)))
    det["min_face"] = max(0, int(det.get("min_face", 50)))
    det["jpeg_quality"] = int(det.get("jpeg_quality", 80))
    if det["jpeg_quality"] < 50: det["jpeg_quality"] = 50
    if det["jpeg_quality"] > 95: det["jpeg_quality"] = 95
    model = str(det.get("model", "hog")).lower()
    det["model"] = "cnn" if model == "cnn" else "hog"

    # ของใหม่
    scale = float(det.get("scale", 0.5))
    scale = 1.0 if scale > 1.0 else (0.3 if scale < 0.3 else scale)
    det["scale"] = scale

    stride = int(det.get("detect_stride", 3))
    det["detect_stride"] = max(1, stride)

    alpha = float(det.get("alpha", 0.18))
    if alpha < 0.05: alpha = 0.05
    if alpha > 0.6:  alpha = 0.6
    det["alpha"] = alpha

    det["max_locs"] = max(1, int(det.get("max_locs", 6)))

    config["det"] = det
    # windows: ต้องเป็นลิสต์ของ {"start":"HH:MM","end":"HH:MM"}
    ws = config.get("windows", [])
    if not isinstance(ws, list):
        ws = []
    good = []
    for w in ws:
        try:
            st = str(w.get("start","")).strip()
            ed = str(w.get("end","")).strip()
            if _is_valid_hhmm(st) and _is_valid_hhmm(ed):
                good.append({"start": st, "end": ed})
        except Exception:
            pass
    config["windows"] = good
    save_json(CONFIG_FILE, config)

def _ensure_smoothers(n: int, alpha: float):
    """ทำให้มี smoothers อย่างน้อย n ตัว และอัปเดตค่าหนืดให้ตรงกับ config ปัจจุบัน"""
    global _smoothers
    if len(_smoothers) < n:
        for _ in range(n - len(_smoothers)):
            _smoothers.append(_SmoothCircle(alpha=alpha))
    for i in range(len(_smoothers)):
        _smoothers[i].alpha = alpha


@app.get("/get_attendance_time")
async def get_attendance_time():
    return {"start": config.get("start"), "end": config.get("end")}

@app.post("/set_attendance_time")
async def set_attendance_time(
    start: str = Form(...), end: str = Form(...),
    _: Any = Depends(auth_guard)
):
    # ตรวจ HH:MM
    if not (_is_valid_hhmm(start) and _is_valid_hhmm(end)):
        raise HTTPException(status_code=400, detail="รูปแบบเวลาไม่ถูกต้อง (ต้องเป็น HH:MM)")
    config["start"], config["end"] = start, end
    save_json(CONFIG_FILE, config)
    return {"message": "อัปเดตเวลาการเช็คชื่อเรียบร้อย ✅", "config": {"start": start, "end": end}}

# ใหม่: กำหนดหลายช่วงเวลาได้ (รองรับทั้งวัน/ช่วงบ่าย)
@app.get("/get_attendance_windows")
async def get_attendance_windows():
    return {"windows": config.get("windows", [])}

@app.post("/set_attendance_windows")
async def set_attendance_windows(
    windows_json: str = Form(..., description='JSON array เช่น [{"start":"08:00","end":"09:30"},{"start":"13:00","end":"15:00"}]'),
    _: Any = Depends(auth_guard)
):
    try:
        arr = json.loads(windows_json)
        if not isinstance(arr, list):
            raise ValueError("ต้องเป็นลิสต์ของช่วงเวลา")
        fixed = []
        for it in arr:
            st = str(it.get("start","")).strip()
            ed = str(it.get("end","")).strip()
            if not (_is_valid_hhmm(st) and _is_valid_hhmm(ed)):
                raise ValueError(f"รูปแบบเวลาไม่ถูกต้อง: {it}")
            fixed.append({"start": st, "end": ed})
        config["windows"] = fixed
        save_json(CONFIG_FILE, config)
        return {"message": "อัปเดตช่วงเวลาเช็คชื่อ (หลายช่วง) เรียบร้อย ✅", "windows": fixed}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"รูปแบบ JSON ไม่ถูกต้อง: {e}")
