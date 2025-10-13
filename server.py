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

# ตั้งค่าการจับ/ความแม่นยำผ่าน API (ขยายจากเดิม)
@app.post("/set_detection")
async def set_detection(
    strict: float = Form(None), loose: float = Form(None),
    votes: int = Form(None), window: float = Form(None),
    model: str = Form(None), upsample: int = Form(None),
    jitters: int = Form(None), margin: float = Form(None),
    min_face: int = Form(None), jpeg_quality: int = Form(None),
    # ใหม่
    scale: float = Form(None), detect_stride: int = Form(None),
    alpha: float = Form(None), max_locs: int = Form(None),
    _: Any = Depends(auth_guard)
):
    det = config["det"]
    if strict is not None: det["strict"] = float(strict)
    if loose  is not None: det["loose"]  = float(loose)
    if votes  is not None: det["votes"]  = max(1, int(votes))
    if window is not None: det["window"] = max(0.5, float(window))
    if model  is not None: det["model"]  = str(model).lower()
    if upsample is not None: det["upsample"] = int(upsample)
    if jitters  is not None: det["jitters"]  = int(jitters)
    if margin   is not None: det["margin"]   = float(margin)
    if min_face is not None: det["min_face"] = int(min_face)
    if jpeg_quality is not None: det["jpeg_quality"] = int(jpeg_quality)
    if scale is not None: det["scale"] = float(scale)
    if detect_stride is not None: det["detect_stride"] = int(detect_stride)
    if alpha is not None: det["alpha"] = float(alpha)
    if max_locs is not None: det["max_locs"] = int(max_locs)
    _ensure_det_sanity()
    return {"message": "อัปเดตโหมดการจับสำเร็จ ✅", "det": config["det"]}

# ---------- Attendance ----------
_att_lock = threading.Lock()

def load_attendance():
    return load_json_safe(ATTEND_FILE, default=[])

def save_attendance(data):
    save_json(ATTEND_FILE, data)

def load_logs():
    return load_json_safe(LOG_FILE, default=[])

def save_logs(data):
    save_json(LOG_FILE, data)

# ===== จัดการ student_id =====
_faces_lock = threading.Lock()
faces_data = load_json_safe(FACES_DB, default={})

def _hm_lt(a, b):
    """Return True if time a < b; ถ้า a == b → ครบวัน"""
    if a == b:
        return False  # ครบวัน
    return (a[0], a[1]) < (b[0], b[1])


def _migrate_faces_data():
    """
    รองรับรูปแบบเก่า/ใหม่:
      - เก่า: name: [enc, enc, ...] หรือ name: [128 ตัวเลข]
      - ใหม่: name: { "encodings":[...], "student_id": "XXXX" }
    """
    changed = False
    if not isinstance(faces_data, dict):
        return

    for name, val in list(faces_data.items()):
        if isinstance(val, dict):
            encs = val.get("encodings", [])
            if isinstance(encs, list) and encs and isinstance(encs[0], (int, float)) and len(encs) == 128:
                val["encodings"] = [encs]
                changed = True
            sid = val.get("student_id", "")
            if sid is None:
                sid = ""
            val["student_id"] = str(sid)
            faces_data[name] = val
            continue

        if isinstance(val, list):
            if val and isinstance(val[0], (int, float)) and len(val) == 128:
                new_obj = {"encodings": [val], "student_id": ""}
            else:
                new_obj = {"encodings": val, "student_id": ""}
            faces_data[name] = new_obj
            changed = True

    if changed:
        save_json(FACES_DB, faces_data)

_migrate_faces_data()

def _get_student_id(name: str) -> str:
    v = faces_data.get(name)
    if isinstance(v, dict):
        sid = v.get("student_id")
        return str(sid) if sid is not None else ""
    return ""

def _get_encodings_list(name: str):
    v = faces_data.get(name)
    if isinstance(v, dict):
        return v.get("encodings", [])
    if isinstance(v, list):
        return v
    return []

def add_log(name: str, student_id: str = ""):
    logs = load_logs()
    logs.append({"name": name, "student_id": student_id or "", "time": now_datetime_str()})
    save_logs(logs)

def mark_attendance(name: str):
    """กันเช็คซ้ำภายในวันเดียวกัน + log real-time (บันทึก student_id ด้วย)"""
    today = now_date_str()
    nowt = now_time_str()
    sid = _get_student_id(name)
    with _att_lock:
        data = load_attendance()
        for rec in data:
            if rec.get("name") == name and rec.get("date") == today:
                if not rec.get("student_id"):
                    rec["student_id"] = sid
                    save_attendance(data)
                return {"new": False, **rec}
        rec = {"date": today, "time": nowt, "name": name, "student_id": sid}
        data.append(rec)
        save_attendance(data)
        add_log(name, student_id=sid)
        return {"new": True, **rec}

# ---------- Export helpers ----------
def _write_csv_fallback(path: str, records: list):
    if not records:
        return False
    fieldnames = set()
    for r in records:
        if isinstance(r, dict):
            fieldnames |= set(r.keys())
    fieldnames = list(fieldnames) or ["date", "time", "name"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            if isinstance(r, dict):
                w.writerow({k: r.get(k, "") for k in fieldnames})
    return True

def _write_xlsx_fallback(path: str, records: list):
    if not (records and HAVE_OPENPYXL):
        return False
    wb = Workbook()
    ws = wb.active
    cols = set()
    for r in records:
        if isinstance(r, dict):
            cols |= set(r.keys())
    cols = list(cols) or ["date", "time", "name"]
    ws.append(cols)
    for r in records:
        if isinstance(r, dict):
            ws.append([r.get(c, "") for c in cols])
    wb.save(path)
    return True

def _export_records(records: list, date_token: str):
    """บันทึก CSV/XLSX เป็น attendance_YYYYMMDD.{csv,xlsx} และคืนรายชื่อไฟล์ที่สร้างจริง"""
    created = []
    if not records:
        return created
    csv_path = f"attendance_{date_token}.csv"
    xlsx_path = f"attendance_{date_token}.xlsx"
    if HAVE_PANDAS:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        created.append(csv_path)
        if HAVE_OPENPYXL:
            df.to_excel(xlsx_path, index=False)
            created.append(xlsx_path)
        else:
            print("[EXPORT] openpyxl ไม่พร้อม → เขียนเฉพาะ CSV")
    else:
        if _write_csv_fallback(csv_path, records):
            created.append(csv_path)
        if HAVE_OPENPYXL and _write_xlsx_fallback(xlsx_path, records):
            created.append(xlsx_path)
    return created

def _collect_records_for_date(date_token: str) -> list:
    """ดึง records ทั้งหมดของวันที่ระบุ (YYYYMMDD) จาก SESSIONS_FILE"""
    ymd_dash = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}"
    sessions = load_json_safe(SESSIONS_FILE, default=[])
    bag = []
    for sess in sessions:
        recs = sess.get("records", [])
        for r in recs:
            if isinstance(r, dict) and r.get("date") == ymd_dash:
                bag.append(r)
    return bag

def _ensure_export_files_for_date(date_token: str):
    """ถ้ายังไม่มีไฟล์ export สำหรับวันที่กำหนด ให้ลองสร้างจาก sessions"""
    csv_path = f"attendance_{date_token}.csv"
    xlsx_path = f"attendance_{date_token}.xlsx"
    if os.path.exists(csv_path) or os.path.exists(xlsx_path):
        return
    recs = _collect_records_for_date(date_token)
    if recs:
        created = _export_records(recs, date_token)
        if created:
            print(f"[EXPORT/REBUILD] saved: {', '.join(created)}")

# ---------- Sessions & Round control ----------
is_checking_active = False  # เปิด/ปิดการเช็คชื่อ (ควบคุมโดยเวลา)

def _is_now_in_window(start_hm: str, end_hm: str) -> bool:
    """
    คืนค่า True ถ้าเวลาปัจจุบันอยู่ในช่วง start..end
    รองรับกรณีข้ามเที่ยงคืน: start > end (เช่น 23:00 -> 01:00)
    """
    if not start_hm or not end_hm:
        return False
    st = _hm_to_tuple(start_hm)
    ed = _hm_to_tuple(end_hm)
    if not st or not ed:
        return False
    now = _now_hm_tuple()
    if _hm_lt(st, ed):
        return (not _hm_lt(now, st)) and _hm_lt(now, ed)
    else:
        return (not _hm_lt(now, st)) or _hm_lt(now, ed)

def _is_now_in_any_window() -> bool:
    """ถ้ากำหนด windows ไว้ จะใช้ windows แทน start/end"""
    wins = config.get("windows", [])
    if isinstance(wins, list) and wins:
        for w in wins:
            st, ed = w.get("start"), w.get("end")
            if _is_now_in_window(st, ed):
                return True
        return False
    # fallback: ใช้ start/end เดิม
    return _is_now_in_window(config.get("start"), config.get("end"))

def start_new_session():
    """เริ่มรอบใหม่ → ล้าง attendance และ logs และเปิดโหมดเช็คชื่อ"""
    global is_checking_active
    save_attendance([])
    save_logs([])  # เริ่มรอบใหม่ตัด log เก่าออก
    is_checking_active = True
    print(f"[SESSION] Started at {now_datetime_str()}")

def end_session_and_save():
    """ปิดรอบ → บันทึก session + export CSV/XLSX แล้วปิดโหมดเช็คชื่อ"""
    global is_checking_active
    current = load_attendance()
    sessions = load_json_safe(SESSIONS_FILE, default=[])

    if current:
        now_str = now_datetime_str()
        sessions.append({
            "round_end": now_str,
            "records": current
        })
        save_json(SESSIONS_FILE, sessions)

        ymd = _yyyymmdd_from_today()
        created = _export_records(current, ymd)
        print(f"[EXPORT] saved: {', '.join(created)}")

    # ไม่ล้าง logs ตอนจบ เพื่อให้หน้า Summary ยังสรุปผลได้หลังปิดรอบ
    save_attendance([])
    is_checking_active = False
    print(f"[SESSION] Ended at {now_datetime_str()}")

def session_manager():
    """ควบคุมเวลา start-end/windows ให้เริ่ม/หยุดเองอัตโนมัติ (รองรับข้ามเที่ยงคืน/หลายช่วง)"""
    global is_checking_active
    while True:
        try:
            active_now = _is_now_in_any_window()
            if active_now and not is_checking_active:
                start_new_session()
            if (not active_now) and is_checking_active:
                end_session_and_save()
        except Exception as e:
            print("Session manager error:", e)
        time.sleep(20)

threading.Thread(target=session_manager, daemon=True).start()

# ---------- Faces (many encodings per person) ----------
known_names, known_encodings = [], []

# เพิ่มเมทริกซ์สำหรับคำนวณระยะให้เสถียร/เร็ว (ลดหน่วยความจำด้วย float32)
_known_enc_matrix = np.zeros((0, 128), dtype=np.float32)

def _rebuild_memory_from_faces():
    names, encs = [], []
    for name, val in faces_data.items():
        vecs = []
        if isinstance(val, dict):
            vecs = val.get("encodings", [])
        elif isinstance(val, list):
            vecs = val
        if isinstance(vecs, list) and vecs and isinstance(vecs[0], (int, float)) and len(vecs) == 128:
            vecs = [vecs]
        for vec in vecs:
            try:
                # เก็บเป็น float64 ระหว่างโหลด แล้วค่อยแปลงเป็น float32 ตอนรวมเมทริกซ์
                encs.append(np.array(vec, dtype=np.float64))
                names.append(name)
            except Exception:
                pass
    return names, encs

def load_faces_into_memory():
    global known_names, known_encodings, _known_enc_matrix
    with _faces_lock:
        known_names, known_encodings = _rebuild_memory_from_faces()
        if known_encodings:
            # แปลงเป็น float32 เพื่อลดหน่วยความจำ ~ครึ่งหนึ่ง
            _known_enc_matrix = np.vstack(known_encodings).astype(np.float32, copy=False)
        else:
            _known_enc_matrix = np.zeros((0, 128), dtype=np.float32)

def reload_all_from_disk():
    """โหลดข้อมูล/คอนฟิกทั้งหมดจากดิส์ก, migrate, และรีเซ็ตสถานะ session"""
    global faces_data, config, is_checking_active
    with _faces_lock:
        faces_data = load_json_safe(FACES_DB, default={})
        _migrate_faces_data()
        load_faces_into_memory()
    cfg = load_json_safe(CONFIG_FILE, default=_default_cfg)
    cfg.setdefault("det", _default_cfg["det"])
    for k, v in _default_cfg["det"].items():
        cfg["det"].setdefault(k, v)
    cfg.setdefault("windows", [])
    config = cfg
    _ensure_det_sanity()
    is_checking_active = False
    save_attendance([])
    # ไม่ลบ logs ที่นี่

load_faces_into_memory()
_ensure_det_sanity()

# ---------- Dashboard ----------
@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- APIs ----------
@app.get("/")
async def root():
    return {"message": "Face Recognition API Running ✅", "registered_count": len(faces_data)}

@app.get("/get_status")
async def get_status():
    return {
        "active": is_checking_active,
        "server_time": now_datetime_str(),
        "start": config.get("start"),
        "end": config.get("end"),
        "windows": config.get("windows", [])
    }

@app.get("/list_faces")
async def list_faces():
    with _faces_lock:
        name_list = list(faces_data.keys())

        # map ชื่อ -> student_id
        name_to_sid = {n: faces_data[n].get("student_id", "") for n in name_list}

        # map ชื่อ -> จำนวน encodings
        counts = {n: len(faces_data[n].get("encodings", [])) for n in name_list}

    return {
        "registered_faces": name_list,
        "count": len(name_list),
        "map": name_to_sid,
        "enc_counts": counts
    }


@app.post("/register_face")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...),
    student_id: str = Form(None),
    _: Any = Depends(auth_guard)
):
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="ชื่อว่าง")
    sid = str(student_id).strip() if student_id is not None else ""

    rgb = decode_upload_to_rgb(file)

    # ใช้พารามิเตอร์ตรวจจับตาม config
    det = config.get("det", {})
    model = det.get("model", "hog")
    upsample = det.get("upsample", 1)
    jitters = det.get("jitters", 1)
    min_face = int(det.get("min_face", 50))
    scale = float(det.get("scale", 0.5))

    # ย่อภาพเพื่อลดโหลด
    if scale < 1.0:
        small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        locations = face_recognition.face_locations(small, number_of_times_to_upsample=upsample, model=model)
        # สเกลกลับมาเพื่อใช้กับภาพต้นฉบับ
        def _up(loc):
            t, r, b, l = loc
            return (int(t/scale), int(r/scale), int(b/scale), int(l/scale))
        locations = [_up(x) for x in locations]
    else:
        locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample, model=model)

    if not locations:
        raise HTTPException(status_code=400, detail="ไม่พบใบหน้าในภาพ")

    # เลือกใบหน้าที่ใหญ่สุด เพื่อกันกรณีมีหลายคนในภาพ
    def _area(box):
        t, r, b, l = box
        return max(0, (r - l)) * max(0, (b - t))
    locations = [loc for loc in locations if min(loc[1]-loc[3], loc[2]-loc[0]) >= min_face]
    if not locations:
        raise HTTPException(status_code=400, detail="ใบหน้าเล็กเกินไป/ไกลเกินไป")

    best_loc = max(locations, key=_area)
    encs = face_recognition.face_encodings(rgb, known_face_locations=[best_loc], num_jitters=jitters)
    if not encs:
        raise HTTPException(status_code=400, detail="เข้ารหัสใบหน้าไม่สำเร็จ")
    new_vec = encs[0].astype(np.float32).tolist()  # ลดความจุ

    with _faces_lock:
        cur = faces_data.get(name)
        if cur is None:
            faces_data[name] = {"encodings": [], "student_id": sid}
        elif isinstance(cur, list):
            faces_data[name] = {"encodings": cur, "student_id": sid or ""}
        elif isinstance(cur, dict):
            if sid:
                faces_data[name]["student_id"] = sid

        vecs = faces_data[name]["encodings"]
        # กันซ้ำ: ถ้าใกล้มากกับของเดิม (ระยะ < 0.35) จะไม่เพิ่ม
        is_dup = False
        for old in vecs:
            try:
                d = float(np.linalg.norm(np.array(old, dtype=np.float32) - np.array(new_vec, dtype=np.float32)))
            except Exception:
                d = 1.0
            if d < 0.35:
                is_dup = True
                break
        if not is_dup:
            vecs.append(new_vec)

        save_json(FACES_DB, faces_data)

    load_faces_into_memory()
    return {
        "message": f"บันทึกใบหน้าของ '{name}' สำเร็จ ✅ (encodings={len(faces_data[name]['encodings'])})",
        "total_registered": len(faces_data),
        "student_id": faces_data[name].get("student_id", "")
    }

@app.post("/rename_face")
async def rename_face(
    old_name: str = Form(...),
    new_name: str = Form(...),
    _: Any = Depends(auth_guard)
):
    old_name, new_name = old_name.strip(), new_name.strip()
    if not old_name or not new_name:
        raise HTTPException(status_code=400, detail="old_name/new_name ว่าง")
    with _faces_lock:
        if old_name not in faces_data:
            raise HTTPException(status_code=404, detail=f"ไม่พบชื่อ '{old_name}'")
        if new_name in faces_data:
            raise HTTPException(status_code=409, detail=f"มีชื่อ '{new_name}' อยู่แล้ว")
        faces_data[new_name] = faces_data.pop(old_name)
        save_json(FACES_DB, faces_data)
    # ล้างโหวตค้าง
    _recent_votes.pop(old_name, None)
    load_faces_into_memory()
    return {"message": f"เปลี่ยนชื่อ '{old_name}' → '{new_name}' สำเร็จ ✅"}

@app.post("/update_student_id")
async def update_student_id(
    name: str = Form(...),
    student_id: str = Form(...),
    _: Any = Depends(auth_guard)
):
    name = name.strip()
    sid = str(student_id).strip()
    with _faces_lock:
        if name not in faces_data:
            raise HTTPException(status_code=404, detail=f"ไม่พบชื่อ '{name}'")
        v = faces_data[name]
        if isinstance(v, dict):
            v["student_id"] = sid
        else:
            faces_data[name] = {"encodings": v, "student_id": sid}
        save_json(FACES_DB, faces_data)
    return {"message": f"อัปเดต student_id ของ '{name}' เป็น '{sid}' เรียบร้อย ✅"}

@app.delete("/delete_face")
async def delete_face(name: str = Query(...), _: Any = Depends(auth_guard)):
    with _faces_lock:
        if name not in faces_data:
            raise HTTPException(status_code=404, detail=f"ไม่พบชื่อ '{name}'")
        del faces_data[name]
        save_json(FACES_DB, faces_data)
    _recent_votes.pop(name, None)
    load_faces_into_memory()
    return {"message": f"ลบข้อมูลของ '{name}' เรียบร้อย ✅", "total_registered": len(faces_data)}

# --------- Soft/Strict matching helpers ----------
_recent_votes = defaultdict(deque)  # name -> deque[timestamps]

def _vote_loose(name: str, window: float):
    dq = _recent_votes[name]
    t = time.time()
    dq.append(t)
    while dq and t - dq[0] > window:
        dq.popleft()
    return len(dq)

def _distances_to_known(enc_vec: np.ndarray) -> np.ndarray:
    """
    คำนวณระยะกับทุกคนแบบเวกเตอร์ไรซ์ ใช้ float32 เพื่อลดโหลด
    """
    global _known_enc_matrix
    if _known_enc_matrix.size == 0:
        return np.empty((0,), dtype=np.float32)
    enc32 = enc_vec.astype(np.float32, copy=False)
    # ||A - b|| สำหรับแต่ละแถว A_i
    diff = _known_enc_matrix - enc32
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    return dists.astype(np.float32, copy=False)

def _match_name_with(enc: np.ndarray, strict: float, loose: float, margin: float):
    """
    ใช้ _known_enc_matrix (Nx128) และ known_names (N) ที่เตรียมไว้แล้ว
    คืนค่า (name, mode, best_dist)
    - mode = "strict"|"loose"|None
    - margin: ส่วนต่างระยะ (อันดับ2 - อันดับ1) ขั้นต่ำสำหรับยอมรับในโซน loose
    """
    global _known_enc_matrix, known_names
    if _known_enc_matrix.size == 0:
        return None, None, None

    # คำนวณระยะทั้งหมดในคราวเดียว (float32)
    dists = _distances_to_known(enc.astype(np.float32, copy=False))
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])

    # เงื่อนไข strict
    if best_dist <= strict:
        return known_names[best_idx], "strict", best_dist

    # เงื่อนไข loose + margin check
    if best_dist <= loose:
        if dists.size >= 2:
            sorted_d = np.sort(dists)
            second = float(sorted_d[1]) if sorted_d.shape[0] > 1 else float("inf")
        else:
            second = float("inf")
        if (second - best_dist) >= margin:
            return known_names[best_idx], "loose", best_dist

    return None, None, best_dist

@app.post("/identify_face")
async def identify_face(
    file: UploadFile = File(...),
    tolerance: float = Query(None, ge=0.3, le=0.9),   # override strict
    loose: float = Query(None, ge=0.35, le=1.0),      # override loose
    need_votes: int = Query(None, ge=1, le=5),        # override votes
    vote_window: float = Query(None, ge=0.5, le=10.0),# override window
    mark: bool = Query(False, description="true=บันทึกเช็คชื่อเมื่อ match (ค่าเริ่มต้น False)")
):
    """
    - ไม่บันทึกเช็คชื่ออัตโนมัติ (ค่าเริ่มต้น mark=False)
    - ถ้าต้องบันทึกจริง ให้ส่ง ?mark=1
    """
    rgb = decode_upload_to_rgb(file)

    det = config.get("det", {})
    model = det.get("model", "hog")
    upsample = det.get("upsample", 1)
    jitters = det.get("jitters", 1)
    min_face = int(det.get("min_face", 50))
    margin   = float(det.get("margin", 0.03))
    scale    = float(det.get("scale", 0.5))
    max_locs = int(det.get("max_locs", 6))

    # ตรวจจับที่ภาพย่อ แล้วค่อยสเกลพิกัดกลับ
    if scale < 1.0:
        small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        locations_small = face_recognition.face_locations(small, number_of_times_to_upsample=upsample, model=model)
        def _up(loc):
            t, r, b, l = loc
            return (int(t/scale), int(r/scale), int(b/scale), int(l/scale))
        locations = [_up(x) for x in locations_small]
    else:
        locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample, model=model)

    if not locations:
        raise HTTPException(status_code=400, detail="ไม่พบใบหน้า")

    # กรองใบหน้าเล็ก + จำกัดจำนวนสูงสุด/เรียงจากใหญ่ไปเล็ก
    filt_locs = [loc for loc in locations if min(loc[1]-loc[3], loc[2]-loc[0]) >= min_face]
    if not filt_locs:
        raise HTTPException(status_code=400, detail="ใบหน้าเล็กเกินไป/ไกลเกินไป")
    filt_locs.sort(key=lambda b: (b[2]-b[0])*(b[1]-b[3]), reverse=True)
    filt_locs = filt_locs[:max_locs]

    # เข้ารหัสบนภาพเต็ม (ความแม่นยำดีกว่า) — หากอยากเร่งสุด ให้ใช้ small แทน
    encodings = face_recognition.face_encodings(rgb, filt_locs, num_jitters=jitters)
    if not encodings:
        raise HTTPException(status_code=400, detail="เข้ารหัสใบหน้าไม่สำเร็จ")

    det_cfg = config.get("det", {})
    strict_thr = float(tolerance if tolerance is not None else det_cfg.get("strict", 0.62))
    loose_thr  = float(loose     if loose     is not None else det_cfg.get("loose", 0.75))
    votes_need = int(need_votes  if need_votes is not None else det_cfg.get("votes", 2))
    win_secs   = float(vote_window if vote_window is not None else det_cfg.get("window", 3.0))
    if loose_thr <= strict_thr:
        loose_thr = strict_thr + 0.05

    results = []
    for enc in encodings:
        name, mode, dist = _match_name_with(enc, strict_thr, loose_thr, margin)
        if name is None:
            results.append({"name": "Unknown", "student_id": "", "matched": False, "distance": dist})
            continue

        sid = _get_student_id(name)
        if not mark:
            results.append({"name": name, "student_id": sid, "matched": (mode in ("strict", "loose")), "distance": dist, "mode": mode or "none"})
            continue

        if mode == "strict":
            result = mark_attendance(name)
            results.append({"name": name, "student_id": sid, "matched": True, "distance": dist, "mode": "strict", "new": result["new"]})
        else:
            c = _vote_loose(name, win_secs)
            if c >= votes_need:
                _recent_votes[name].clear()
                result = mark_attendance(name)
                results.append({"name": name, "student_id": sid, "matched": True, "distance": dist, "mode": "loose-confirm", "new": result["new"]})
            else:
                results.append({"name": name, "student_id": sid, "matched": False, "distance": dist, "mode": "loose-wait", "votes": c})

    return {"results": results, "thresholds": {"strict": strict_thr, "loose": loose_thr, "votes": votes_need, "window": win_secs, "margin": margin}, "mark": mark}

@app.get("/attendance")
async def get_attendance(limit: int = Query(50, ge=1, le=500)):
    return load_attendance()[-limit:]

@app.delete("/attendance")
async def clear_attendance(_: Any = Depends(auth_guard)):
    save_attendance([])
    save_logs([])
    return {"message": "ลบข้อมูลเช็คชื่อเรียบร้อยแล้ว ✅"}

@app.get("/attendance_sessions")
async def get_sessions():
    return load_json_safe(SESSIONS_FILE, default=[])

# ---------- Logs & Summary ----------
def _normalize_logs(raw_logs):
    norm = []
    for it in raw_logs:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if name is None and isinstance(it.get("detail"), dict):
            name = it["detail"].get("name")
        if name is None:
            name = "Unknown"
        t = it.get("time") or it.get("timestamp")
        if t is None and it.get("date") and it.get("time"):
            t = f"{it['date']} {it['time']}"
        if t is None:
            t = now_datetime_str()
        sid = it.get("student_id")
        if not sid:
            sid = _get_student_id(name)
        norm.append({"name": name, "student_id": sid or "", "time": t})
    return norm

@app.get("/attendance_logs")
async def attendance_logs(limit: int = Query(50, ge=1, le=500)):
    logs = _normalize_logs(load_logs())
    return {"logs": logs[-limit:]}

@app.get("/logs")
async def logs_alias(limit: int = Query(50, ge=1, le=500)):
    logs = _normalize_logs(load_logs())
    return {"logs": logs[-limit:]}

@app.delete("/logs")
async def clear_logs_api(_: Any = Depends(auth_guard)):
    save_logs([])
    return {"message": "ล้าง logs เรียบร้อยแล้ว ✅"}

@app.get("/attendance_summary")
async def attendance_summary(
    date: str = Query(None, description="รูปแบบ YYYYMMDD หรือ YYYY-MM-DD; เว้นว่าง=วันนี้")
):
    """
    สรุปผลการเช็คชื่อของวันที่กำหนด (หรือวันนี้ถ้าไม่ส่ง)
    ใช้ข้อมูลจาก logs ของวันนั้นก่อน → ถ้าว่าง fallback ไปที่ attendance.json → ถ้ายังว่าง ใช้ sessions
    """
    if date:
        try:
            date_token = _normalize_date_token(date)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        date_token = _yyyymmdd_from_today()
    ymd_dash = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:8]}"

    raw_logs = _normalize_logs(load_logs())
    logs = [x for x in raw_logs if isinstance(x.get("time"), str) and x["time"].startswith(ymd_dash)]

    if not logs:
        current_att = [r for r in load_attendance() if r.get("date") == ymd_dash]
        for r in current_att:
            logs.append({
                "name": r.get("name", "Unknown"),
                "student_id": r.get("student_id", ""),
                "time": f"{r.get('date','')} {r.get('time','')}".strip()
            })

    if not logs:
        recs = _collect_records_for_date(date_token)
        for r in recs:
            logs.append({
                "name": r.get("name", "Unknown"),
                "student_id": r.get("student_id", ""),
                "time": f"{r.get('date','')} {r.get('time','')}".strip()
            })

    summary_counts = {}
    first_times = {}
    for item in logs:
        key = f"{item.get('name','Unknown')}|{item.get('student_id','')}"
        summary_counts[key] = summary_counts.get(key, 0) + 1
        t = item.get("time", "")
        if key not in first_times and t:
            first_times[key] = t

    pretty = []
    for k, cnt in summary_counts.items():
        nm, sid = k.split("|", 1)
        pretty.append({
            "name": nm,
            "student_id": sid,
            "count": cnt,
            "first_time": first_times.get(k, "")
        })

    pretty.sort(key=lambda x: (x["name"], x["student_id"]))

    return {"summary": pretty, "total": len(logs), "date": ymd_dash}

# ============== Real-time Camera Stream ==============

def _open_camera():
    cam_index = int(os.environ.get("CAM_INDEX", "0"))
    cap = cv2.VideoCapture(cam_index)
    # ตั้งค่าพื้นฐาน
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # พยายามใช้ MJPG เพื่อให้ดึงเฟรมลื่นขึ้น/CPU ต่ำลง (ถ้ากล้องรองรับ)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    # ลดดีเลย์ buffer (ถ้าซัพพอร์ต)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

# โครงสร้างช่วยในการ smooth วงกลม
class _SmoothCircle:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.cx = None
        self.cy = None
        self.r = None

    def update(self, cx, cy, r, alpha_override: float = None):
        a = self.alpha if alpha_override is None else alpha_override
        if self.cx is None:
            self.cx, self.cy, self.r = cx, cy, r
        else:
            self.cx = int((1 - a) * self.cx + a * cx)
            self.cy = int((1 - a) * self.cy + a * cy)
            self.r  = int((1 - a) * self.r  + a * r)
        return self.cx, self.cy, self.r

_smoothers: List[_SmoothCircle] = []

def _ensure_smoothers(n, alpha: float):
    """ทำให้มี smoothers อย่างน้อย n ตัว และอัปเดตค่าหนืดให้ตรงกับ config ปัจจุบัน"""
    global _smoothers
    if len(_smoothers) < n:
        for _ in range(n - len(_smoothers)):
            _smoothers.append(_SmoothCircle(alpha=alpha))
    # อัปเดต alpha ของตัวที่มีอยู่
    for i in range(min(n, len(_smoothers))):
        _smoothers[i].alpha = alpha
    return _smoothers[:n]

def _frame_generator():
    """
    - ตรวจจับตำแหน่งใบหน้าและวาดวงสีเขียว "ตลอดเวลา"
    - เข้ารหัสใบหน้า/ทำ matching + mark_attendance เฉพาะช่วง active (ตาม windows/start/end)
    - โหลดค่า strict/loose/votes/window และ det-params ใหม่ทุกลูป (hot update)
    - ปรับ pipeline ให้ลื่น/CPU ต่ำ: ใช้ grab()/retrieve(), ย่อภาพก่อน detect, ตรวจทุก N เฟรม
    """
    cap = _open_camera()
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้")
        return

    idx = 0
    cached_locs = []         # พิกัดบน "ภาพเต็ม" เพื่อวาดได้คม
    cache_expire_ts = 0.0    # หมดอายุ cache เมื่อครบเวลา

    try:
        while True:
            # ดึงเฟรมล่าสุดเพื่อลดหน่วง
            if not cap.grab():
                cap.release()
                time.sleep(0.2)
                cap = _open_camera()
                continue
            ok, frame = cap.retrieve()
            if not ok:
                cap.release()
                time.sleep(0.2)
                cap = _open_camera()
                continue

            now_ts = time.time()

            det = config.get("det", {})
            model = det.get("model", "hog")
            upsample = det.get("upsample", 1)
            jitters = det.get("jitters", 1)
            min_face = int(det.get("min_face", 50))
            strict_thr = float(det.get("strict", 0.62))
            loose_thr  = float(det.get("loose", 0.75))
            votes_need = int(det.get("votes", 2))
            win_secs   = float(det.get("window", 3.0))
            margin     = float(det.get("margin", 0.03))
            jpeg_q     = int(det.get("jpeg_quality", 80))
            scale      = float(det.get("scale", 0.5))
            stride     = int(det.get("detect_stride", 3))
            max_locs   = int(det.get("max_locs", 6))
            alpha      = float(det.get("alpha", 0.18))
            if loose_thr <= strict_thr:
                loose_thr = strict_thr + 0.05

            need_detect = (idx % max(1, stride) == 0) or (now_ts > cache_expire_ts)

            if need_detect:
                # แปลงเป็น RGB และย่อก่อนหาใบหน้า
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale < 1.0:
                    rgb_small = cv2.resize(rgb_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    locations_small = face_recognition.face_locations(
                        rgb_small, number_of_times_to_upsample=upsample, model=model
                    )
                    # กรอง/จำกัดจำนวน และสเกลพิกัดกลับ
                    cand = []
                    for loc in locations_small:
                        t, r, b, l = loc
                        t, r, b, l = int(t/scale), int(r/scale), int(b/scale), int(l/scale)
                        if min(r-l, b-t) >= min_face:
                            cand.append((t, r, b, l))
                    # เรียงใบหน้าใหญ่ก่อนและจำกัดจำนวน
                    cand.sort(key=lambda b: (b[2]-b[0])*(b[1]-b[3]), reverse=True)
                    locations = cand[:max_locs]
                else:
                    locations = face_recognition.face_locations(
                        rgb_full, number_of_times_to_upsample=upsample, model=model
                    )
                    locations = [loc for loc in locations if min(loc[1]-loc[3], loc[2]-loc[0]) >= min_face]
                    # จำกัดจำนวน
                    locations.sort(key=lambda b: (b[2]-b[0])*(b[1]-b[3]), reverse=True)
                    locations = locations[:max_locs]

                cached_locs = locations[:]
                cache_expire_ts = now_ts + 0.45  # cache สั้น ๆ ป้องกันสั่น/กิน CPU

                # ทำ matching เฉพาะช่วง active
                if is_checking_active and locations:
                    # เข้ารหัสจากภาพ "เต็ม" เพื่อความแม่นยำ (สมดุลระหว่างความลื่น/ความถูกต้อง)
                    encodings = face_recognition.face_encodings(rgb_full, known_face_locations=locations, num_jitters=jitters)
                    if encodings:
                        for enc in encodings:
                            name, mode, dist = _match_name_with(enc, strict_thr, loose_thr, margin)
                            if name is None:
                                continue
                            if mode == "strict":
                                result = mark_attendance(name)
                                if result["new"]:
                                    print(f"[ATTEND(strict)] {name} @ {result['time']} (d={dist:.3f})")
                            else:
                                c = _vote_loose(name, win_secs)
                                if c >= votes_need:
                                    _recent_votes[name].clear()
                                    result = mark_attendance(name)
                                    if result["new"]:
                                        print(f"[ATTEND(loose)] {name} @ {result['time']} (d={dist:.3f})")

            # วาดวงสีเขียวรอบหน้า จาก cached_locs → ใช้ smoothing (alpha ปรับได้สด ๆ)
            smoothers = _ensure_smoothers(len(cached_locs), alpha=alpha)
            for i, (top, right, bottom, left) in enumerate(cached_locs):
                cx = int((left + right) / 2)
                cy = int((top + bottom) / 2)
                radius = int(0.55 * max(right - left, bottom - top))
                scx, scy, sr = smoothers[i].update(cx, cy, radius, alpha_override=alpha)
                cv2.circle(frame, (scx, scy), sr, (0, 255, 0), 3)

            # เข้ารหัส JPEG ปรับคุณภาพได้
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
            if ok2:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            idx += 1
    finally:
        cap.release()

@app.get("/video_feed")
def video_feed():
    # ถ้าเปิดกล้องไม่ได้ → แจ้ง 503 ชัดเจน
    test = _open_camera()
    ok = test.isOpened()
    test.release()
    if not ok:
        raise HTTPException(status_code=503, detail="ไม่สามารถเปิดกล้องได้")
    
    
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ---------- Export API ----------
@app.get("/export_today")
async def export_today(
    file_type: str = Query("csv", pattern="^(csv|xlsx)$"),
    _: Any = Depends(auth_guard)
):
    """ดาวน์โหลดรายงานการเช็คชื่อวันนี้ (ไฟล์ที่ export ตอนปิดรอบ) — ถ้าไม่มีจะพยายามสร้างย้อนหลังจาก sessions"""
    date_token = _yyyymmdd_from_today()
    path = f"attendance_{date_token}.{file_type}"

    if not os.path.exists(path):
        if file_type == "xlsx" and not HAVE_OPENPYXL:
            raise HTTPException(status_code=400, detail="ยังไม่สามารถส่ง XLSX ได้ (ไม่มี openpyxl) — ใช้ CSV แทน")
        _ensure_export_files_for_date(date_token)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ยังไม่มีไฟล์ของวันนี้ (รอบยังไม่ปิดหรือไม่มีข้อมูล)")

    media = "text/csv" if file_type == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(path, media_type=media, filename=os.path.basename(path))

@app.get("/export_session")
async def export_session(
    date: str = Query(..., description="รูปแบบ YYYYMMDD หรือ YYYY-MM-DD"),
    file_type: str = Query("csv", pattern="^(csv|xlsx)$"),
    _: Any = Depends(auth_guard)
):
    """ดาวน์โหลดรายงานของวันที่กำหนด (ถ้าไม่มีไฟล์จะสร้างย้อนหลังจาก sessions)"""
    try:
        date_token = _normalize_date_token(date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    path = f"attendance_{date_token}.{file_type}"

    if not os.path.exists(path):
        if file_type == "xlsx" and not HAVE_OPENPYXL:
            raise HTTPException(status_code=400, detail="ยังไม่สามารถส่ง XLSX ได้ (ไม่มี openpyxl) — ใช้ CSV แทน")
        _ensure_export_files_for_date(date_token)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"ไม่พบไฟล์สำหรับ {date} และไม่พบข้อมูลใน sessions")

    media = "text/csv" if file_type == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(path, media_type=media, filename=os.path.basename(path))

# ---------- Backup / Restore ----------
@app.get("/backup")
async def backup(_: Any = Depends(auth_guard)):
    """แพ็กไฟล์สำคัญทั้งหมดเป็น backup.zip ให้ดาวน์โหลด (รวมรายงาน attendance_*.csv/.xlsx)"""
    files = [CONFIG_FILE, FACES_DB, ATTEND_FILE, SESSIONS_FILE, LOG_FILE]
    files += glob.glob("attendance_*.csv")
    files += glob.glob("attendance_*.xlsx")

    zip_path = "backup.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            if os.path.exists(p):
                zf.write(p)
    return FileResponse(zip_path, media_type="application/zip", filename="backup.zip")

def _safe_extract_all(zf: zipfile.ZipFile, target_dir: str):
    """ป้องกัน Zip Slip: ไม่ยอมให้ไฟล์ทะลุออกนอก target_dir"""
    target_dir = os.path.abspath(target_dir)
    for member in zf.infolist():
        fname = member.filename
        # ป้องกัน absolute path หรือ '..'
        if os.path.isabs(fname) or ".." in os.path.normpath(fname).split(os.sep):
            raise HTTPException(status_code=400, detail=f"ไฟล์ไม่ปลอดภัยใน zip: {fname}")
        dest = os.path.abspath(os.path.join(target_dir, fname))
        if not dest.startswith(target_dir):
            raise HTTPException(status_code=400, detail=f"ไฟล์ไม่ปลอดภัยใน zip: {fname}")
    zf.extractall(target_dir)

@app.post("/restore")
async def restore(file: UploadFile = File(...), _: Any = Depends(auth_guard)):
    """อัปโหลด backup.zip เพื่อคืนค่าข้อมูลทั้งหมด แล้วรีโหลดใบหน้า/คอนฟิกเข้าหน่วยความจำ"""
    # ตรวจสกุลไฟล์ (ป้องกันอัปไฟล์อื่น)
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ .zip")
    data = await file.read()
    try:
        buf = io.BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            _safe_extract_all(zf, ".")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="ไฟล์ zip ไม่ถูกต้อง")
    # รีโหลดทั้งหมด
    reload_all_from_disk()
    return {"message": "Restore สำเร็จ และรีโหลดข้อมูลทั้งหมดแล้ว ✅"}

@app.post("/reload_all")
async def reload_all(_: Any = Depends(auth_guard)):
    """โหลดข้อมูล/คอนฟิกจากไฟล์ + migrate + รีเซ็ต session state (ไม่ลบ logs)"""
    reload_all_from_disk()
    return {"message": "รีโหลดข้อมูลจากดิสก์เรียบร้อย ✅"}

@app.get("/student")
async def student(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})
