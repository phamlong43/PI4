import cv2
import dlib
import numpy as np
import threading
import time
import requests
from datetime import datetime
import os

# ========== dlib model ==========
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# ========== face db ==========
DB_FILE = "face_db.npz"
THRESHOLD = 0.4
embeddings = []
labels = []

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

# ========== API ==========
API_ATTENDANCE = "http://localhost:8080/api/attendance"
API_REGISTER_REQUEST = "http://localhost:8080/api/face-register-requests"
POLL_INTERVAL = 5

# ========== shared state ==========
mode = "idle"     # idle, verify, register
target_user = None
target_attendance_id = None
target_register_user_id = None
recognition_start_time = None
lock = threading.Lock()

# ========== utility ==========
def compare_embeddings(emb1, emb2):
    dist = np.linalg.norm(emb1 - emb2)
    return dist, dist < THRESHOLD

def put_attendance(attendance_id, status):
    url = f"{API_ATTENDANCE}/{attendance_id}"
    payload = {"status": status}
    
    now_time = datetime.now().isoformat()
    if status == "in":
        payload["checkIn"] = now_time
    elif status == "complete":
        payload["checkOut"] = now_time
        
    try:
        res = requests.put(url, json=payload, timeout=5)
        print(f"[API] PUT {url} {payload} -> {res.status_code}")
    except Exception as e:
        print(f"[API] Error {e}")

def patch_register_request(request_id, status):
    url = f"{API_REGISTER_REQUEST}/{request_id}"
    payload = {"status": status}
    try:
        res = requests.patch(url, json=payload, timeout=5)
        print(f"[API] PATCH {url} {payload} -> {res.status_code}")
    except Exception as e:
        print(f"[API] Error {e}")

def save_face(name, embedding):
    embeddings.append(embedding)
    labels.append(name)
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)
    print(f"[DB] Saved face for {name}")

# ========== polling thread ==========
def poll_pending():
    global mode, target_user, target_attendance_id, recognition_start_time
    while True:
        try:
            res = requests.get(API_ATTENDANCE, timeout=5)
            if res.status_code == 200:
                data = res.json()
                pending = [x for x in data if x["status"] == "pending"]
                with lock:
                    if mode == "idle" and pending:
                        record = pending[0]
                        target_user = record["user"]["username"]
                        target_attendance_id = record["id"]
                        recognition_start_time = time.time()
                        mode = "verify"
                        print(f"[Pending] Need verify: {target_user} (ID={target_attendance_id})")
        except Exception as e:
            print(f"[poll_pending] {e}")
        time.sleep(POLL_INTERVAL)

def poll_out():
    global mode, target_user, target_attendance_id, recognition_start_time
    while True:
        try:
            res = requests.get(API_ATTENDANCE, timeout=5)
            if res.status_code == 200:
                data = res.json()
                outs = [x for x in data if x["status"] == "out"]
                with lock:
                    if mode == "idle" and outs:
                        record = outs[0]
                        target_user = record["user"]["username"]
                        target_attendance_id = record["id"]
                        recognition_start_time = time.time()
                        mode = "verify_out"
                        print(f"[Out] Need verify checkout: {target_user} (ID={target_attendance_id})")
        except Exception as e:
            print(f"[poll_out] {e}")
        time.sleep(POLL_INTERVAL)

def poll_register_request():
    global mode, target_user, target_register_user_id
    while True:
        try:
            res = requests.get(API_REGISTER_REQUEST, timeout=5)
            if res.status_code == 200:
                data = res.json()
                pending = [x for x in data if x["status"] == "PENDING"]
                with lock:
                    if mode == "idle" and pending:
                        record = pending[0]
                        target_user = record["user"]["username"]
                        target_register_user_id = record["id"]
                        mode = "register"
                        print(f"[Register] Need register for: {target_user} (RequestID={target_register_user_id})")
        except Exception as e:
            print(f"[poll_register_request] {e}")
        time.sleep(POLL_INTERVAL)

# ========== camera loop ==========
def camera_loop():
    global mode, target_user, target_attendance_id, target_register_user_id, recognition_start_time
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        with lock:
            current_mode = mode
            current_user = target_user
            current_attendance_id = target_attendance_id
            current_register_request_id = target_register_user_id
            current_time = recognition_start_time

        for face in faces:
            landmarks = sp(gray, face)
            emb = np.array(face_encoder.compute_face_descriptor(frame, landmarks))

            matched_name = "Unknown"
            max_score = 0

            for db_name, db_emb in zip(labels, embeddings):
                dist, match = compare_embeddings(db_emb, emb)
                score = 1 - dist
                if match and score > max_score:
                    matched_name = db_name
                    max_score = score

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0),2)
            cv2.putText(frame, matched_name, (face.left(), face.top()-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            if current_mode == "verify" and current_user:
                cv2.putText(frame, f"VERIFY: {current_user}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if matched_name == current_user:
                    if current_time and time.time() - current_time <= 60:
                        put_attendance(current_attendance_id, status="in")
                        with lock:
                            mode = "idle"
                            target_user = None
                            target_attendance_id = None
                            recognition_start_time = None
                    elif current_time and time.time() - current_time > 60:
                        put_attendance(current_attendance_id, status="invalid")
                        with lock:
                            mode = "idle"
                            target_user = None
                            target_attendance_id = None
                            recognition_start_time = None

            elif current_mode == "verify_out" and current_user:
                cv2.putText(frame, f"VERIFY OUT: {current_user}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if matched_name == current_user:
                    if current_time and time.time() - current_time <= 60:
                        put_attendance(current_attendance_id, status="complete")
                        with lock:
                            mode = "idle"
                            target_user = None
                            target_attendance_id = None
                            recognition_start_time = None
                    elif current_time and time.time() - current_time > 60:
                        put_attendance(current_attendance_id, status="invalid")
                        with lock:
                            mode = "idle"
                            target_user = None
                            target_attendance_id = None
                            recognition_start_time = None

            elif current_mode == "register" and current_user:
                cv2.putText(frame, f"REGISTER: {current_user}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                save_face(current_user, emb)
                patch_register_request(current_register_request_id, status="DONE")
                with lock:
                    mode = "idle"
                    target_user = None
                    target_register_user_id = None

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== main ==========
def main():
    t1 = threading.Thread(target=poll_pending, daemon=True)
    t2 = threading.Thread(target=poll_out, daemon=True)
    t3 = threading.Thread(target=poll_register_request, daemon=True)
    t1.start()
    t2.start()
    t3.start()
    camera_loop()

if __name__ == "__main__":
    main()
