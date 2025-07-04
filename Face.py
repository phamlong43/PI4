import cv2
import dlib
import numpy as np
import threading
import time
import requests
from datetime import datetime

# dlib model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# face db
DB_FILE = "face_db.npz"
THRESHOLD = 0.4
embeddings = []
labels = []

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

# API
API_BASE = "http://localhost:8080/api/attendance"
POLL_INTERVAL = 5

# shared state
target_user = None
target_attendance_id = None
recognition_start_time = None
lock = threading.Lock()

def compare_embeddings(emb1, emb2):
    dist = np.linalg.norm(emb1 - emb2)
    return dist, dist < THRESHOLD

def put_attendance(attendance_id, status, checkout_time=None):
    url = f"{API_BASE}/{attendance_id}"
    payload = {"status": status}
    if checkout_time:
        payload["checkOut"] = checkout_time
    try:
        res = requests.put(url, json=payload, timeout=5)
        print(f"[API] PUT {url} {payload} -> {res.status_code}")
    except Exception as e:
        print(f"[API] Error {e}")

def poll_pending():
    global target_user, target_attendance_id, recognition_start_time
    while True:
        try:
            res = requests.get(API_BASE, timeout=5)
            if res.status_code == 200:
                data = res.json()
                pending = [x for x in data if x["status"]=="pending"]
                for record in pending:
                    with lock:
                        if target_user is None:
                            target_user = record["user"]["username"]
                            target_attendance_id = record["id"]
                            recognition_start_time = time.time()
                            print(f"[Pending] Need verify: {target_user} (ID={target_attendance_id})")
        except Exception as e:
            print(f"[poll_pending] {e}")
        time.sleep(POLL_INTERVAL)

def poll_in():
    global target_user, target_attendance_id, recognition_start_time
    while True:
        try:
            res = requests.get(API_BASE, timeout=5)
            if res.status_code == 200:
                data = res.json()
                ins = [x for x in data if x["status"]=="in"]
                for record in ins:
                    with lock:
                        if target_user is None:
                            target_user = record["user"]["username"]
                            target_attendance_id = record["id"]
                            recognition_start_time = time.time()
                            print(f"[In] Need verify checkout: {target_user} (ID={target_attendance_id})")
        except Exception as e:
            print(f"[poll_in] {e}")
        time.sleep(POLL_INTERVAL)

def camera_loop():
    global target_user, target_attendance_id, recognition_start_time
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

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

            # vẽ bounding box
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0),2)
            cv2.putText(frame, f"{matched_name}", (face.left(), face.top()-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            with lock:
                if target_user:
                    cv2.putText(frame, f"VERIFY: {target_user}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
                    if matched_name == target_user:
                        # thành công trong 60s
                        if recognition_start_time and time.time() - recognition_start_time <= 60:
                            # nếu status pending thì put in
                            put_attendance(target_attendance_id, status="in")
                            target_user = None
                            target_attendance_id = None
                            recognition_start_time = None
                    elif time.time() - recognition_start_time > 60:
                        # hết 60s mà vẫn chưa match đúng thì put invalid
                        put_attendance(target_attendance_id, status="invalid")
                        target_user = None
                        target_attendance_id = None
                        recognition_start_time = None
                    # các trường hợp khác (unknown hoặc người khác) bỏ qua không put gì cả

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    t1 = threading.Thread(target=poll_pending, daemon=True)
    t2 = threading.Thread(target=poll_in, daemon=True)
    t1.start()
    t2.start()
    camera_loop()

if __name__ == "__main__":
    main()
