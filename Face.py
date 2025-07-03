import requests
import cv2
import dlib
import numpy as np
import os
import threading
import time
from datetime import datetime

# dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# database
DB_FILE = "face_db.npz"
embeddings = []
labels = []
THRESHOLD = 0.4

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

# API
API_BASE = "http://localhost:8080/api/attendance"
POLL_INTERVAL = 5

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

def compare_embeddings(emb1, emb2):
    dist = np.linalg.norm(emb1 - emb2)
    return dist, dist < THRESHOLD

def recognize_face_once(timeout=60):
    cap = cv2.VideoCapture(0)
    name = None
    start_time = time.time()
    if not cap.isOpened():
        print("[-] Camera error")
        return None
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
                if match and (1 - dist) > max_score:
                    matched_name = db_name
                    max_score = (1 - dist)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0),2)
            cv2.putText(frame, f"{matched_name}", (face.left(), face.top()-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.imshow("Verify", frame)
            name = matched_name
            break

        if name != None:
            break

        if time.time() - start_time > timeout:
            print("[TIMEOUT] 1 phút không ai xác thực")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return name

def poll_pending():
    while True:
        try:
            res = requests.get(API_BASE, timeout=5)
            if res.status_code == 200:
                data = res.json()
                pending = [x for x in data if x["status"]=="pending"]
                for record in pending:
                    print(f"[Pending] ID={record['id']} user={record['user']['username']}")
                    recognized_name = recognize_face_once()
                    if recognized_name == record["user"]["username"]:
                        put_attendance(record["id"], status="in")
                    elif recognized_name == "Unknown" or recognized_name is None:
                        put_attendance(record["id"], status="invalid")
                    else:
                        put_attendance(record["id"], status="invalid")
        except Exception as e:
            print(f"[poll_pending] {e}")
        time.sleep(POLL_INTERVAL)

def poll_in():
    while True:
        try:
            res = requests.get(API_BASE, timeout=5)
            if res.status_code == 200:
                data = res.json()
                ins = [x for x in data if x["status"]=="in"]
                for record in ins:
                    print(f"[In] ID={record['id']} user={record['user']['username']}")
                    recognized_name = recognize_face_once()
                    if recognized_name == record["user"]["username"]:
                        now = datetime.now().isoformat()
                        put_attendance(record["id"], status="complete", checkout_time=now)
                    elif recognized_name == "Unknown" or recognized_name is None:
                        put_attendance(record["id"], status="invalid")
                    else:
                        put_attendance(record["id"], status="invalid")
        except Exception as e:
            print(f"[poll_in] {e}")
        time.sleep(POLL_INTERVAL)

def main():
    t1 = threading.Thread(target=poll_pending, daemon=True)
    t2 = threading.Thread(target=poll_in, daemon=True)
    t1.start()
    t2.start()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
