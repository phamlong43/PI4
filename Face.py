import cv2
import mediapipe as mp
import dlib
import numpy as np
import pickle
import os

# MediaPipe
mp_face = mp.solutions.face_detection

# Dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load database
DB_PATH = "faces.pkl"
try:
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
    if not isinstance(face_db, dict):
        face_db = {}
except:
    face_db = {}

def get_face_embedding(image, rect):
    shape = shape_predictor(image, rect)
    descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(descriptor)

def register_face(name, embedding):
    face_db[name] = embedding
    with open(DB_PATH, "wb") as f:
        pickle.dump(face_db, f)
    print(f"[INFO] Đã lưu khuôn mặt cho {name}")

def recognize_face(embedding, threshold=0.6):
    for name, db_emb in face_db.items():
        dist = np.linalg.norm(db_emb - embedding)
        if dist < threshold:
            return name, dist
    return "Unknown", None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Không mở được camera.")
    exit()

print("[INFO] Bắt đầu camera. Nhấn 'r' để đăng ký, 'q' để thoát.")

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_small)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = small_frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)
                face_img = small_frame[y1:y2, x1:x2]

                rects = face_detector(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                if rects:
                    emb = get_face_embedding(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), rects[0])
                    name, dist = recognize_face(emb)
                    if name != "Unknown":
                        color = (0, 255, 0)
                        label = f"✅ {name}"
                    else:
                        color = (0, 0, 255)
                        label = "❌ Unknown"

                    cv2.rectangle(small_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(small_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Auth - Press R to Register", small_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            name = input("Nhập tên: ")
            print("[INFO] Chụp mặt không đeo khẩu trang...")
            cv2.waitKey(2000)
            ret, frame1 = cap.read()
            rects = face_detector(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            if rects:
                emb1 = get_face_embedding(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), rects[0])
                print("[INFO] Đeo khẩu trang và giữ nguyên...")
                cv2.waitKey(5000)
                ret, frame2 = cap.read()
                rects2 = face_detector(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
                if rects2:
                    emb2 = get_face_embedding(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), rects2[0])
                    emb_avg = (emb1 + emb2) / 2
                    register_face(name, emb_avg)
                else:
                    print("[WARNING] Không phát hiện mặt có khẩu trang.")
            else:
                print("[WARNING] Không phát hiện khuôn mặt.")

cap.release()
cv2.destroyAllWindows()
