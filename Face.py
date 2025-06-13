import cv2
import mediapipe as mp
import dlib
import numpy as np
import pickle
import os

# Khởi tạo MediaPipe face detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Tệp dữ liệu khuôn mặt đã đăng ký
DB_PATH = "faces.pkl"
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

def get_face_embedding(image, rect):
    shape = shape_predictor(image, rect)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

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
print("[INFO] Camera started. Nhấn 'r' để đăng ký, 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize nhỏ lại cho nhanh trên Pi
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

            # Nhận diện bằng Dlib
            rects = face_detector(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            if rects:
                rect = rects[0]
                emb = get_face_embedding(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), rect)

                name, dist = recognize_face(emb)
                if name != "Unknown":
                    label = f"✅ {name}"
                    color = (0, 255, 0)
                else:
                    label = "❌ Unverified"
                    color = (0, 0, 255)

                cv2.rectangle(small_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(small_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Auth (press 'r' to register)", small_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('r'):
        name = input("Nhập tên cho khuôn mặt: ")
        print("[INFO] Đưa khuôn mặt không đeo khẩu trang vào khung...")
        cv2.waitKey(3000)
        ret, frame1 = cap.read()
        rects = face_detector(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        if rects:
            emb1 = get_face_embedding(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), rects[0])
            print("[INFO] Bây giờ đeo khẩu trang và nhìn vào khung...")
            cv2.waitKey(5000)
            ret, frame2 = cap.read()
            rects2 = face_detector(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            if rects2:
                emb2 = get_face_embedding(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), rects2[0])
                emb_avg = (emb1 + emb2) / 2
                register_face(name, emb_avg)
            else:
                print("[WARNING] Không tìm thấy khuôn mặt có khẩu trang.")
        else:
            print("[WARNING] Không tìm thấy khuôn mặt.")

cap.release()
cv2.destroyAllWindows()
