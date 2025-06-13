import cv2
import dlib
import numpy as np
import os
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

# === Cấu hình ===
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# === Hàm xử lý ===
def get_face_embedding_full(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    if len(dets) == 0:
        return None
    shape = predictor(gray, dets[0])
    return np.array(face_rec_model.compute_face_descriptor(img, shape))

def get_face_embedding_partial(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    if len(dets) == 0:
        return None
    shape = predictor(gray, dets[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    top = min(landmarks[17:27, 1]) - 20
    bottom = max(landmarks[36:48, 1]) + 10
    left = min(landmarks[0:17, 0])
    right = max(landmarks[0:17, 0])
    h, w, _ = img.shape
    top, bottom = max(0, top), min(h, bottom)
    left, right = max(0, left), min(w, right)

    cropped = img[top:bottom, left:right]
    if cropped.size == 0:
        return None
    resized = cv2.resize(cropped, (150, 150))
    rect = dlib.rectangle(0, 0, 149, 149)
    shape = predictor(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), rect)
    return np.array(face_rec_model.compute_face_descriptor(resized, shape))

def save_embedding(name, embedding, masked=False):
    suffix = "masked" if masked else "normal"
    np.save(os.path.join(EMBEDDING_DIR, f"{name}_{suffix}.npy"), embedding)

def load_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDING_DIR):
        if file.endswith(".npy"):
            parts = file[:-4].split("_")
            if len(parts) == 2:
                name, mode = parts
                emb = np.load(os.path.join(EMBEDDING_DIR, file))
                embeddings[f"{name}_{mode}"] = emb
    return embeddings

def verify_face(embedding, known_embeddings, masked=False, threshold=0.6):
    target_suffix = "masked" if masked else "normal"
    for key, emb in known_embeddings.items():
        if key.endswith(target_suffix):
            sim = cosine_similarity([embedding], [emb])[0][0]
            if sim > threshold:
                return key.replace(f"_{target_suffix}", ""), sim
    return "Unknown", 0.0

# === Chương trình chính ===
mode = input("Chọn chế độ [register / verify]: ").strip().lower()
cap = cv2.VideoCapture(0)

if mode == "register":
    name = input("Nhập tên: ").strip()
    print("[1/2] Đưa mặt vào khung (KHÔNG đeo khẩu trang)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            face_img = frame[y1:y2, x1:x2]

            emb = get_face_embedding_full(face_img)
            if emb is not None:
                save_embedding(name, emb, masked=False)
                print("[✔] Đã lưu khuôn mặt không đeo khẩu trang.")
                break
        cv2.imshow("Đăng ký KHÔNG khẩu trang", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    input("➡️ Đeo khẩu trang rồi nhấn Enter để tiếp tục...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            face_img = frame[y1:y2, x1:x2]

            emb = get_face_embedding_partial(face_img)
            if emb is not None:
                save_embedding(name, emb, masked=True)
                print("[✔] Đã lưu khuôn mặt ĐEO khẩu trang.")
                break
        cv2.imshow("Đăng ký CÓ khẩu trang", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

elif mode == "verify":
    known_embeddings = load_embeddings()
    print("📷 Đưa mặt vào khung (có thể đeo khẩu trang)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                face_img = frame[y1:y2, x1:x2]

                emb = get_face_embedding_partial(face_img)
                name, sim = verify_face(emb, known_embeddings, masked=True)
                if name == "Unknown":
                    emb = get_face_embedding_full(face_img)
                    name, sim = verify_face(emb, known_embeddings, masked=False)

                label = f"{name} ({sim:.2f})" if name != "Unknown" else name
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("Xác thực khuôn mặt", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("❌ Chế độ không hợp lệ. Dùng 'register' hoặc 'verify'")
    cap.release()
