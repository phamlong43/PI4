import cv2
import mediapipe as mp
import numpy as np
import os
import time
import pickle

# Khởi tạo MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Đường dẫn thư mục lưu embeddings
EMBED_DIR = "embeddings"
if not os.path.exists(EMBED_DIR):
    os.makedirs(EMBED_DIR)

# Hàm lấy embedding từ landmarks (chỉ lấy phần trên khuôn mặt để tránh khẩu trang)
def extract_embedding(face_landmarks):
    upper_ids = list(range(10, 338))  # bỏ phần cằm dưới
    landmarks = []
    for idx in upper_ids:
        lm = face_landmarks.landmark[idx]
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks).flatten()

# Hàm so sánh embedding hiện tại với cơ sở dữ liệu
def recognize(embedding_now, threshold=0.25):
    for file in os.listdir(EMBED_DIR):
        if file.endswith(".npy"):
            name = file[:-4]
            saved_embedding = np.load(os.path.join(EMBED_DIR, file))
            dist = np.linalg.norm(saved_embedding - embedding_now)
            if dist < threshold:
                return name
    return "Unknown"

# Khởi động webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Ấn 'r' để đăng ký, ESC để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể truy cập webcam.")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    frame_out = frame.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame_out,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Tính embedding
            embedding = extract_embedding(face_landmarks)

            # Kiểm tra phím nhấn
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                name = input("Nhập tên để đăng ký: ")
                save_path = os.path.join(EMBED_DIR, f"{name}.npy")
                np.save(save_path, embedding)
                print(f"Đã lưu khuôn mặt của {name}")
            else:
                name = recognize(embedding)
                cv2.putText(frame_out, name, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

    else:
        cv2.putText(frame_out, "Không thấy khuôn mặt", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame_out)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
