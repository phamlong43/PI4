import cv2
import numpy as np
import os
from itertools import combinations
import random
import mediapipe as mp
import dlib

# Load models
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Approximate mapping: MediaPipe 468-point -> 68 landmark indexes (to simulate Dlib format)
LANDMARK_68_INDEXES = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109, 10, 338, 297, 332,
    284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21
][:68]  # Ensure 68 points

class FakeFullObjectDetection:
    def __init__(self, rect, points):
        self._rect = rect
        self._points = points

    def num_parts(self):
        return len(self._points)

    def part(self, idx):
        return self._points[idx]

    def rect(self):
        return self._rect

DB_FILE = "face_db.npz"
embeddings = []
labels = []
THRESHOLD = 0.5

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

def suggest_optimal_threshold():
    global THRESHOLD
    if len(embeddings) < 2:
        return

    same_dists = []
    diff_dists = []

    for (i1, emb1), (i2, emb2) in combinations(enumerate(embeddings), 2):
        dist = np.linalg.norm(emb1 - emb2)
        if labels[i1] == labels[i2]:
            same_dists.append(dist)
        else:
            diff_dists.append(dist)

    if not diff_dists:
        return

    thresholds = np.linspace(0.2, 5.0, 250)
    best_threshold = 0.5
    best_acc = 0

    for t in thresholds:
        if t >= 0.5:
            continue
        tp = np.sum(np.array(same_dists) <= t)
        fn = np.sum(np.array(same_dists) > t)
        tn = np.sum(np.array(diff_dists) > t)
        fp = np.sum(np.array(diff_dists) <= t)
        acc = (tp + tn) / (tp + tn + fp + fn)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    THRESHOLD = best_threshold

def compare_embeddings(embedding1, embedding2):
    dist = np.linalg.norm(embedding1 - embedding2)
    return dist, dist < THRESHOLD

def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

def compute_embedding(image, face_rect):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    h, w = image.shape[:2]

    points = []
    for idx in LANDMARK_68_INDEXES:
        if idx >= len(landmarks.landmark):
            continue
        lm = landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append(dlib.point(x, y))

    if len(points) != 68:
        return None

    shape = FakeFullObjectDetection(face_rect, points)
    return np.array(face_encoder.compute_face_descriptor(image, shape))

def is_face_centered(face_rect, frame_shape, threshold_ratio=0.2):
    face_center_x = (face_rect.left() + face_rect.right()) // 2
    face_center_y = (face_rect.top() + face_rect.bottom()) // 2
    frame_center_x = frame_shape[1] // 2
    frame_center_y = frame_shape[0] // 2
    center_diff = np.linalg.norm([face_center_x - frame_center_x, face_center_y - frame_center_y])
    return center_diff < threshold_ratio * min(frame_shape[0], frame_shape[1])

def register_multi_pose(cap):
    required_poses = ["frontal"]

    captured_embeddings = []
    name = input("Nhap ten nguoi dung: ").strip()
    if not name:
        print("[-] Ten khong hop le.")
        return
    if name in labels:
        print("[!] Ten da ton tai.")
        return

    print("[*] Huong dan nguoi dung thuc hien tung goc...")
    for pose in required_poses:
        print(f"[{pose.upper()}] Nhin thang vao camera")
        pose_captured = False

        while not pose_captured:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            if len(faces) > 0:
                face_rect = faces[0]

                if is_face_centered(face_rect, frame.shape):
                    cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, "Nhan 'c' de chup", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.imshow("Dang ky", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        emb = compute_embedding(frame, face_rect)
                        if emb is None:
                            continue
                        for saved_emb in embeddings:
                            if np.linalg.norm(emb - saved_emb) < THRESHOLD:
                                print("[!] Khuon mat da ton tai trong he thong.")
                                cv2.destroyWindow("Dang ky")
                                return
                        captured_embeddings.append(emb)
                        print(f"[+] Da chup goc {pose}")
                        pose_captured = True
                else:
                    cv2.putText(frame, "Can giua khung hinh!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Dang ky", frame)
                    cv2.waitKey(1)
            else:
                cv2.putText(frame, "Khong tim thay khuon mat", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Dang ky", frame)
                cv2.waitKey(1)

    if len(captured_embeddings) == len(required_poses):
        avg_embedding = np.mean(captured_embeddings, axis=0)
        embeddings.append(avg_embedding)
        labels.append(name)
        save_db()
        print(f"[+] Dang ky hoan tat cho {name}")
    else:
        print("[!] Dang ky chua hoan tat.")

    cv2.destroyWindow("Dang ky")

def verify_faces_on_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face_rect in faces:
        emb = compute_embedding(frame, face_rect)
        if emb is None:
            continue

        matched_name = "Unknown"
        max_score = 0.0

        for name, reg_emb in zip(labels, embeddings):
            dist, matched = compare_embeddings(reg_emb, emb)
            score = max(0, 1 - dist) * 100
            if matched and score > max_score:
                matched_name = name
                max_score = score

        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
        text = f"{matched_name} ({max_score:.2f}%)"
        cv2.putText(frame, text, (face_rect.left(), face_rect.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[-] Khong the ket noi den stream MJPEG.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[-] Loi doc frame tu stream.")
            continue

        verify_faces_on_frame(frame)

        cv2.putText(frame, "'v': Xac thuc | 'r': Dang ky | 'q': Thoat", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            register_multi_pose(cap)
        if key == ord('q'):
            break

        cv2.imshow("Nhan dien khuon mat", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
