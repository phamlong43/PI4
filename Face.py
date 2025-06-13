import cv2
import numpy as np
import os
from itertools import combinations
import random
import mediapipe as mp

DB_FILE = "face_db.npz"
embeddings = []
labels = []
THRESHOLD = 0.8

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

LANDMARK_INDEXES = list(range(468))

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

    thresholds = np.linspace(0.2, 2.0, 250)
    best_threshold = 0.8
    best_acc = 0

    for t in thresholds:
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

def compute_embedding(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    points = []
    for idx in LANDMARK_INDEXES:
        lm = face_landmarks.landmark[idx]
        points.extend([lm.x, lm.y, lm.z])

    return np.array(points)

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

            emb = compute_embedding(frame)
            if emb is not None:
                cv2.putText(frame, "Nhan 'c' de chup", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Dang ky", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    for saved_emb in embeddings:
                        if np.linalg.norm(emb - saved_emb) < THRESHOLD:
                            print("[!] Khuon mat da ton tai trong he thong.")
                            cv2.destroyWindow("Dang ky")
                            return
                    captured_embeddings.append(emb)
                    print(f"[+] Da chup goc {pose}")
                    pose_captured = True
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
    emb = compute_embedding(frame)
    if emb is None:
        return

    matched_name = "Unknown"
    max_score = 0.0

    for name, reg_emb in zip(labels, embeddings):
        dist, matched = compare_embeddings(reg_emb, emb)
        score = max(0, 1 - dist / THRESHOLD) * 100
        if matched and score > max_score:
            matched_name = name
            max_score = score

    h, w = frame.shape[:2]
    cv2.putText(frame, f"{matched_name} ({max_score:.2f}%)", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    stream_url = "https://3945-2402-800-6106-e118-ec93-b42a-93f3-6d7b.ngrok-free.app/video_feed"
    cap = cv2.VideoCapture(stream_url)

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
