import cv2
import numpy as np
import os
from itertools import combinations
import random
import mediapipe as mp
import dlib

# Load models
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

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

def extract_face_rect_from_landmarks(landmarks, img_shape):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    h, w, _ = img_shape
    left = int(min(x_coords) * w)
    right = int(max(x_coords) * w)
    top = int(min(y_coords) * h)
    bottom = int(max(y_coords) * h)
    return dlib.rectangle(left, top, right, bottom)

def compute_embedding(image, landmarks):
    face_rect = extract_face_rect_from_landmarks(landmarks, image.shape)
    points = dlib.full_object_detection(face_rect, [
        dlib.point(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in landmarks
    ])
    return np.array(face_encoder.compute_face_descriptor(image, points))

def is_face_centered(face_rect, frame_shape, threshold_ratio=0.2):
    face_center_x = (face_rect.left() + face_rect.right()) // 2
    face_center_y = (face_rect.top() + face_rect.bottom()) // 2
    frame_center_x = frame_shape[1] // 2
    frame_center_y = frame_shape[0] // 2
    center_diff = np.linalg.norm([face_center_x - frame_center_x, face_center_y - frame_center_y])
    return center_diff < threshold_ratio * min(frame_shape[0], frame_shape[1])

def register_multi_pose(cap):
    required_poses = ["frontal"]  # Only frontal pose with MediaPipe support

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                face_rect = extract_face_rect_from_landmarks(landmarks, frame.shape)

                if is_face_centered(face_rect, frame.shape):
                    cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, "Nhan 'c' de chup", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.imshow("Dang ky", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        emb = compute_embedding(frame, landmarks)
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            face_rect = extract_face_rect_from_landmarks(landmarks.landmark, frame.shape)
            emb = compute_embedding(frame, landmarks.landmark)

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
