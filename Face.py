import dlib
import cv2
import numpy as np
import os
import time
import random
from itertools import combinations

# Load model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

THRESHOLD = 0.4
POSES = ["frontal", "looking left", "looking right", "looking up", "looking down"]

def get_pose_filename(pose):
    return f"face_db_{pose.replace(' ', '_')}.npz"

def load_pose_data(pose):
    filename = get_pose_filename(pose)
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        return list(data["embeddings"]), list(data["labels"])
    return [], []

def save_pose_data(pose, embeddings, labels):
    filename = get_pose_filename(pose)
    np.savez(filename, embeddings=embeddings, labels=labels)

def compute_embedding(image, face_rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = sp(gray, face_rect)
    embedding = face_encoder.compute_face_descriptor(image, landmarks)
    return np.array(embedding)

def is_face_centered(face, frame_shape, threshold_ratio=0.2):
    face_center_x = (face.left() + face.right()) // 2
    face_center_y = (face.top() + face.bottom()) // 2
    frame_center_x = frame_shape[1] // 2
    frame_center_y = frame_shape[0] // 2
    center_diff = np.linalg.norm([face_center_x - frame_center_x, face_center_y - frame_center_y])
    return center_diff < threshold_ratio * min(frame_shape[0], frame_shape[1])

def get_pose_direction(landmarks):
    nose = landmarks.part(30)
    chin = landmarks.part(8)
    forehead = landmarks.part(27)
    left_cheek = landmarks.part(2)
    right_cheek = landmarks.part(14)

    face_height = chin.y - forehead.y
    nose_chin_dist = chin.y - nose.y
    vertical_ratio = nose_chin_dist / face_height if face_height > 0 else 0

    face_width = right_cheek.x - left_cheek.x
    nose_left_dist = nose.x - left_cheek.x
    horizontal_ratio = nose_left_dist / face_width if face_width > 0 else 0

    if vertical_ratio < 0.65:
        return "looking down"
    elif vertical_ratio > 0.8:
        return "looking up"
    if horizontal_ratio > 0.6:
        return "looking left"
    elif horizontal_ratio < 0.4:
        return "looking right"
    return "frontal"

def register_multi_pose(cap):
    required_poses = {
        "frontal": "Nhin thang vao camera",
        "looking left": "Nghieng trai",
        "looking right": "Nghieng phai",
        "looking up": "Ngan len",
        "looking down": "Cui xuong"
    }

    captured_embeddings = []
    name = input("Nhap ten nguoi dung: ").strip()
    if not name:
        print("[-] Ten khong hop le.")
        return

    for pose in required_poses:
        pose_embs, pose_lbls = load_pose_data(pose)
        if name in pose_lbls:
            print(f"[!] Ten da ton tai trong goc {pose}.")
            return

    os.makedirs("videos", exist_ok=True)
    ret, frame = cap.read()
    if not ret:
        print("[-] Loi khi doc frame dau tien.")
        return

    frame_height, frame_width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"videos/{name}_{int(time.time())}.avi"
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))

    print("[*] Huong dan nguoi dung thuc hien tung goc...")
    for pose, message in required_poses.items():
        print(f"[{pose.upper()}] {message}")
        pose_captured = False
        pose_start_time = None
        while not pose_captured:
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            display_frame = frame.copy()

            if len(faces) == 1:
                face = faces[0]
                if is_face_centered(face, frame.shape):
                    landmarks = sp(gray, face)
                    detected_pose = get_pose_direction(landmarks)

                    cv2.rectangle(display_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{message}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    if detected_pose == pose:
                        cv2.putText(display_frame, f"Goc {pose} dung - Dang chup...", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if pose_start_time is None:
                            pose_start_time = time.time()
                        elif time.time() - pose_start_time >= 1.0:
                            emb = compute_embedding(frame, face)

                            if pose == "frontal":
                                existing_embs, existing_lbls = load_pose_data("frontal")
                                for saved_emb in existing_embs:
                                    dist = np.linalg.norm(emb - saved_emb)
                                    if dist < THRESHOLD:
                                        print("[!] Khuon mat da ton tai trong he thong.")
                                        video_writer.release()
                                        cv2.destroyAllWindows()
                                        return

                            captured_embeddings.append((pose, emb))
                            print(f"[+] Da chup goc {pose}")
                            pose_captured = True
                            pose_start_time = None
                    else:
                        pose_start_time = None
                        cv2.putText(display_frame, f"Goc hien tai: {detected_pose}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    pose_start_time = None
                    cv2.putText(display_frame, "Can giua khung hinh!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                pose_start_time = None
                cv2.putText(display_frame, "Can co 1 khuon mat trong khung!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Dang ky", display_frame)
            cv2.waitKey(1)

    video_writer.release()

    if len(captured_embeddings) == len(required_poses):
        for pose, emb in captured_embeddings:
            pose_embs, pose_lbls = load_pose_data(pose)
            pose_embs.append(emb)
            pose_lbls.append(name)
            save_pose_data(pose, pose_embs, pose_lbls)
        print(f"[+] Dang ky hoan tat cho {name}")
        print(f"[+] Da luu video dang ky: {video_filename}")
    else:
        print("[!] Dang ky chua hoan tat.")

    cv2.destroyAllWindows()

def verify_faces_on_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = sp(gray, face)
        detected_pose = get_pose_direction(landmarks)
        pose_embeddings, pose_labels = load_pose_data(detected_pose)

        embedding = face_encoder.compute_face_descriptor(frame, landmarks)
        embedding = np.array(embedding)

        matched_name = "Unknown"
        max_score = 0.0

        for name, reg_emb in zip(pose_labels, pose_embeddings):
            dist = np.linalg.norm(reg_emb - embedding)
            score = max(0, 1 - dist) * 100
            if dist < THRESHOLD and score > max_score:
                matched_name = name
                max_score = score

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        text = f"{matched_name} ({max_score:.2f}%)"
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Pose: {detected_pose}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Khong the mo camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        verify_faces_on_frame(frame)

        cv2.putText(frame, "'r': Dang ky | 'q': Thoat", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Nhan dien khuon mat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            register_multi_pose(cap)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
