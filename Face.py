import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity

CUSTOM_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    234, 93, 132, 127, 162, 107, 55, 103,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 195,
    5, 4, 1, 19, 94,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    67, 69, 108, 109, 151, 337, 299, 298, 301, 447,
    345, 446, 265, 353, 276, 283, 282, 285, 336, 330
]

DB_PATH = "csdl"
os.makedirs(DB_PATH, exist_ok=True)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

def normalize_landmarks(landmarks):
    points = np.array([[landmarks[i].x, landmarks[i].y] for i in CUSTOM_IDX])
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    rotation_matrix = eigvecs
    aligned = centered @ rotation_matrix
    norm = np.linalg.norm(aligned)
    if norm > 0:
        aligned /= norm
    return aligned.flatten()

def luu(name, vec):
    with open(os.path.join(DB_PATH, f"{name}.pkl"), "wb") as f:
        pickle.dump(vec, f)

def tai():
    X, y = [], []
    for file in os.listdir(DB_PATH):
        with open(os.path.join(DB_PATH, file), "rb") as f:
            X.append(pickle.load(f))
            y.append(file[:-4])
    return np.array(X), np.array(y)

MODEL_PATH = "mlp_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)
    X, y = tai()
    if len(X) > 0:
        model.fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

cap = cv2.VideoCapture(0)
print("Nhan 'r' de dang ky, 'q' de thoat.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ketqua = face_mesh.process(rgb)

    if ketqua.multi_face_landmarks:
        lm = ketqua.multi_face_landmarks[0].landmark
        vec = normalize_landmarks(lm)
        try:
            THRESHOLD_PROBA = 0.85
            THRESHOLD_COSINE = 0.85
            probs = model.predict_proba([vec])[0]
            prob = np.max(probs)
            label = model.classes_[np.argmax(probs)]

            if len(model.classes_) == 1:
                if prob == 1.0:
                    db_vecs, db_labels = tai()
                    cos_sim = cosine_similarity([vec], [db_vecs[0]])[0][0]
                    if cos_sim >= THRESHOLD_COSINE:
                        txt = f"{db_labels[0]} ({cos_sim:.2f})"
                    else:
                        txt = "Nguoi la"
                else:
                    txt = "Nguoi la"
            else:
                idx = np.argmax(probs)
                db_vecs, db_labels = tai()
                matched_vec = db_vecs[idx]
                cos_sim = cosine_similarity([vec], [matched_vec])[0][0]
                if prob >= THRESHOLD_PROBA and cos_sim >= THRESHOLD_COSINE:
                    txt = f"{label} ({prob:.2f}, {cos_sim:.2f})"
                else:
                    txt = "Nguoi la"
        except:
            txt = "Chua co model"

        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for i in CUSTOM_IDX:
            p = lm[i]
            x, y = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Nhan dien khuon mat", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        ten = input("Nhap ten nguoi dung: ")
        vecs = []
        print("Dang ky 5 mau mat, vui long giu nguyen khuon mat.")
        for i in range(5):
            input(f"Nhan Enter khi san sang cho anh thu {i+1}...")
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ketqua = face_mesh.process(rgb)
            if ketqua.multi_face_landmarks:
                lm = ketqua.multi_face_landmarks[0].landmark
                vec = normalize_landmarks(lm)
                vecs.append(vec)
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            luu(ten, mean_vec)
            X, y = tai()
            model.fit(X, y)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            print(f"Da luu cho {ten} va cap nhat model")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
