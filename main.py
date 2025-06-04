import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# --- Khởi tạo MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Load model TensorFlow Lite ---
interpreter = tf.lite.Interpreter(model_path="action_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Hàm dự đoán ---
def predict_action(sequence):
    input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), output_data

# --- Các hành động ---
actions = ["standing", "walking", "waving", "jumping", "sitting"]  # ví dụ

# --- Chuỗi lưu landmarks ---
seq_length = 30
sequence = deque(maxlen=seq_length)

# --- Mở camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        pose_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        sequence.append(pose_array)

        if len(sequence) == seq_length:
            action_id, confidence = predict_action(np.array(sequence))
            text = f"Action: {actions[action_id]} ({confidence[0][action_id]:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
    else:
        sequence.clear()  # reset nếu không phát hiện người

    cv2.imshow("Human Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
