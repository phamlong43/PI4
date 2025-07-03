import cv2
import mediapipe as mp
from multiprocessing import Process, Queue
import numpy as np
import tensorflow as tf
import time

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
QUEUE_MAX_SIZE = 2

POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONFIDENCE = 0.7
POSE_MIN_TRACKING_CONFIDENCE = 0.7

NO_POSE_FRAMES_THRESHOLD = 10

# Load mô hình TFLite
interpreter = tf.lite.Interpreter(model_path="MLP_Body.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = {0: "Sit", 1: "Sleep", 2: "Stand"}

def predict_pose(landmarks):
    if len(landmarks) != 33:
        return "Unknown", 0.0

    # Flatten thành vector 99 chiều
    flat = np.array([[coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]], dtype=np.float32)

    # Dự đoán
    interpreter.set_tensor(input_details[0]['index'], flat)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    label_idx = int(np.argmax(output))
    confidence = float(output[0][label_idx])
    return labels[label_idx], confidence

def capture_frame(queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if queue.full():
            try: queue.get_nowait()
            except: pass
        queue.put(frame)
    cap.release()

def pose_inference(input_queue, output_queue):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        model_complexity=POSE_MODEL_COMPLEXITY,
        static_image_mode=False,
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE
    ) as pose:

        no_pose_detected_count = 0
        last_landmarks = None
        label_text = ""
        confidence = 0.0

        while True:
            frame = input_queue.get()
            if frame is None:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                no_pose_detected_count = 0
                last_landmarks = results.pose_landmarks.landmark
                label_text, confidence = predict_pose(last_landmarks)
            else:
                no_pose_detected_count += 1
                if no_pose_detected_count > NO_POSE_FRAMES_THRESHOLD:
                    label_text = "Unknown"
                    confidence = 0.0

            annotated = frame.copy()
            if last_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )

            annotated = cv2.flip(annotated, 1)
            cv2.putText(
                annotated,
                f"{label_text} ({confidence*100:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            if output_queue.full():
                try: output_queue.get_nowait()
                except: pass
            output_queue.put(annotated)

def display_frame(queue):
    while True:
        try:
            frame = queue.get(timeout=1)
        except:
            continue
        if frame is None:
            break
        cv2.imshow("Pose Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_q = Queue(maxsize=QUEUE_MAX_SIZE)
    output_q = Queue(maxsize=QUEUE_MAX_SIZE)

    p_capture = Process(target=capture_frame, args=(input_q,))
    p_infer = Process(target=pose_inference, args=(input_q, output_q))
    p_display = Process(target=display_frame, args=(output_q,))

    p_capture.start()
    p_infer.start()
    p_display.start()

    p_display.join()
    input_q.put(None)
    output_q.put(None)
    p_infer.join()
    p_capture.join()
