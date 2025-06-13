import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from multiprocessing import Process, Queue
import time

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
QUEUE_MAX_SIZE = 3

def capture_frame(queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Capture: Lỗi - Không mở được camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Capture: Đang chạy...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Capture: Không đọc được frame.")
            break

        if queue.full():
            try: queue.get_nowait()
            except: pass

        queue.put(frame)

    cap.release()
    print("Capture: Đã thoát.")


def hand_inference(input_queue, output_queue):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    interpreter = tf.lite.Interpreter(model_path="MLP_Hand.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    le = joblib.load("label_encoder.pkl")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Inference: Đang chạy...")

    while True:
        frame = input_queue.get()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        annotated_frame = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                if len(landmark_list) == 63:
                    input_data = np.array(landmark_list, dtype=np.float32).reshape(1, -1)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])

                    label_index = np.argmax(prediction)
                    confidence = prediction[0][label_index]
                    gesture_label = le.inverse_transform([label_index])[0]

                    print(f"[Dự đoán] {gesture_label} ({confidence*100:.1f}%)")

                    cv2.putText(
                        annotated_frame,
                        f"{gesture_label} ({confidence*100:.1f}%)",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        2
                    )
                else:
                    print("Không đủ điểm landmark.")
        else:
            print("Không thấy bàn tay.")

        if output_queue.full():
            try: output_queue.get_nowait()
            except: pass
        output_queue.put(annotated_frame)


def display_frame(queue):
    print("Display: Đang chạy...")
    while True:
        try:
            frame = queue.get(timeout=1)
        except:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if frame is None:
            break

        cv2.imshow("Hand Gesture (TFLite)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Display: Đã thoát.")


if __name__ == '__main__':
    print("Main: Khởi tạo các tiến trình...")
    input_q = Queue(maxsize=QUEUE_MAX_SIZE)
    output_q = Queue(maxsize=QUEUE_MAX_SIZE)

    p_capture = Process(target=capture_frame, args=(input_q,))
    p_infer = Process(target=hand_inference, args=(input_q, output_q))
    p_display = Process(target=display_frame, args=(output_q,))

    p_capture.start()
    p_infer.start()
    p_display.start()

    print("Main: Đang chạy. Nhấn 'q' để thoát.")

    p_display.join()
    input_q.put(None)
    output_q.put(None)
    p_infer.join()
    p_capture.join()
    print("Main: Tất cả tiến trình đã thoát.")
