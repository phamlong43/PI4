import cv2
import mediapipe as mp
from multiprocessing import Process, Queue
import time

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
QUEUE_MAX_SIZE = 3

HANDS_MAX_NUM_HANDS = 2
HANDS_MIN_DETECTION_CONFIDENCE = 0.5
HANDS_MIN_TRACKING_CONFIDENCE = 0.5

def capture_frame(queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Capture Process: Lỗi - Không thể mở camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Capture Process: Đang chạy...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Capture Process: Không thể đọc frame, thoát.")
            break

        if queue.full():
            try:
                queue.get_nowait()
            except Exception:
                pass 

        queue.put(frame)
        
    cap.release()
    print("Capture Process: Đã thoát.")

def hand_inference(input_queue, output_queue):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=HANDS_MAX_NUM_HANDS,
        min_detection_confidence=HANDS_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=HANDS_MIN_TRACKING_CONFIDENCE
    )

    frame_count = 0
    last_processed_frame = None

    print("Inference Process: Đang chạy...")
    while True:
        frame = input_queue.get() 
        if frame is None:
            print("Inference Process: Nhận tín hiệu thoát, thoát.")
            break

        frame_count += 1
        
        if frame_count % 2 == 0: 
            frame_flipped = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False

            results = hands.process(img_rgb)

            img_rgb.flags.writeable = True

            annotated_frame = frame_flipped.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                    )
                last_processed_frame = annotated_frame
            else:
                if last_processed_frame is not None:
                    annotated_frame = last_processed_frame.copy()
                else:
                    annotated_frame = frame_flipped.copy()

            if output_queue.full():
                try:
                    output_queue.get_nowait()
                except Exception:
                    pass

            output_queue.put(annotated_frame)
        else:
            if last_processed_frame is not None:
                if output_queue.full():
                    try:
                        output_queue.get_nowait()
                    except Exception:
                        pass
                output_queue.put(last_processed_frame.copy())
            else:
                if output_queue.full():
                    try:
                        output_queue.get_nowait()
                    except Exception:
                        pass
                output_queue.put(cv2.flip(frame, 1).copy())

    print("Inference Process: Đã thoát.")

def display_frame(queue):
    print("Display Process: Đang chạy...")

    while True:
        try:
            frame = queue.get(timeout=1) 
        except Exception: 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if frame is None:
            print("Display Process: Nhận tín hiệu thoát, thoát.")
            break

        cv2.imshow("Hand Gesture (Optimized)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Display Process: Đã thoát.")

if __name__ == '__main__':
    print("Chương trình chính: Khởi tạo Queues và Processes.")
    input_q = Queue(maxsize=QUEUE_MAX_SIZE)
    output_q = Queue(maxsize=QUEUE_MAX_SIZE)

    p_capture = Process(target=capture_frame, args=(input_q,))
    p_infer = Process(target=hand_inference, args=(input_q, output_q))
    p_display = Process(target=display_frame, args=(output_q,))

    p_capture.start()
    p_infer.start()
    p_display.start()

    print("Chương trình chính: Các tiến trình đang chạy. Nhấn 'q' trên cửa sổ hiển thị để thoát.")
    
    p_display.join()

    print("Chương trình chính: Display Process đã thoát. Đang gửi tín hiệu dừng...")
    input_q.put(None) 
    output_q.put(None) 

    p_infer.join()
    p_capture.join()

    print("Chương trình chính: Tất cả các tiến trình đã thoát. Chương trình kết thúc.")
