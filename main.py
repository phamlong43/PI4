import cv2
import mediapipe as mp
from multiprocessing import Process, Queue
import time

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
QUEUE_MAX_SIZE = 2 

POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONFIDENCE = 0.7 
POSE_MIN_TRACKING_CONFIDENCE = 0.7 

def capture_frame(queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"Capture Process: Bắt đầu với độ phân giải {FRAME_WIDTH}x{FRAME_HEIGHT}")

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

def pose_inference(input_queue, output_queue):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        model_complexity=POSE_MODEL_COMPLEXITY,
        static_image_mode=False,
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE
    ) as pose:
        print("Pose Inference Process: Bắt đầu.")
        frame_count = 0 
        
        while True:
            frame = input_queue.get()
            if frame is None:
                print("Pose Inference Process: Nhận tín hiệu thoát, thoát.")
                break

            annotated_frame_to_send = frame.copy() 
            
            frame_count += 1
            
            if frame_count % 3 == 0: 
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                results = pose.process(image)
                
                image.flags.writeable = True

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame_to_send, 
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
            
            annotated_frame_to_send = cv2.flip(annotated_frame_to_send, 1)

            if output_queue.full():
                try:
                    output_queue.get_nowait()
                except Exception:
                    pass

            output_queue.put(annotated_frame_to_send)

    print("Pose Inference Process: Đã thoát.")

def display_frame(queue):
    print("Display Process: Bắt đầu.")
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

        cv2.imshow('Pose Estimation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    print("Display Process: Đã thoát.")

if __name__ == '__main__':
    print("Chương trình chính: Khởi tạo Queues và Processes.")
    input_q = Queue(maxsize=QUEUE_MAX_SIZE)
    output_q = Queue(maxsize=QUEUE_MAX_SIZE)

    p_capture = Process(target=capture_frame, args=(input_q,))
    p_infer = Process(target=pose_inference, args=(input_q, output_q))
    p_display = Process(target=display_frame, args=(output_q,))

    p_capture.start()
    p_infer.start()
    p_display.start()

    print("Chương trình chính: Đang chạy các tiến trình. Nhấn 'q' để thoát.")
    
    p_display.join()

    print("Chương trình chính: Phát hiện Display Process đã thoát. Đang gửi tín hiệu dừng tới các tiến trình khác.")
    input_q.put(None)
    output_q.put(None)
    
    p_infer.join()
    p_capture.join()

    print("Chương trình chính: Tất cả các tiến trình đã thoát. Chương trình kết thúc.")
