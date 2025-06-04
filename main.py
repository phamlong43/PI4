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

NO_POSE_FRAMES_THRESHOLD = 10 

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
        last_valid_pose_landmarks = None 
        no_pose_detected_count = 0 
        
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
                    last_valid_pose_landmarks = results.pose_landmarks
                    no_pose_detected_count = 0 
                else:
                    no_pose_detected_count += 1 
            
            if no_pose_detected_count >= NO_POSE_FRAMES_THRESHOLD:
                last_valid_pose_landmarks = None 
            
            if last_valid_pose_landmarks:
