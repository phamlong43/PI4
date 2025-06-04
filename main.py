import cv2
import mediapipe as mp
from multiprocessing import Process, Queue
import time # Thêm thư viện time để kiểm soát tốc độ khung hình và debug

# --- Cấu hình cho Raspberry Pi 4 ---
# Giảm độ phân giải để xử lý nhanh hơn trên Pi4
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Kích thước Queue:
# - Quá lớn: tăng độ trễ (latency), tiêu tốn RAM
# - Quá nhỏ: dễ bị bỏ lỡ frame nếu các tiến trình không đồng bộ
# Maxsize 2-3 thường là tốt cho Pi4 để giảm latency mà vẫn đủ buffer.
QUEUE_MAX_SIZE = 3 

# Cấu hình MediaPipe Pose:
# model_complexity=0 (nhẹ nhất, nhanh nhất) là lựa chọn tối ưu cho Pi4
# static_image_mode=False (tối ưu cho video stream)
# min_detection_confidence và min_tracking_confidence có thể điều chỉnh
# để cân bằng giữa độ chính xác và tốc độ. Giảm nhẹ có thể tăng tốc.
POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# --- Các hàm xử lý ---

def capture_frame(queue):
    """
    Tiến trình bắt frame từ camera và đưa vào queue.
    Giảm độ phân giải và có thể giới hạn FPS.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Thử đặt FPS. Không phải camera nào cũng hỗ trợ chính xác.
    # cap.set(cv2.CAP_PROP_FPS, 30) # Đặt thử 30 FPS, có thể điều chỉnh
    
    print(f"Capture Process: Bắt đầu với độ phân giải {FRAME_WIDTH}x{FRAME_HEIGHT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Capture Process: Không thể đọc frame, thoát.")
            break
        
        # Nếu queue đầy, bỏ frame cũ nhất để luôn có frame mới nhất
        if queue.full():
            try:
                queue.get_nowait() # Sử dụng get_nowait để không bị chặn
            except Exception as e:
                # print(f"Capture Process: Lỗi khi bỏ frame cũ: {e}")
                pass # Bỏ qua lỗi nếu queue trống bất ngờ (hiếm)

        queue.put(frame)
        
        # Có thể thêm một delay nhỏ nếu CPU của bạn quá tải hoặc bạn muốn FPS ổn định
        # time.sleep(0.01) # Delay 10ms tương đương khoảng 100 FPS tối đa, chỉ dùng khi cần
        
    cap.release()
    print("Capture Process: Đã thoát.")

def pose_inference(input_queue, output_queue):
    """
    Tiến trình suy luận pose bằng MediaPipe.
    Xử lý frame từ input_queue và đưa kết quả vào output_queue.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Khởi tạo MediaPipe Pose với cấu hình tối ưu cho Pi4
    with mp_pose.Pose(
        model_complexity=POSE_MODEL_COMPLEXITY,
        static_image_mode=False,
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE
    ) as pose:
        print("Pose Inference Process: Bắt đầu.")
        while True:
            # Chờ và lấy frame từ input_queue. Hàm .get() sẽ chặn cho đến khi có frame.
            frame = input_queue.get() 
            if frame is None: # Tín hiệu thoát (nếu sử dụng .put(None) để báo hiệu kết thúc)
                print("Pose Inference Process: Nhận tín hiệu thoát, thoát.")
                break

            # Chuyển đổi màu sắc và đặt cờ writeable cho MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Xử lý suy luận pose
            results = pose.process(image)
            
            image.flags.writeable = True # Đặt lại cờ sau khi xử lý

            # Tạo bản sao của frame để vẽ lên mà không ảnh hưởng đến frame gốc (nếu cần)
            annotated_frame = frame.copy()

            # Vẽ các điểm pose lên frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2), # Xanh lá
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2) # Đỏ
                )
            
            # Nếu output_queue đầy, bỏ frame cũ nhất
            if output_queue.full():
                try:
                    output_queue.get_nowait()
                except Exception as e:
                    # print(f"Pose Inference Process: Lỗi khi bỏ frame cũ: {e}")
                    pass # Bỏ qua lỗi

            output_queue.put(annotated_frame)
    print("Pose Inference Process: Đã thoát.")

def display_frame(queue):
    """
    Tiến trình hiển thị frame lên màn hình.
    """
    print("Display Process: Bắt đầu.")
    while True:
        # Lấy frame nếu có. Sử dụng .get_nowait() để không bị chặn nếu queue trống.
        # Hoặc .get() và kiểm tra None nếu bạn muốn báo hiệu kết thúc.
        try:
            frame = queue.get(timeout=1) # Chờ tối đa 1 giây, nếu không có sẽ raise Empty
        except Exception: # queue.Empty
            # print("Display Process: Queue hiển thị trống.")
            if cv2.waitKey(1) & 0xFF == ord('q'): # Vẫn kiểm tra phím thoát ngay cả khi không có frame mới
                break
            continue # Tiếp tục vòng lặp

        if frame is None: # Tín hiệu thoát (nếu sử dụng .put(None) để báo hiệu kết thúc)
            print("Display Process: Nhận tín hiệu thoát, thoát.")
            break

        cv2.imshow('Pose Estimation on RPi4', frame)
        
        # Kiểm tra phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    print("Display Process: Đã thoát.")

# --- Chương trình chính ---

if __name__ == '__main__':
    print("Chương trình chính: Khởi tạo Queues và Processes.")
    # Khởi tạo Queues với kích thước tối ưu cho Pi4
    input_q = Queue(maxsize=QUEUE_MAX_SIZE)  # Từ capture -> inference
    output_q = Queue(maxsize=QUEUE_MAX_SIZE) # Từ inference -> display

    # Khởi tạo các tiến trình (Processes)
    p_capture = Process(target=capture_frame, args=(input_q,))
    p_infer = Process(target=pose_inference, args=(input_q, output_q))
    p_display = Process(target=display_frame, args=(output_q,))

    # Bắt đầu các tiến trình
    p_capture.start()
    p_infer.start()
    p_display.start()

    print("Chương trình chính: Đang chạy các tiến trình. Nhấn 'q' để thoát.")
    
    # Chờ các tiến trình kết thúc
    # Trong môi trường thực tế, bạn có thể muốn một cơ chế thoát mềm hơn,
    # ví dụ như gửi tín hiệu None vào queue khi muốn dừng.
    p_display.join() # Chờ tiến trình hiển thị kết thúc (khi người dùng nhấn 'q')

    # Khi tiến trình display kết thúc, chúng ta cần gửi tín hiệu để các tiến trình khác thoát
    print("Chương trình chính: Phát hiện Display Process đã thoát. Đang gửi tín hiệu dừng tới các tiến trình khác.")
    input_q.put(None) # Gửi tín hiệu dừng tới inference process
    output_q.put(None) # Gửi tín hiệu dừng tới display process (phòng hờ)
    
    p_infer.join()
    p_capture.join()

    print("Chương trình chính: Tất cả các tiến trình đã thoát. Chương trình kết thúc.")
