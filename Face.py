import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.layers as layers

# Custom Conv2D cho các model đã huấn luyện
class StandardizedConv2DWithOverride1(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d((inputs), (kernel - mean) / tf.sqrt(var + 1e-10),
                            padding="VALID", strides=list(self.strides))

class StandardizedConv2DModel2(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        standardized_kernel = (kernel - mean) / tf.sqrt(var + 1e-10)
        return tf.nn.conv2d(inputs, standardized_kernel,
                            padding=self.padding.upper(), strides=list(self.strides))

# Load model AI
emotion_model = load_model('emotion.hdf5',
                           custom_objects={'StandardizedConv2DWithOverride': StandardizedConv2DWithOverride1})
stress_model = load_model('model.h5',
                          custom_objects={'StandardizedConv2DWithOverride': StandardizedConv2DModel2})

emotion_classes = ['surprise', 'fear', 'sadness', 'disgust', 'contempt', 'happy', 'anger']
stress_classes = ['relaxed', 'stress']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img, target_size=(48, 48)):
    face_resized = cv2.resize(face_img, target_size)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    img_array = face_gray.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.repeat(img_array, 3, axis=2)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class StressEmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Nhận diện Stress và Cảm xúc')
        self.apply_styles()

        # Layout chính
        self.image_label = QLabel()
        self.face_label = QLabel()
        self.image_label.setFixedSize(320, 240)
        self.face_label.setFixedSize(160, 160)
        self.image_label.setStyleSheet("border: 1px solid #aaa;")
        self.face_label.setStyleSheet("border: 1px solid #aaa;")

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.image_label)
        images_layout.addWidget(self.face_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        self.select_btn = QPushButton('Chọn ảnh')
        self.open_cam_btn = QPushButton('Mở camera')
        self.capture_btn = QPushButton('Chụp ảnh')
        self.capture_btn.setEnabled(False)
        self.detect_emotion_btn = QPushButton('Nhận diện cảm xúc')
        self.detect_stress_btn = QPushButton('Nhận diện stress')
        self.quit_btn = QPushButton('Thoát')

        # Đặt ObjectName để áp dụng CSS riêng
        self.detect_stress_btn.setObjectName("stressBtn")
        self.detect_emotion_btn.setObjectName("emotionBtn")
        self.quit_btn.setObjectName("quitBtn")

        self.select_btn.clicked.connect(self.load_image)
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.capture_btn.clicked.connect(self.capture_image)
        self.detect_emotion_btn.clicked.connect(self.detect_emotion)
        self.detect_stress_btn.clicked.connect(self.detect_stress)
        self.quit_btn.clicked.connect(self.close)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.select_btn)
        buttons_layout.addWidget(self.open_cam_btn)
        buttons_layout.addWidget(self.capture_btn)
        buttons_layout.addWidget(self.detect_emotion_btn)
        buttons_layout.addWidget(self.detect_stress_btn)
        buttons_layout.addWidget(self.quit_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(images_layout)
        main_layout.addWidget(self.result_text)
        main_layout.addLayout(buttons_layout)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)
        self.setLayout(main_layout)

        self.image = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f4f6f8;
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#stressBtn {
                background-color: #f44336;
            }
            QPushButton#stressBtn:hover {
                background-color: #e53935;
            }
            QPushButton#emotionBtn {
                background-color: #2196F3;
            }
            QPushButton#emotionBtn:hover {
                background-color: #1976D2;
            }
            QPushButton#quitBtn {
                background-color: #9e9e9e;
            }
            QPushButton#quitBtn:hover {
                background-color: #757575;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QLabel {
                background-color: white;
                border-radius: 4px;
            }
        """)

    def load_image(self):
        self.stop_camera()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                qimg = self.convert_cv_to_qt(self.image, self.image_label.width(), self.image_label.height())
                self.image_label.setPixmap(QPixmap.fromImage(qimg))
                self.face_label.clear()
                self.result_text.setText(f"Đã chọn ảnh:\n{file_path}")

    def open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.result_text.setText("Không thể mở camera.")
                self.cap = None
                return
            self.timer.start(30)
            self.open_cam_btn.setText("Tắt camera")
            self.capture_btn.setEnabled(True)
            self.result_text.setText("Đang mở camera. Nhấn 'Chụp ảnh' để nhận diện.")
        else:
            self.stop_camera()

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.open_cam_btn.setText("Mở camera")
        self.capture_btn.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.result_text.setText("Lỗi khi lấy hình từ camera.")
            self.stop_camera()
            return
        self.image = frame
        qimg = self.convert_cv_to_qt(frame, self.image_label.width(), self.image_label.height())
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.face_label.clear()

    def capture_image(self):
        if self.image is None:
            self.result_text.setText("Chưa có ảnh để chụp.")
            return
        self.stop_camera()
        self.result_text.setText("Ảnh đã được chụp. Nhấn 'Nhận diện' để dự đoán.")
        qimg = self.convert_cv_to_qt(self.image, self.image_label.width(), self.image_label.height())
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.face_label.clear()

    def detect_face(self):
        if self.image is None:
            self.result_text.setText("Vui lòng chọn ảnh hoặc chụp ảnh trước.")
            return None
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            self.result_text.setText("Không tìm thấy khuôn mặt.")
            self.face_label.clear()
            return None
        x, y, w, h = faces[0]
        face_img = self.image[y:y + h, x:x + w]
        qimg_face = self.convert_cv_to_qt(face_img, self.face_label.width(), self.face_label.height())
        self.face_label.setPixmap(QPixmap.fromImage(qimg_face))
        return preprocess_face(face_img)

    def detect_emotion(self):
        img_array = self.detect_face()
        if img_array is None:
            return
        try:
            preds = emotion_model.predict(img_array)
            idx = np.argmax(preds, axis=1)[0]
            emotion_result = emotion_classes[idx]
            result = f"Cảm xúc chính: {emotion_result}"
            self.result_text.setText(result)
        except Exception as e:
            self.result_text.setText(f"Lỗi khi dự đoán cảm xúc: {str(e)}")

    def detect_stress(self):
        img_array = self.detect_face()
        if img_array is None:
            return
        try:
            preds_stress = stress_model.predict(img_array)
            stress_prob = preds_stress[0][1]
            stress_result = stress_classes[np.argmax(preds_stress, axis=1)[0]]

            preds_emotion = emotion_model.predict(img_array)
            idx_emotion = np.argmax(preds_emotion, axis=1)[0]
            emotion_result = emotion_classes[idx_emotion]
            emotion_prob = preds_emotion[0][idx_emotion]

            positive_emotions = ['happy', 'surprise']
            negative_emotions = ['fear', 'sadness', 'disgust', 'contempt', 'anger']

            avg_prob = ((stress_prob + emotion_prob) / 2) * 100

            if stress_result == 'stress':
                if emotion_result in negative_emotions:
                    severity = f"Stress nặng ({avg_prob:.2f}%)"
                else:
                    severity = f"Stress nhẹ ({avg_prob:.2f}%)"
            else:
                severity = "Không stress"

            result = f"Kết quả stress:\nRelaxed: {preds_stress[0][0] * 100:.2f}%\nStress: {stress_prob * 100:.2f}%\n"
            result += f"\n--> Tình trạng stress: {severity}\n"
            result += f"--> Cảm xúc chính: {emotion_result} ({emotion_prob * 100:.2f}%)"
            self.result_text.setText(result)
        except Exception as e:
            self.result_text.setText(f"Lỗi khi dự đoán stress: {str(e)}")

    def convert_cv_to_qt(self, cv_img, width=None, height=None):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        if width and height:
            rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StressEmotionDetectionApp()
    window.show()
    sys.exit(app.exec_())
