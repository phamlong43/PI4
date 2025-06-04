import pyaudio
import wave
import json
from vosk import Model, KaldiRecognizer
import os
import sys # Thêm thư viện sys để kiểm tra input/output

# --- Cấu hình ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Tần số mẫu (Hz) - Vosk models thường mong đợi 16kHz
CHUNK_SIZE = 4096 # Kích thước khối âm thanh để xử lý (nhỏ hơn cho realtime)

# Đặt đường dẫn đến thư mục mô hình Vosk tiếng Việt đã giải nén của bạn
MODEL_PATH = "model/vosk-model-vi-latest" # Thay thế bằng đường dẫn mô hình tiếng Việt của bạn

# --- Hàm chính ---

def nhan_dang_giong_noi_lien_tuc():
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy thư mục mô hình Vosk tại {MODEL_PATH}")
        print("Vui lòng tải và giải nén mô hình Vosk tiếng Việt vào đúng đường dẫn.")
        print("Bạn có thể tìm các mô hình tại https://alphacephei.com/vosk/models hoặc các nguồn cộng đồng.")
        return

    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Đang lắng nghe và nhận dạng giọng nói liên tục...")
    print("Nói 'thoát' hoặc nhấn Ctrl+C để dừng.")

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False) # Xử lý tràn bộ đệm
            if rec.AcceptWaveform(data):
                ket_qua_json = json.loads(rec.Result())
                if "text" in ket_qua_json and ket_qua_json["text"].strip():
                    print(f"Kết quả: {ket_qua_json['text']}")
                    if ket_qua_json["text"].strip().lower() == "thoát":
                        print("Phát hiện từ 'thoát'. Dừng nhận dạng.")
                        break
            else:
                partial_result_json = json.loads(rec.PartialResult())
                if "partial" in partial_result_json and partial_result_json["partial"].strip():
                    sys.stdout.write(f"\rĐang nhận dạng: {partial_result_json['partial']}   ")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nĐã dừng nhận dạng bằng Ctrl+C.")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi: {e}")
    finally:
        print("\nHoàn thành nhận dạng.")
        stream.stop_stream()
        stream.close()
        p.terminate()

# --- Logic chính ---

if __name__ == "__main__":
    while True:
        print("\n--- Menu Nhận dạng Giọng nói ---")
        print("1. Bắt đầu nhận dạng giọng nói liên tục")
        print("2. Thoát")

        lua_chon = input("Nhập lựa chọn của bạn: ")

        if lua_chon == '1':
            nhan_dang_giong_noi_lien_tuc()
        elif lua_chon == '2':
            print("Đang thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

