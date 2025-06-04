import pyaudio
import wave
import json
from vosk import Model, KaldiRecognizer

# --- Cấu hình ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Tần số mẫu (Hz) - Vosk models thường mong đợi 16kHz
CHUNK_SIZE = 8192 # Kích thước khối âm thanh để xử lý
RECORD_SECONDS = 5 # Thời gian ghi âm cho mỗi lần nhận dạng

# Đặt đường dẫn đến thư mục mô hình Vosk đã giải nén của bạn
# Ví dụ: nếu bạn giải nén vosk-model-small-en-us-0.15 vào thư mục 'model'
MODEL_PATH = "model/vosk-model-small-en-us-0.15" 

# --- Hàm chính ---

def ghi_am_va_nhan_dang_giong_noi():
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy thư mục mô hình Vosk tại {MODEL_PATH}")
        print("Vui lòng tải và giải nén mô hình Vosk vào đúng đường dẫn.")
        return

    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print(f"Đang ghi âm {RECORD_SECONDS} giây và nhận dạng giọng nói...")
    print("Vui lòng nói rõ ràng.")

    frames = []
    text_ket_qua = ""

    for i in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
        if rec.AcceptWaveform(data):
            ket_qua_json = json.loads(rec.Result())
            if "text" in ket_qua_json:
                text_ket_qua += ket_qua_json["text"] + " "
        else:
            pass # rec.PartialResult() có thể được dùng để hiển thị kết quả tạm thời

    # Lấy kết quả cuối cùng sau khi dừng ghi âm
    ket_qua_json_cuoi = json.loads(rec.FinalResult())
    if "text" in ket_qua_json_cuoi:
        text_ket_qua += ket_qua_json_cuoi["text"]

    print("Hoàn thành ghi âm và nhận dạng.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return text_ket_qua.strip()

# --- Logic chính ---

if __name__ == "__main__":
    import os # Đảm bảo os được import khi chạy trực tiếp

    while True:
        print("\n--- Menu Nhận dạng Giọng nói ---")
        print("1. Bắt đầu nhận dạng giọng nói")
        print("2. Thoát")

        lua_chon = input("Nhập lựa chọn của bạn: ")

        if lua_chon == '1':
            van_ban_nhan_dang = ghi_am_va_nhan_dang_giong_noi()
            if van_ban_nhan_dang:
                print("\n--- Văn bản nhận dạng được ---")
                print(van_ban_nhan_dang)
                print("------------------------------")
            else:
                print("\nKhông nhận dạng được văn bản nào.")
        elif lua_chon == '2':
            print("Đang thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

