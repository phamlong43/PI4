import pyaudio
import wave
import numpy as np
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
import pickle
import os

# --- Cấu hình ---
DINH_DANG = pyaudio.paInt16
KENH = 1
TAN_SO_LAY_MAU = 16000 # Hz
KICH_THUOC_KHUNG = 1024
THOI_GIAN_GHI_AM = 5 # Giây cho việc đăng ký/xác minh
THU_MUC_NGUOI_DUNG = "enrolled_speakers"
TEN_FILE_GHI_AM_TAM = "temp_recording.wav"

# --- Hàm hỗ trợ ---

def ghi_am_audio(ten_file, thoi_luong=THOI_GIAN_GHI_AM):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=DINH_DANG,
                        channels=KENH,
                        rate=TAN_SO_LAY_MAU,
                        input=True,
                        frames_per_buffer=KICH_THUOC_KHUNG)

    print(f"Đang ghi âm {thoi_luong} giây âm thanh cho '{ten_file}'...")
    frames = []

    for i in range(0, int(TAN_SO_LAY_MAU / KICH_THUOC_KHUNG * thoi_luong)):
        data = stream.read(KICH_THUOC_KHUNG)
        frames.append(data)

    print("Hoàn thành ghi âm.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(ten_file, 'wb') as wf:
        wf.setnchannels(KENH)
        wf.setsampwidth(audio.get_sample_size(DINH_DANG))
        wf.setframerate(TAN_SO_LAY_MAU)
        wf.writeframes(b''.join(frames))

def trich_xuat_dac_trung_mfcc(duong_dan_audio):
    try:
        (tan_so, tin_hieu) = wave.read(duong_dan_audio)
        if tin_hieu.dtype.kind == 'i':
            tin_hieu = tin_hieu.astype(np.float32) / (2**15)
        
        dac_trung_mfcc = mfcc(tin_hieu, tan_so, numcep=13, nfilt=26, nfft=512)
        return dac_trung_mfcc
    except Exception as e:
        print(f"Lỗi khi trích xuất MFCC từ {duong_dan_audio}: {e}")
        return None

def huan_luyen_mo_hinh_nguoi_noi(ten_nguoi_noi, duong_dan_file_audio):
    dac_trung = trich_xuat_dac_trung_mfcc(duong_dan_file_audio)
    if dac_trung is None or len(dac_trung) == 0:
        print(f"Không trích xuất được đặc trưng cho {ten_nguoi_noi}. Huấn luyện mô hình thất bại.")
        return False

    gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200, random_state=0)
    
    try:
        gmm.fit(dac_trung)
    except ValueError as e:
        print(f"Lỗi khi khớp GMM cho {ten_nguoi_noi}: {e}. Đảm bảo đủ dữ liệu cho n_components.")
        return False

    duong_dan_mo_hinh = os.path.join(THU_MUC_NGUOI_DUNG, f"{ten_nguoi_noi}.gmm")
    with open(duong_dan_mo_hinh, 'wb') as f:
        pickle.dump(gmm, f)
    print(f"Mô hình cho '{ten_nguoi_noi}' đã được huấn luyện và lưu vào {duong_dan_mo_hinh}")
    return True

def xac_minh_nguoi_noi(ten_nguoi_noi, duong_dan_file_audio):
    duong_dan_mo_hinh = os.path.join(THU_MUC_NGUOI_DUNG, f"{ten_nguoi_noi}.gmm")
    if not os.path.exists(duong_dan_mo_hinh):
        print(f"Lỗi: Không tìm thấy mô hình cho '{ten_nguoi_noi}'. Vui lòng đăng ký người nói này trước.")
        return False

    with open(duong_dan_mo_hinh, 'rb') as f:
        gmm_da_dang_ky = pickle.load(f)

    dac_trung_kiem_tra = trich_xuat_dac_trung_mfcc(duong_dan_file_audio)
    if dac_trung_kiem_tra is None or len(dac_trung_kiem_tra) == 0:
        print("Không trích xuất được đặc trưng từ âm thanh kiểm tra. Xác minh thất bại.")
        return False

    diem_so = gmm_da_dang_ky.score(dac_trung_kiem_tra)
    
    # Ngưỡng xác minh. Cần điều chỉnh thực nghiệm.
    NGUONG_XAC_MINH = -100 

    print(f"Điểm xác minh cho '{ten_nguoi_noi}': {diem_so:.2f}")
    if diem_so > NGUONG_XAC_MINH:
        print(f"--> Người nói '{ten_nguoi_noi}' ĐÃ ĐƯỢC XÁC MINH! (Điểm: {diem_so:.2f} > Ngưỡng: {NGUONG_XAC_MINH})")
        return True
    else:
        print(f"--> Người nói '{ten_nguoi_noi}' CHƯA ĐƯỢC XÁC MINH. (Điểm: {diem_so:.2f} <= Ngưỡng: {NGUONG_XAC_MINH})")
        return False

# --- Logic chính ---

if __name__ == "__main__":
    os.makedirs(THU_MUC_NGUOI_DUNG, exist_ok=True)

    while True:
        print("\n--- Menu Xác thực Giọng nói ---")
        print("1. Đăng ký Người nói Mới")
        print("2. Xác minh Người nói Hiện có")
        print("3. Thoát")

        lua_chon = input("Nhập lựa chọn của bạn: ")

        if lua_chon == '1':
            ten_nguoi_noi = input("Nhập tên người nói để đăng ký: ").strip()
            if not ten_nguoi_noi:
                print("Tên người nói không được để trống.")
                continue
            
            duong_dan_audio = os.path.join(THU_MUC_NGUOI_DUNG, f"enroll_{ten_nguoi_noi}.wav")
            ghi_am_audio(duong_dan_audio, thoi_luong=THOI_GIAN_GHI_AM)
            if huan_luyen_mo_hinh_nguoi_noi(ten_nguoi_noi, duong_dan_audio):
                print(f"Đã đăng ký thành công '{ten_nguoi_noi}'.")
            else:
                print(f"Không thể đăng ký '{ten_nguoi_noi}'.")

        elif lua_chon == '2':
            ten_nguoi_noi_can_xac_minh = input("Nhập tên người nói cần xác minh: ").strip()
            if not ten_nguoi_noi_can_xac_minh:
                print("Tên người nói không được để trống.")
                continue

            duong_dan_audio_xac_minh = TEN_FILE_GHI_AM_TAM
            ghi_am_audio(duong_dan_audio_xac_minh, thoi_luong=THOI_GIAN_GHI_AM)
            
            if xac_minh_nguoi_noi(ten_nguoi_noi_can_xac_minh, duong_dan_audio_xac_minh):
                print("Xác minh thành công!")
            else:
                print("Xác minh thất bại.")
            
            if os.path.exists(duong_dan_audio_xac_minh):
                os.remove(duong_dan_audio_xac_minh)

        elif lua_chon == '3':
            print("Đang thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
