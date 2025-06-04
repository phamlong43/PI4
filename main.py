import pyaudio
import wave
import numpy as np
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
import pickle
import os
import scipy.io.wavfile as wavfile 

# --- Cấu hình ---
DINH_DANG = pyaudio.paInt16
KENH = 1
TAN_SO_LAY_MAU = 16000 
KICH_THUOC_KHUNG = 1024
THOI_GIAN_GHI_AM_LAN_DAU = 5 
SO_LAN_GHI_AM_DANG_KY = 3 

THU_MUC_NGUOI_DUNG = "enrolled_speakers"
TEN_FILE_GHI_AM_TAM = "temp_recording.wav" 

# Ngưỡng xác minh. Cần điều chỉnh thực nghiệm.
NGUONG_XAC_MINH_CO_DINH = -100 

# --- Hàm hỗ trợ ---

def ghi_am_audio(ten_file, thoi_luong=THOI_GIAN_GHI_AM_LAN_DAU):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=DINH_DANG,
                        channels=KENH,
                        rate=TAN_SO_LAY_MAU,
                        input=True,
                        frames_per_buffer=KICH_THUOC_KHUNG)

    print(f"Đang ghi âm {thoi_luong} giây âm thanh cho '{ten_file}'...")
    frames = []

    for i in range(0, int(TAN_SO_LAY_MA0U / KICH_THUOC_KHUNG * thoi_luong)):
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
        (tan_so, tin_hieu) = wavfile.read(duong_dan_audio)
        if tin_hieu.dtype.kind == 'i': 
            tin_hieu = tin_hieu.astype(np.float32) / (2**15) 
        dac_trung_mfcc = mfcc(tin_hieu, tan_so, numcep=13, nfilt=26, nfft=512)
        return dac_trung_mfcc
    except Exception as e:
        print(f"Lỗi khi trích xuất MFCC từ {duong_dan_audio}: {e}")
        return None

def huan_luyen_mo_hinh_nguoi_noi_da_lan(ten_nguoi_noi, so_lan_ghi_am=SO_LAN_GHI_AM_DANG_KY):
    tat_ca_dac_trung = []
    
    thu_muc_ghi_am_dang_ky = os.path.join(THU_MUC_NGUOI_DUNG, "enroll_audios", ten_nguoi_noi)
    os.makedirs(thu_muc_ghi_am_dang_ky, exist_ok=True)

    for i in range(so_lan_ghi_am):
        print(f"\n--- Đăng ký '{ten_nguoi_noi}' - Lần {i+1}/{so_lan_ghi_am} ---")
        temp_audio_path = os.path.join(thu_muc_ghi_am_dang_ky, f"enroll_{ten_nguoi_noi}_lan_{i+1}.wav")
        ghi_am_audio(temp_audio_path, thoi_luong=THOI_GIAN_GHI_AM_LAN_DAU)
        
        dac_trung_lan_nay = trich_xuat_dac_trung_mfcc(temp_audio_path)
        if dac_trung_lan_nay is None or len(dac_trung_lan_nay) == 0:
            print(f"Không trích xuất được đặc trưng từ lần ghi âm thứ {i+1}. Vui lòng thử lại.")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False 
        
        if len(tat_ca_dac_trung) == 0:
            tat_ca_dac_trung = dac_trung_lan_nay
        else:
            tat_ca_dac_trung = np.vstack((tat_ca_dac_trung, dac_trung_lan_nay))
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    if len(tat_ca_dac_trung) == 0:
        print(f"Không có đủ đặc trưng để huấn luyện mô hình cho {ten_nguoi_noi}.")
        return False

    gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=200, random_state=0)
    
    try:
        gmm.fit(tat_ca_dac_trung)
    except ValueError as e:
        print(f"Lỗi khi khớp GMM cho {ten_nguoi_noi}: {e}. Đảm bảo đủ dữ liệu cho n_components.")
        return False

    duong_dan_mo_hinh = os.path.join(THU_MUC_NGUOI_DUNG, f"{ten_nguoi_noi}.gmm")
    with open(duong_dan_mo_hinh, 'wb') as f:
        pickle.dump(gmm, f)
    print(f"Mô hình cho '{ten_nguoi_noi}' đã được huấn luyện và lưu vào {duong_dan_mo_hinh}")
    
    if os.path.exists(thu_muc_ghi_am_dang_ky) and not os.listdir(thu_muc_ghi_am_dang_ky):
        os.rmdir(thu_muc_ghi_am_dang_ky)
    
    return True

def xac_minh_nguoi_noi(ten_nguoi_noi, nguong_xac_minh):
    duong_dan_mo_hinh = os.path.join(THU_MUC_NGUOI_DUNG, f"{ten_nguoi_noi}.gmm")
    if not os.path.exists(duong_dan_mo_hinh):
        print(f"Lỗi: Không tìm thấy mô hình cho '{ten_nguoi_noi}'. Vui lòng đăng ký người nói này trước.")
        return False, 0.0

    with open(duong_dan_mo_hinh, 'rb') as f:
        gmm_da_dang_ky = pickle.load(f)

    duong_dan_audio_xac_minh = TEN_FILE_GHI_AM_TAM
    ghi_am_audio(duong_dan_audio_xac_minh, thoi_luong=THOI_GIAN_GHI_AM_LAN_DAU)
    
    dac_trung_kiem_tra = trich_xuat_dac_trung_mfcc(duong_dan_audio_xac_minh)
    if dac_trung_kiem_tra is None or len(dac_trung_kiem_tra) == 0:
        print("Không trích xuất được đặc trưng từ âm thanh kiểm tra. Xác minh thất bại.")
        if os.path.exists(duong_dan_audio_xac_minh):
            os.remove(duong_dan_audio_xac_minh)
        return False, 0.0

    diem_so = gmm_da_dang_ky.score(dac_trung_kiem_tra)
    
    if os.path.exists(duong_dan_audio_xac_minh):
        os.remove(duong_dan_audio_xac_minh)

    # Tính toán tỷ lệ thành công (accuracy %)
    # Công thức này mang tính ước lượng và trực quan hóa.
    # Càng gần ngưỡng từ phía trên, % càng cao.
    # Ngưỡng càng âm sâu, điểm số âm càng gần 0 thì % càng cao.
    if diem_so >= 0: # Nếu điểm số không âm, coi như khớp rất tốt
        ty_le_thanh_cong = 100.0
    else:
        # Chuẩn hóa điểm số so với ngưỡng
        # Ví dụ: Ngưỡng = -100. Điểm = -50 -> tỉ lệ = 50%. Điểm = -10 -> tỉ lệ = 90%.
        # Điểm = -120 -> tỉ lệ = -20% (sẽ bị cắt về 0).
        ty_le_thanh_cong = (diem_so - nguong_xac_minh) / abs(nguong_xac_minh) * 100 
        ty_le_thanh_cong = max(0.0, min(100.0, ty_le_thanh_cong)) # Đảm bảo nằm trong khoảng [0, 100]

    print(f"Điểm xác minh thô: {diem_so:.2f} (Ngưỡng: {nguong_xac_minh:.2f})")
    
    if diem_so > nguong_xac_minh:
        print(f"Kết quả: **XÁC MINH THÀNH CÔNG!** Tỷ lệ phù hợp: {ty_le_thanh_cong:.2f}%")
        return True, ty_le_thanh_cong
    else:
        print(f"Kết quả: **XÁC MINH THẤT BẠI.** Tỷ lệ phù hợp: {ty_le_thanh_cong:.2f}%")
        return False, ty_le_thanh_cong

# --- Logic chính ---

if __name__ == "__main__":
    os.makedirs(THU_MUC_NGUOI_DUNG, exist_ok=True)
    os.makedirs(os.path.join(THU_MUC_NGUOI_DUNG, "enroll_audios"), exist_ok=True) 

    while True:
        print("\n--- Menu Xác thực Giọng nói ---")
        print("1. Đăng ký Người nói Mới (3 lần nói)")
        print("2. Xác minh Người nói Hiện có")
        print("3. Thoát")

        lua_chon = input("Nhập lựa chọn của bạn: ")

        if lua_chon == '1':
            ten_nguoi_noi = input("Nhập tên người nói để đăng ký: ").strip()
            if not ten_nguoi_noi:
                print("Tên người nói không được để trống.")
                continue
            
            if huan_luyen_mo_hinh_nguoi_noi_da_lan(ten_nguoi_noi):
                print(f"Đã đăng ký thành công '{ten_nguoi_noi}'.")
            else:
                print(f"Không thể đăng ký '{ten_nguoi_noi}'.")

        elif lua_chon == '2':
            ten_nguoi_noi_can_xac_minh = input("Nhập tên người nói cần xác minh: ").strip()
            if not ten_nguoi_noi_can_xac_minh:
                print("Tên người nói không được để trống.")
                continue

            # Gọi hàm xác minh và nó sẽ tự in kết quả
            thanh_cong, ty_le = xac_minh_nguoi_noi(ten_nguoi_noi_can_xac_minh, NGUONG_XAC_MINH_CO_DINH)
            
        elif lua_chon == '3':
            print("Đang thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
