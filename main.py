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
THOI_GIAN_GHI_AM = 5 
THU_MUC_NGUOI_DUNG = "enrolled_speakers"
TEN_FILE_GHI_AM_TAM = "temp_recording.wav" # Được sử dụng cho việc ghi âm tạm thời khi xác minh thủ công

# --- Hàm hỗ trợ (giữ nguyên từ mã trước) ---

def ghi_am_audio(ten_file, thoi_luong=THOI_GIAN_GHI_AM):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=DINH_DANG, channels=KENH, rate=TAN_SO_LAY_MAU, input=True, frames_per_buffer=KICH_THUOC_KHUNG)
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
        (tan_so, tin_hieu) = wavfile.read(duong_dan_audio)
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

def lay_mo_hinh_gmm(ten_nguoi_noi):
    """Tải mô hình GMM đã đăng ký của một người nói."""
    model_path = os.path.join(THU_MUC_NGUOI_DUNG, f"{ten_nguoi_noi}.gmm")
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy mô hình cho '{ten_nguoi_noi}'.")
        return None
    with open(model_path, 'rb') as f:
        gmm = pickle.load(f)
    return gmm

# --- Hàm đánh giá hiệu suất mới ---

def danh_gia_hieu_suat(nguong_xac_minh, test_data_dir="test_data"):
    print(f"\n--- Đánh giá hiệu suất với ngưỡng: {nguong_xac_minh} ---")
    
    true_accepts = 0  # True Positives: Người hợp lệ được chấp nhận
    false_rejects = 0 # False Negatives: Người hợp lệ bị từ chối
    
    false_accepts = 0 # False Positives: Kẻ mạo danh được chấp nhận
    true_rejects = 0  # True Negatives: Kẻ mạo danh bị từ chối

    # Lấy danh sách các người nói đã đăng ký
    enrolled_speakers = [f.split('.')[0] for f in os.listdir(THU_MUC_NGUOI_DUNG) if f.endswith('.gmm')]
    
    if not enrolled_speakers:
        print("Chưa có người nói nào được đăng ký. Vui lòng đăng ký trước.")
        return

    # 1. Đánh giá Tỷ lệ Từ chối Sai (FRR - False Rejection Rate)
    # Kiểm tra các mẫu giọng nói hợp lệ (genuine)
    print("\nĐánh giá các mẫu hợp lệ (Genuine/Client):")
    for speaker in enrolled_speakers:
        gmm_model = lay_mo_hinh_gmm(speaker)
        if gmm_model is None:
            continue

        genuine_samples_dir = os.path.join(test_data_dir, 'genuine', speaker)
        if not os.path.exists(genuine_samples_dir):
            print(f"Không tìm thấy thư mục mẫu hợp lệ cho '{speaker}': {genuine_samples_dir}")
            continue

        for audio_file in os.listdir(genuine_samples_dir):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(genuine_samples_dir, audio_file)
                features = trich_xuat_dac_trung_mfcc(audio_path)
                if features is None or len(features) == 0:
                    continue
                
                score = gmm_model.score(features)
                print(f"  '{speaker}' với file '{audio_file}': Điểm {score:.2f}")

                if score > nguong_xac_minh:
                    true_accepts += 1
                else:
                    false_rejects += 1
    
    total_genuine_attempts = true_accepts + false_rejects
    frr = (false_rejects / total_genuine_attempts) * 100 if total_genuine_attempts > 0 else 0
    print(f"Số lần chấp nhận đúng: {true_accepts}")
    print(f"Số lần từ chối sai (FR): {false_rejects}")
    print(f"Tỷ lệ Từ chối Sai (FRR): {frr:.2f}%")

    # 2. Đánh giá Tỷ lệ Chấp nhận Sai (FAR - False Acceptance Rate)
    # Kiểm tra các mẫu giọng nói không hợp lệ (imposter)
    print("\nĐánh giá các mẫu không hợp lệ (Imposter):")
    imposter_samples_dir = os.path.join(test_data_dir, 'imposter')
    if not os.path.exists(imposter_samples_dir):
        print(f"Không tìm thấy thư mục mẫu không hợp lệ: {imposter_samples_dir}")
    else:
        for audio_file in os.listdir(imposter_samples_dir):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(imposter_samples_dir, audio_file)
                # Cần biết mẫu imposter này cố gắng giả mạo ai
                # Quy ước đặt tên file ví dụ: imposter_C_as_A.wav (người C giả mạo A)
                try:
                    parts = audio_file.replace('.wav', '').split('_as_')
                    if len(parts) == 2:
                        imposter_speaker = parts[0] # Người thực sự nói
                        target_speaker = parts[1]    # Người bị giả mạo
                    elif len(parts) == 3: # Cho trường hợp "imposter_C_as_A"
                        imposter_speaker = parts[1] 
                        target_speaker = parts[2]
                    else:
                        print(f"Tên file imposter không đúng quy ước: {audio_file}. Bỏ qua.")
                        continue

                    gmm_model = lay_mo_hinh_gmm(target_speaker)
                    if gmm_model is None:
                        continue # Bỏ qua nếu người bị giả mạo chưa đăng ký

                    features = trich_xuat_dac_trung_mfcc(audio_path)
                    if features is None or len(features) == 0:
                        continue

                    score = gmm_model.score(features)
                    print(f"  '{imposter_speaker}' giả mạo '{target_speaker}' với file '{audio_file}': Điểm {score:.2f}")

                    if score > nguong_xac_minh:
                        false_accepts += 1 # Kẻ mạo danh được chấp nhận (lỗi)
                    else:
                        true_rejects += 1  # Kẻ mạo danh bị từ chối (đúng)
                except Exception as e:
                    print(f"Lỗi xử lý file imposter {audio_file}: {e}")
                    continue

    total_imposter_attempts = false_accepts + true_rejects
    far = (false_accepts / total_imposter_attempts) * 100 if total_imposter_attempts > 0 else 0
    print(f"Số lần chấp nhận sai (FA): {false_accepts}")
    print(f"Số lần từ chối đúng: {true_rejects}")
    print(f"Tỷ lệ Chấp nhận Sai (FAR): {far:.2f}%")

    # 3. Tính toán Độ chính xác tổng thể (Accuracy)
    total_correct = true_accepts + true_rejects
    total_attempts = total_genuine_attempts + total_imposter_attempts
    
    accuracy = (total_correct / total_attempts) * 100 if total_attempts > 0 else 0
    print(f"\nTổng số lần thử: {total_attempts}")
    print(f"Tổng số lần đúng: {total_correct}")
    print(f"Độ chính xác tổng thể (Accuracy): {accuracy:.2f}%")

    return {
        "FRR": frr,
        "FAR": far,
        "Accuracy": accuracy,
        "True_Accepts": true_accepts,
        "False_Rejects": false_rejects,
        "False_Accepts": false_accepts,
        "True_Rejects": true_rejects
    }

# --- Logic chính (đã cập nhật để bao gồm menu đánh giá) ---

if __name__ == "__main__":
    os.makedirs(THU_MUC_NGUOI_DUNG, exist_ok=True)
    os.makedirs(os.path.join("test_data", "genuine"), exist_ok=True)
    os.makedirs(os.path.join("test_data", "imposter"), exist_ok=True)

    while True:
        print("\n--- Menu Xác thực Giọng nói ---")
        print("1. Đăng ký Người nói Mới")
        print("2. Xác minh Người nói Hiện có (Thủ công)")
        print("3. Đánh giá Hiệu suất Hệ thống")
        print("4. Thoát")

        lua_chon = input("Nhập lựa chọn của bạn: ")

        if lua_chon == '1':
            ten_nguoi_noi = input("Nhập tên người nói để đăng ký: ").strip()
            if not ten_nguoi_noi:
                print("Tên người nói không được để trống.")
                continue
            
            # Tạo thư mục con cho mẫu đăng ký nếu bạn muốn giữ lại
            os.makedirs(os.path.join(THU_MUC_NGUOI_DUNG, "enroll_audios"), exist_ok=True)
            duong_dan_audio = os.path.join(THU_MUC_NGUOI_DUNG, "enroll_audios", f"enroll_{ten_nguoi_noi}.wav")
            
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
            
            # Lấy ngưỡng xác minh từ hàm đánh giá (hoặc sử dụng một ngưỡng cố định)
            # Ở đây ta dùng lại ngưỡng mặc định -100, bạn có thể thay đổi
            NGUONG_XAC_MINH_THU_CONG = -100 
            
            # Hàm verify_speaker chưa được cập nhật để sử dụng NGUONG_XAC_MINH_THU_CONG
            # Cần tải lại mô hình và tính score trực tiếp để sử dụng ngưỡng động
            gmm_model = lay_mo_hinh_gmm(ten_nguoi_noi_can_xac_minh)
            if gmm_model is None:
                print("Không thể xác minh.")
            else:
                dac_trung_kiem_tra = trich_xuat_dac_trung_mfcc(duong_dan_audio_xac_minh)
                if dac_trung_kiem_tra is None or len(dac_trung_kiem_tra) == 0:
                    print("Không trích xuất được đặc trưng từ âm thanh kiểm tra. Xác minh thất bại.")
                else:
                    diem_so = gmm_model.score(dac_trung_kiem_tra)
                    print(f"Điểm xác minh cho '{ten_nguoi_noi_can_xac_minh}': {diem_so:.2f}")
                    if diem_so > NGUONG_XAC_MINH_THU_CONG:
                        print(f"--> Người nói '{ten_nguoi_noi_can_xac_minh}' ĐÃ ĐƯỢC XÁC MINH! (Điểm: {diem_so:.2f} > Ngưỡng: {NGUONG_XAC_MINH_THU_CONG})")
                    else:
                        print(f"--> Người nói '{ten_nguoi_noi_can_xac_minh}' CHƯA ĐƯỢC XÁC MINH. (Điểm: {diem_so:.2f} <= Ngưỡng: {NGUONG_XAC_MINH_THU_CONG})")
            
            if os.path.exists(duong_dan_audio_xac_minh):
                os.remove(duong_dan_audio_xac_minh)

        elif lua_chon == '3':
            # Người dùng có thể nhập ngưỡng để thử nghiệm
            try:
                input_nguong = input("Nhập ngưỡng xác minh để đánh giá (mặc định: -100): ")
                nguong_de_danh_gia = float(input_nguong) if input_nguong else -100
                danh_gia_hieu_suat(nguong_de_danh_gia)
            except ValueError:
                print("Ngưỡng không hợp lệ. Vui lòng nhập một số.")

        elif lua_chon == '4':
            print("Đang thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
