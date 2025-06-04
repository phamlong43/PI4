import pyaudio
import wave
import numpy as np
import pickle
import os
import scipy.io.wavfile as wavfile 

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

DINH_DANG = pyaudio.paInt16
KENH = 1
TAN_SO_LAY_MAU = 16000 
KICH_THUOC_KHUNG = 1024
THOI_GIAN_GHI_AM_LAN_DAU = 3 
SO_LAN_GHI_AM_DANG_KY = 2 

THU_MUC_NGUOI_DUNG = "enrolled_speakers_ai" 
TEN_FILE_GHI_AM_TAM = "temp_recording_ai.wav" 

NGUONG_XAC_MINH_CO_DINH = 0.75 

try:
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                                run_opts={"device":"cpu"}) 
except Exception as e:
    print(f"Lỗi khi tải mô hình SpeechBrain: {e}")
    classifier = None 

def ghi_am_audio(ten_file, thoi_luong=THOI_GIAN_GHI_AM_LAN_DAU):
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

def trich_xuat_embedding_ai(audio_path):
    if classifier is None:
        print("Mô hình AI chưa được tải. Không thể trích xuất embedding.")
        return None
    try:
        signal, sr = torchaudio.load(audio_path)
        
        if sr != TAN_SO_LAY_MAU:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TAN_SO_LAY_MAU)
            signal = resampler(signal)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        embedding = classifier.encode_batch(signal.squeeze(0).unsqueeze(0)).cpu().numpy().squeeze()
        return embedding
    except Exception as e:
        print(f"Lỗi khi trích xuất embedding từ {audio_path}: {e}")
        return None

def huan_luyen_mo_hinh_nguoi_noi_ai_da_lan(ten_nguoi_noi, so_lan_ghi_am=SO_LAN_GHI_AM_DANG_KY):
    tat_ca_embeddings = []
    
    thu_muc_ghi_am_dang_ky = os.path.join(THU_MUC_NGUOI_DUNG, "enroll_audios", ten_nguoi_noi)
    os.makedirs(thu_muc_ghi_am_dang_ky, exist_ok=True)

    for i in range(so_lan_ghi_am):
        print(f"\n--- Đăng ký '{ten_nguoi_noi}' - Lần {i+1}/{so_lan_ghi_am} ---")
        temp_audio_path = os.path.join(thu_muc_ghi_am_dang_ky, f"enroll_{ten_nguoi_noi}_lan_{i+1}.wav")
        ghi_am_audio(temp_audio_path, thoi_luong=THOI_GIAN_GHI_AM_LAN_DAU)
        
        embedding_lan_nay = trich_xuat_embedding_ai(temp_audio_path)
        if embedding_lan_nay is None or len(embedding_lan_nay) == 0:
            print(f"Không trích xuất được embedding từ lần ghi âm thứ {i+1}. Vui lòng thử lại.")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False 
        
        tat_ca_embeddings.append(embedding_lan_nay)
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    if not tat_ca_embeddings:
        print(f"Không có đủ embeddings để tạo mẫu giọng nói cho {ten_nguoi_noi}.")
        return False

    speaker_template = np.mean(tat_ca_embeddings, axis=0)
    
    duong_dan_mau = os.path.join(THU_MUC_NGUOI_DUNG, f"{ten_nguoi_noi}.pkl") 
    with open(duong_dan_mau, 'wb') as f:
        pickle.dump(speaker_template, f)
    print(f"Mẫu giọng nói AI cho '{ten_nguoi_noi}' đã được huấn luyện và lưu vào {duong_dan_mau}")
    
    if os.path.exists(thu_muc_ghi_am_dang_ky) and not os.listdir(thu_muc_ghi_am_dang_ky):
        os.rmdir(thu_muc_ghi_am_dang_ky)
    
    return True

def xac_minh_nguoi_noi_ai(ten_nguoi_noi, nguong_xac_minh):
    duong_dan_mau = os.path.join(THU_MUC_NGUOI_DUNG, f"{ten_nguoi_noi}.pkl")
    if not os.path.exists(duong_dan_mau):
        print(f"Lỗi: Không tìm thấy mẫu giọng nói AI cho '{ten_nguoi_noi}'. Vui lòng đăng ký người nói này trước.")
        return False, 0.0

    with open(duong_dan_mau, 'rb') as f:
        mau_da_dang_ky = pickle.load(f)

    duong_dan_audio_xac_minh = TEN_FILE_GHI_AM_TAM
    ghi_am_audio(duong_dan_audio_xac_minh, thoi_luong=THOI_GIAN_GHI_AM_LAN_DAU)
    
    embedding_kiem_tra = trich_xuat_embedding_ai(duong_dan_audio_xac_minh)
    if embedding_kiem_tra is None or len(embedding_kiem_tra) == 0:
        print("Không trích xuất được embedding từ âm thanh kiểm tra. Xác minh thất bại.")
        if os.path.exists(duong_dan_audio_xac_minh):
            os.remove(duong_dan_audio_xac_minh)
        return False, 0.0

    if os.path.exists(duong_dan_audio_xac_minh):
        os.remove(duong_dan_audio_xac_minh)

    do_tuong_dong = np.dot(mau_da_dang_ky, embedding_kiem_tra) / (np.linalg.norm(mau_da_dang_ky) * np.linalg.norm(embedding_kiem_tra))
    
    if do_tuong_dong >= 1.0: 
        ty_le_phu_hop = 100.0
    elif do_tuong_dong >= nguong_xac_minh: 
        ty_le_phu_hop = 75 + (do_tuong_dong - nguong_xac_minh) / (1.0 - nguong_xac_minh) * 25
    else: 
        ty_le_phu_hop = (do_tuong_dong - (-1.0)) / (nguong_xac_minh - (-1.0)) * 75
        ty_le_phu_hop = max(0.0, min(75.0, ty_le_phu_hop)) 

    print(f"Độ tương đồng (Cosine Similarity): {do_tuong_dong:.4f} (Ngưỡng: {nguong_xac_minh:.4f})")
    
    if do_tuong_dong > nguong_xac_minh:
        print(f"Kết quả: **XÁC MINH THÀNH CÔNG!** Tỷ lệ phù hợp: {ty_le_phu_hop:.2f}%")
        return True, ty_le_phu_hop
    else:
        print(f"Kết quả: **XÁC MINH THẤT BẠI.** Tỷ lệ phù hợp: {ty_le_phu_hop:.2f}%")
        return False, ty_le_phu_hop

if __name__ == "__main__":
    if classifier is None:
        print("Không thể chạy ứng dụng do mô hình AI không được tải.")
        exit()

    os.makedirs(THU_MUC_NGUOI_DUNG, exist_ok=True)
    os.makedirs(os.path.join(THU_MUC_NGUOI_DUNG, "enroll_audios"), exist_ok=True) 

    while True:
        print("\n--- Menu Xác thực Giọng nói (AI) ---")
        print("1. Đăng ký Người nói Mới (Dùng AI - 2 lần nói)")
        print("2. Xác minh Người nói Hiện có (Dùng AI)")
        print("3. Thoát")

        lua_chon = input("Nhập lựa chọn của bạn: ")

        if lua_chon == '1':
            ten_nguoi_noi = input("Nhập tên người nói để đăng ký: ").strip()
            if not ten_nguoi_noi:
                print("Tên người nói không được để trống.")
                continue
            
            if huan_luyen_mo_hinh_nguoi_noi_ai_da_lan(ten_nguoi_noi):
                print(f"Đã đăng ký thành công '{ten_nguoi_noi}'.")
            else:
                print(f"Không thể đăng ký '{ten_nguoi_noi}'.")

        elif lua_chon == '2':
            ten_nguoi_noi_can_xac_minh = input("Nhập tên người nói cần xác minh: ").strip()
            if not ten_nguoi_noi_can_xac_minh:
                print("Tên người nói không được để trống.")
                continue

            thanh_cong, ty_le = xac_minh_nguoi_noi_ai(ten_nguoi_noi_can_xac_minh, NGUONG_XAC_MINH_CO_DINH)
            
        elif lua_chon == '3':
            print("Đang thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
