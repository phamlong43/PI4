import os
import torch
import torchaudio
import yaml
from model import RawNet  # Đảm bảo file model.py đã sẵn sàng

# === Load cấu hình và mô hình ===
with open("model_config_RawNet2.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RawNet(config['model'], device).to(device)
model.load_state_dict(torch.load("rawnet2_patience_7.pth", map_location=device))
model.eval()

# === Hàm tiền xử lý ===
def preprocess_audio(file_path, target_sr=16000, max_len=64600):
    audio, sr = torchaudio.load(file_path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    audio = audio.mean(dim=0)
    audio = audio / torch.max(torch.abs(audio))
    if audio.shape[0] < max_len:
        audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[0]))
    else:
        audio = audio[:max_len]
    return audio.unsqueeze(0)

# === Hàm dự đoán ===
def predict(file_path):
    audio = preprocess_audio(file_path).to(device)
    with torch.no_grad():
        output = model(audio)
        prob = torch.softmax(output, dim=1)[0]
        label_id = torch.argmax(prob).item()
        label = "Bonafide" if label_id == 0 else "Spoof"
        confidence = prob[label_id].item()
    return label, confidence

# === Duyệt toàn bộ file trong thư mục ===
folder_path = "50_data_voice"
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

print(f"\n📁 Phát hiện giả mạo giọng nói trong thư mục: {folder_path}\n")
for wav_file in wav_files:
    file_path = os.path.join(folder_path, wav_file)
    label, conf = predict(file_path)
    print(f"{wav_file:30} ➜ {label} ({conf*100:.2f}%)")
