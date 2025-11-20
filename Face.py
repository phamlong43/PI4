# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os
import joblib
import subprocess
import sys

# -----------------------
# Dictionary chứa thông tin thuật toán
# -----------------------
algo_info = {
    "ascon128": {"key_size": 128, "binary": "./bench_ascon"},
    "aes128": {"key_size": 128, "binary": "./bench_aes"},
    "aes256": {"key_size": 256, "binary": "./bench_aes"},
    "chacha20": {"key_size": 128, "binary": "./bench_chacha20"},
    "grain128": {"key_size": 128, "binary": "./bench_grain"},
    "led64": {"key_size": 64, "binary": "./bench_led"},
    "led128": {"key_size": 128, "binary": "./bench_led"},
    "present80": {"key_size": 80, "binary": "./bench_present"},
    "present128": {"key_size": 128, "binary": "./bench_present"},
    "simon32_64": {"key_size": 64, "binary": "./bench_simon"},
    "simon64_128": {"key_size": 128, "binary": "./bench_simon"},
    "speck32_64": {"key_size": 64, "binary": "./bench_speck"},
    "speck64_128": {"key_size": 128, "binary": "./bench_speck"},
    "twine80": {"key_size": 80, "binary": "./bench_TWINE"},
    "twine128": {"key_size": 128, "binary": "./bench_TWINE"}
}

# -----------------------
# PRESENT encrypt Python (byte-wise sbox)
# -----------------------
def present_encrypt_python(data: str) -> str:
    sbox = [0xC,5,6,0xB,9,0,0xA,0xD,3,0xE,0xF,8,4,7,1,2]
    out_bytes = []
    for b in data.encode('utf-8'):
        out_bytes.append((sbox[b >> 4] << 4) | sbox[b & 0x0F])
    return ''.join(f'{b:02X}' for b in out_bytes)

# -----------------------
# Hàm encrypt_data
# -----------------------
def encrypt_data(data: str, algo: str, binary: str) -> str:
    try:
        if 'present' in algo.lower():
            # Dùng Python implementation cho PRESENT
            ciphertext = present_encrypt_python(data)
        else:
            # Ghi tạm vào file
            temp_input = 'temp_input.txt'
            with open(temp_input, 'w', encoding='utf-8') as f:
                f.write(data)

            cmd = [binary, temp_input]
            if algo in ['aes128', 'aes256']:
                cmd.extend(['--keysize', str(algo_info[algo]['key_size'])])

            result = subprocess.run(cmd, text=True, capture_output=True, check=True)
            os.remove(temp_input)

            # Tìm Ciphertext từ stdout
            ciphertext = None
            for line in result.stdout.splitlines():
                if line.startswith('Ciphertext: '):
                    ciphertext = line.split('Ciphertext: ')[1].strip()
                    break
            if not ciphertext:
                print(f"Lỗi: Không tìm thấy Ciphertext trong output của {algo}")
                ciphertext = None

        return ciphertext
    except Exception as e:
        print(f"Lỗi khi mã hóa {algo}: {e}")
        return None

# -----------------------
# Hàm đọc data.txt
# -----------------------
def read_data_file(file_path='data.txt'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        size_bytes = len(content.encode('utf-8'))
        return content, size_bytes
    except:
        return "phamjLong", len("phamjLong".encode('utf-8'))

# -----------------------
# Lấy metrics phần cứng Pi
# -----------------------
def get_hardware_metrics():
    try:
        cpu_avg = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        temp_file = '/sys/class/thermal/thermal_zone0/temp'
        temp = float(open(temp_file).read()) / 1000 if os.path.exists(temp_file) else 40.0
        return {
            'cpu_avg_%': cpu_avg,
            'ram_%': ram_usage,
            'temp_start_C': temp,
            'stress_level': 1.0,
            'raw_energy_J': 0.0,
            'mem_used_MB': ram.used / (1024*1024),
            'mem_cached_MB': ram.cached / (1024*1024),
            'mem_buffers_MB': ram.buffers / (1024*1024),
            'disk_read_MB': 0.0,
            'disk_write_MB': 0.0,
            'disk_read_count': 0.0,
            'disk_write_count': 0.0,
            'net_sent_MB': 0.0,
            'net_recv_MB': 0.0
        }
    except:
        return {
            'cpu_avg_%': 20.0, 'ram_%': 20.0, 'temp_start_C': 40.0, 'stress_level': 1.0,
            'raw_energy_J': 0.0, 'mem_used_MB': 100.0, 'mem_cached_MB': 50.0, 'mem_buffers_MB': 10.0,
            'disk_read_MB': 0.0, 'disk_write_MB': 0.0, 'disk_read_count': 0.0, 'disk_write_count': 0.0,
            'net_sent_MB': 0.0, 'net_recv_MB': 0.0
        }

# -----------------------
# Hàm tính performance score
# -----------------------
def calculate_performance_score(exec_time, energy, temp_end, cpu_avg, ram_usage):
    time_score = 100*(1 - np.clip(exec_time/0.1,0,1))
    energy_score = 100*(1 - np.clip(energy/0.01,0,1))
    temp_score = 100*(1 - abs(temp_end-40)/20)
    cpu_score = 100*(1 - abs(cpu_avg-20)/40)
    ram_score = 100*(1 - abs(ram_usage-20)/40)
    score = 0.3*time_score + 0.3*energy_score + 0.2*temp_score + 0.1*cpu_score + 0.1*ram_score
    return np.round(np.clip(score,0,100),2)

# -----------------------
# Hàm dự đoán thuật toán tối ưu
# -----------------------
def predict_best_algorithm(model_time, model_energy, scaler_X, scaler_time, scaler_energy):
    algorithms = list(algo_info.keys())
    algo_columns = [f'algo_{algo}' for algo in algorithms]
    feature_cols = [
        'size_bytes', 'stress_level', 'cpu_avg_%', 'ram_%', 'temp_start_C',
        'raw_energy_J', 'mem_used_MB', 'mem_cached_MB', 'mem_buffers_MB',
        'disk_read_MB', 'disk_write_MB', 'disk_read_count', 'disk_write_count',
        'net_sent_MB', 'net_recv_MB'
    ] + algo_columns

    data_content, size_bytes = read_data_file()
    hw_metrics = get_hardware_metrics()
    input_data = {'size_bytes': size_bytes, **hw_metrics}
    for algo in algo_columns:
        input_data[algo] = 0.0
    df_input = pd.DataFrame([input_data])

    input_features = df_input[feature_cols].astype(float).values
    input_features_norm = scaler_X.transform(input_features)
    sample_features = torch.tensor(input_features_norm, dtype=torch.float32).to('cpu')[0]

    algo_scores = []
    for algo in algorithms:
        features = sample_features.clone()
        for i, col in enumerate(feature_cols):
            if col in algo_columns:
                features[i] = 1.0 if col==f'algo_{algo}' else 0.0
        time_preds,_ = model_time(features.unsqueeze(0))
        energy_preds,_ = model_energy(features.unsqueeze(0))
        avg_time = np.maximum(0, scaler_time.inverse_transform(time_preds.cpu().numpy()).mean())
        avg_energy = np.maximum(0, scaler_energy.inverse_transform(energy_preds.cpu().numpy()).mean())
        avg_temp = hw_metrics['temp_start_C']
        avg_cpu = hw_metrics['cpu_avg_%']
        avg_ram = hw_metrics['ram_%']
        score = calculate_performance_score(avg_time, avg_energy, avg_temp, avg_cpu, avg_ram)
        ciphertext = encrypt_data(data_content, algo, algo_info[algo]['binary'])
        algo_scores.append({
            'algorithm': algo,
            'avg_exec_time_s': avg_time,
            'avg_energy_J': avg_energy,
            'avg_temp_C': avg_temp,
            'avg_cpu_%': avg_cpu,
            'avg_ram_%': avg_ram,
            'performance_score': score,
            'ciphertext': ciphertext
        })

    df_ranked = pd.DataFrame(algo_scores)
    df_ranked = df_ranked.sort_values(by='performance_score', ascending=False)
    df_ranked = df_ranked[~df_ranked['algorithm'].isin(['aes128','aes256'])]

    print("\n" + "="*60)
    print("ALGORITHM RANKING TABLE")
    print("="*60)
    print(df_ranked[['algorithm','avg_exec_time_s','avg_energy_J',
                     'avg_temp_C','avg_cpu_%','avg_ram_%','performance_score']].to_string(index=False))
    print("="*60)

    best_algo_row = df_ranked.iloc[0]
    best_algo = best_algo_row['algorithm']
    best_ciphertext = best_algo_row['ciphertext']

    print(f"\nBEST OPTIMIZED ALGORITHM: {best_algo}")
    print(f"Ciphertext:\n{best_ciphertext}")

    # Lưu ciphertext
    with open('encrypted_output.txt', 'w') as f:
        f.write(best_ciphertext)

    return df_ranked, best_ciphertext

# -----------------------
# Load mô hình và scaler
# -----------------------
try:
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_time = joblib.load('scaler_time.pkl')
    scaler_energy = joblib.load('scaler_energy.pkl')
    
    inp_dim = 15 + len(algo_info)  # 15 features + thuật toán
    model_time = torch.load('tabnet_time_model.pth', map_location='cpu')
    model_energy = torch.load('tabnet_energy_model.pth', map_location='cpu')
    model_time.eval()
    model_energy.eval()

    print("\nĐang dự đoán thuật toán tối ưu và mã hóa dữ liệu...")
    ranked_df, encrypted_data = predict_best_algorithm(model_time, model_energy,
                                                        scaler_X, scaler_time, scaler_energy)
except Exception as e:
    print(f"Lỗi khi tải model hoặc scaler: {e}")
    sys.exit(1)
