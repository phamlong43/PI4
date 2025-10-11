import os, time, subprocess, psutil, csv, random
from datetime import datetime
from threading import Thread, Event

# -----------------------------
ALGORITHMS = {
    "ascon128": 128,
    "ascon80pq": 80,
    "speck32_64": 64,
    "speck64_128": 128,
    "present80": 80,
    "present128": 128,
    "aes128": 128,
    "aes256": 256,
    "chacha20": 128,
    "grain128": 128
}

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

BASE_POWER_W = 2.5
MAX_POWER_W = 7.0

CSV_FILE = "pi4_crypto_benchmark_fullstress.csv"

# Stress level configuration
STRESS_MODE = "progressive"  # "progressive" hoặc "random"
MAX_TEMP_C = 70.0
WARNING_TEMP_C = 65.0

# Global stress level (0.0 to 1.0)
stress_level = 0.0

# -----------------------------
def get_cpu_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode().strip()
        return float(out.replace("temp=", "").replace("'C", ""))
    except:
        return None

def estimate_energy(exec_time_s, cpu_usage_percent):
    avg_power = BASE_POWER_W + (MAX_POWER_W - BASE_POWER_W) * (cpu_usage_percent / 100.0)
    return avg_power * exec_time_s

# CPU stress với điều chỉnh cấp độ
def cpu_stress(stop_event):
    global stress_level
    while not stop_event.is_set():
        # Tính số cores dựa trên stress level
        num_cores = max(1, int(psutil.cpu_count() * stress_level))
        if num_cores == 0:
            time.sleep(0.01)
            continue
        
        loop_size = int(500000 * (0.3 + stress_level * 0.7))  # 500k-1000k
        for _ in range(num_cores):
            _ = [x*x for x in range(loop_size)]
        time.sleep(0.01)

# RAM stress với điều chỉnh cấp độ
def ram_stress(stop_event):
    global stress_level
    total_mem = psutil.virtual_memory().total
    blocks = []

    while not stop_event.is_set():
        try:
            mem = psutil.virtual_memory()
            
            if mem.available < 100*1024*1024:  # Nếu còn < 100MB
                blocks.clear()
                time.sleep(0.1)
                continue
            
            # RAM target dựa trên stress level (5% - 50%)
            min_ram = 5
            max_ram = 50
            target_percent = min_ram + (max_ram - min_ram) * stress_level
            target_bytes = int(total_mem * target_percent / 100)

            current_bytes = sum(len(b) for b in blocks)

            if current_bytes < target_bytes:
                alloc_bytes = min(target_bytes - current_bytes, 20*1024*1024)
                blocks.append(bytearray(alloc_bytes))
            elif current_bytes > target_bytes:
                while blocks and current_bytes > target_bytes:
                    b = blocks.pop(0)
                    current_bytes -= len(b)
            
            time.sleep(0.05)
        except MemoryError:
            blocks.clear()
            time.sleep(0.1)

# Disk I/O stress với điều chỉnh cấp độ
def disk_io_stress(stop_event):
    global stress_level
    while not stop_event.is_set():
        try:
            # Disk size dựa trên stress level (0.5MB - 2MB)
            min_size = 0.5 * 1024 * 1024
            max_size = 2 * 1024 * 1024
            io_size = int(min_size + (max_size - min_size) * stress_level)
            
            fname = "/tmp/io_test.tmp"
            with open(fname, "wb") as f:
                f.write(os.urandom(io_size))
            with open(fname, "rb") as f:
                _ = f.read()
            os.remove(fname)
        except Exception as e:
            pass
        time.sleep(0.1)

# Network stress
def network_stress(stop_event):
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while not stop_event.is_set():
            try:
                sock.sendto(b"x"*512, ("127.0.0.1", 5005))
            except:
                pass
            time.sleep(0.05)
        sock.close()
    except Exception as e:
        pass

def benchmark_algorithm_fullstress(alg, size):
    global stress_level
    temp_start = get_cpu_temp()
    t0 = time.perf_counter()

    # Simulate algorithm execution
    _ = [x*x for x in range(size*100)]

    t1 = time.perf_counter()
    temp_end = get_cpu_temp()

    cpu_percore = psutil.cpu_percent(interval=1, percpu=True)
    cpu_avg = sum(cpu_percore)/len(cpu_percore)
    ram_usage = psutil.virtual_memory().percent
    freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
    exec_time = t1-t0
    energy = estimate_energy(exec_time, cpu_avg)
    
    return {
        "algorithm": alg, "size": size, "cpu_avg": cpu_avg, "cpu_per_core": cpu_percore,
        "ram": ram_usage, "freq": freq, "temp_start": temp_start, "temp_end": temp_end,
        "exec_time": exec_time, "energy": energy, "stress_level": round(stress_level, 2)
    }

# Cập nhật stress level
def update_stress_level(alg_index, alg_count, size_index, size_count):
    global stress_level
    if STRESS_MODE == "progressive":
        # Tăng dần từ 0 đến 1.0
        total_iterations = alg_count * size_count
        current_iteration = alg_index * size_count + size_index
        stress_level = min(1.0, current_iteration / total_iterations)
    elif STRESS_MODE == "random":
        # Random từ 0 đến 1.0
        stress_level = random.uniform(0.0, 1.0)

# Chạy benchmark với tất cả các thuật toán
def run_benchmark_all_fullstress():
    global stress_level
    stop_event = Event()

    # Start stress threads
    threads = [
        Thread(target=cpu_stress, args=(stop_event,), daemon=True),
        Thread(target=ram_stress, args=(stop_event,), daemon=True),
        Thread(target=disk_io_stress, args=(stop_event,), daemon=True),
        Thread(target=network_stress, args=(stop_event,), daemon=True)
    ]
    for th in threads:
        th.start()

    alg_list = list(ALGORITHMS.keys())
    alg_count = len(alg_list)
    size_count = len(SIZES)

    # Write CSV
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "size_bytes", "stress_level", "cpu_avg_%", "cpu_per_core_%",
            "ram_%", "freq_MHz", "temp_start_C", "temp_end_C", "exec_time_s", "energy_J", "timestamp"
        ])

        stop_benchmark = False
        for alg_idx, alg in enumerate(alg_list):
            for size_idx, size in enumerate(SIZES):
                if stop_benchmark:
                    break
                
                # Cập nhật stress level
                update_stress_level(alg_idx, alg_count, size_idx, size_count)
                
                row = benchmark_algorithm_fullstress(alg, size)
                timestamp = datetime.now().isoformat()
                
                print(f"\n[Stress: {row['stress_level']}] Algorithm: {row['algorithm']} | Size: {row['size']} bytes")
                print(f"  CPU avg: {row['cpu_avg']:.2f}% | RAM: {row['ram']:.2f}% | Freq: {row['freq']:.2f} MHz")
                print(f"  Temp: {row['temp_start']:.1f}°C → {row['temp_end']:.1f}°C")
                print(f"  Exec time: {row['exec_time']:.6f}s | Energy: {row['energy']:.6f}J")
                
                writer.writerow([
                    row["algorithm"], row["size"], row["stress_level"],
                    round(row["cpu_avg"], 2), row["cpu_per_core"], round(row["ram"], 2),
                    round(row["freq"], 2), row["temp_start"], row["temp_end"],
                    round(row["exec_time"], 6), round(row["energy"], 6), timestamp
                ])
                
                if row["temp_end"] >= MAX_TEMP_C:
                    print(f"\n⚠️  CPU {row['temp_end']:.1f}°C >= {MAX_TEMP_C}°C, dừng benchmark")
                    stop_benchmark = True
                    break
    
    stop_event.set()
    print(f"\nBenchmark hoàn thành! Kết quả lưu vào: {CSV_FILE}")

# -----------------------------
if __name__ == "__main__":
    print(f"Mode stress: {STRESS_MODE}")
    print(f"  - 'progressive': Tăng dần từ 0 → 100%")
    print(f"  - 'random': Random 0-100% mỗi lần\n")
    run_benchmark_all_fullstress()
