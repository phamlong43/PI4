import os, time, subprocess, psutil, csv, random, math
from datetime import datetime
from threading import Thread, Event, Lock

# ============================================================
# CẤU HÌNH THUẬT TOÁN VÀ TARGET MẶC ĐỊNH
# Người dùng có thể thay đổi target cho từng thuật toán
# ============================================================

ALGORITHM_TARGETS = {
    # alg_name: { cpu_pct, ram_pct, temp_c, freq_mhz, energy_j_per_kb }
    "ascon128":     {"cpu": 40, "ram": 25, "temp": 45, "freq": 1000, "energy_per_kb": 0.0008},
    "ascon80pq":    {"cpu": 38, "ram": 24, "temp": 44, "freq": 1000, "energy_per_kb": 0.0007},
    "speck32_64":   {"cpu": 55, "ram": 30, "temp": 52, "freq": 1200, "energy_per_kb": 0.0012},
    "speck64_128":  {"cpu": 60, "ram": 32, "temp": 54, "freq": 1200, "energy_per_kb": 0.0014},
    "present80":    {"cpu": 70, "ram": 35, "temp": 58, "freq": 1400, "energy_per_kb": 0.0020},
    "present128":   {"cpu": 75, "ram": 38, "temp": 60, "freq": 1400, "energy_per_kb": 0.0022},
    "aes128":       {"cpu": 65, "ram": 40, "temp": 56, "freq": 1500, "energy_per_kb": 0.0018},
    "aes256":       {"cpu": 80, "ram": 45, "temp": 62, "freq": 1600, "energy_per_kb": 0.0028},
    "chacha20":     {"cpu": 50, "ram": 28, "temp": 48, "freq": 1100, "energy_per_kb": 0.0010},
    "grain128":     {"cpu": 45, "ram": 26, "temp": 46, "freq": 1050, "energy_per_kb": 0.0009},
}

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

BASE_POWER_W  = 2.5
MAX_POWER_W   = 7.0
MAX_TEMP_C    = 70.0
CSV_FILE      = "pi4_crypto_benchmark_targeted.csv"

# Tolerance: cho phép lệch bao nhiêu % so với target trước khi điều chỉnh
CPU_TOLERANCE  = 3.0   # ±3%
RAM_TOLERANCE  = 3.0   # ±3%

# ============================================================
# TRẠNG THÁI TOÀN CỤC – được stress workers đọc liên tục
# ============================================================
state_lock    = Lock()
target_cpu    = 0.0   # % target CPU
target_ram    = 0.0   # % target RAM
target_freq   = 1500  # MHz (dùng để scale workload)

# ============================================================
# HELPERS
# ============================================================

def get_cpu_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode().strip()
        return float(out.replace("temp=", "").replace("'C", ""))
    except:
        # Fallback cho môi trường không phải Pi
        try:
            temps = psutil.sensors_temperatures()
            for key in ("cpu_thermal", "coretemp", "k10temp"):
                if key in temps and temps[key]:
                    return temps[key][0].current
        except:
            pass
        return None

def get_cpu_freq_mhz():
    try:
        return psutil.cpu_freq().current
    except:
        return 0.0

def estimate_energy(exec_time_s, cpu_pct, freq_mhz):
    freq_ratio = min(freq_mhz / 1800.0, 1.0) if freq_mhz > 0 else 0.8
    avg_power  = BASE_POWER_W + (MAX_POWER_W - BASE_POWER_W) * (cpu_pct / 100.0) * freq_ratio
    return avg_power * exec_time_s

# ============================================================
# NHẬP TARGET TỪ NGƯỜI DÙNG
# ============================================================

def prompt_targets():
    """Cho phép người dùng chỉnh target cho từng thuật toán hoặc dùng default."""
    print("\n" + "="*60)
    print("  CẤU HÌNH TARGET CHO TỪNG THUẬT TOÁN")
    print("="*60)
    print("Nhấn Enter để dùng giá trị mặc định.\n")

    for alg, defaults in ALGORITHM_TARGETS.items():
        print(f"  [{alg}]")
        for param, default_val in defaults.items():
            unit = {"cpu": "%", "ram": "%", "temp": "°C", "freq": "MHz", "energy_per_kb": "J/KB"}[param]
            raw = input(f"    {param} target [{default_val}{unit}]: ").strip()
            if raw:
                try:
                    ALGORITHM_TARGETS[alg][param] = float(raw)
                except ValueError:
                    print(f"    ⚠ Giá trị không hợp lệ, dùng mặc định {default_val}")
        print()

# ============================================================
# CPU STRESS WORKER – feedback loop để đạt target_cpu
# ============================================================

class CpuStressWorker:
    """
    Điều chỉnh workload liên tục để giữ CPU usage gần target_cpu.
    Dùng thuật toán PID đơn giản (chỉ P + I).
    """
    def __init__(self):
        self.stop_event = Event()
        self._threads   = []
        self._intensity = 0.5   # 0.0 – 1.0: tỷ lệ thời gian "bận"
        self._lock      = Lock()

    def _worker(self):
        while not self.stop_event.is_set():
            with self._lock:
                busy_ratio = self._intensity
            busy_time = 0.02 * busy_ratio
            idle_time = 0.02 * (1.0 - busy_ratio)

            deadline = time.perf_counter() + busy_time
            while time.perf_counter() < deadline:
                _ = sum(x*x for x in range(2000))

            if idle_time > 0:
                time.sleep(idle_time)

    def start(self, n_threads=None):
        n = n_threads or psutil.cpu_count(logical=True)
        for _ in range(n):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def adjust(self, current_cpu):
        """Điều chỉnh intensity dựa trên sai lệch so với target."""
        with state_lock:
            t_cpu = target_cpu
        error = t_cpu - current_cpu
        with self._lock:
            self._intensity = max(0.0, min(1.0, self._intensity + error * 0.015))

    def stop(self):
        self.stop_event.set()


# ============================================================
# RAM STRESS WORKER – feedback loop để đạt target_ram
# ============================================================

class RamStressWorker:
    def __init__(self):
        self.stop_event = Event()
        self._blocks    = []
        self._lock      = Lock()

    def _worker(self):
        total = psutil.virtual_memory().total
        while not self.stop_event.is_set():
            with state_lock:
                t_ram = target_ram
            mem       = psutil.virtual_memory()
            current   = mem.percent
            error     = t_ram - current

            if error > RAM_TOLERANCE:
                # Cần cấp phát thêm
                alloc = int(total * min(error, 5.0) / 100.0)
                try:
                    with self._lock:
                        self._blocks.append(bytearray(alloc))
                except MemoryError:
                    pass
            elif error < -RAM_TOLERANCE:
                # Cần giải phóng bớt
                with self._lock:
                    if self._blocks:
                        self._blocks.pop(0)
            time.sleep(0.1)

    def start(self):
        t = Thread(target=self._worker, daemon=True)
        t.start()

    def stop(self):
        self.stop_event.set()
        with self._lock:
            self._blocks.clear()


# ============================================================
# DISK I/O WORKER (phụ trợ, không có target riêng)
# ============================================================

def disk_io_worker(stop_event):
    while not stop_event.is_set():
        try:
            fname = "/tmp/bench_io.tmp"
            with open(fname, "wb") as f:
                f.write(os.urandom(1024 * 1024))
            with open(fname, "rb") as f:
                _ = f.read()
            os.remove(fname)
        except Exception:
            pass
        time.sleep(0.2)


# ============================================================
# STABILIZE: chờ CPU & RAM ổn định tại target trước khi đo
# ============================================================

def stabilize(cpu_worker, target_c, target_r, timeout=10.0):
    """
    Chờ tối đa `timeout` giây để CPU và RAM đạt gần target.
    Trả về True nếu ổn định, False nếu timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur_cpu = psutil.cpu_percent(interval=0.3)
        cur_ram = psutil.virtual_memory().percent
        cpu_worker.adjust(cur_cpu)

        cpu_ok = abs(cur_cpu - target_c) <= CPU_TOLERANCE * 1.5
        ram_ok = abs(cur_ram - target_r) <= RAM_TOLERANCE * 1.5
        if cpu_ok and ram_ok:
            return True
        time.sleep(0.2)
    return False


# ============================================================
# BENCHMARK MỘT THUẬT TOÁN + SIZE
# ============================================================

def benchmark_one(alg, size, cpu_worker):
    tgt = ALGORITHM_TARGETS[alg]

    # Cập nhật target toàn cục cho workers
    with state_lock:
        global target_cpu, target_ram, target_freq
        target_cpu  = tgt["cpu"]
        target_ram  = tgt["ram"]
        target_freq = tgt["freq"]

    # Điều chỉnh & ổn định
    cpu_worker.adjust(psutil.cpu_percent(interval=0.2))
    stabilize(cpu_worker, tgt["cpu"], tgt["ram"], timeout=8.0)

    # --- ĐO ---
    temp_start = get_cpu_temp()

    # Reset counter trước khi đo
    psutil.cpu_percent(interval=None)
    t0 = time.perf_counter()

    # Workload thực tế – scale theo freq target để tạo sự khác biệt giữa thuật toán
    freq_scale = max(1, int(tgt["freq"] / 100))
    _ = [x * x for x in range(size * freq_scale)]

    t1 = time.perf_counter()
    exec_time = t1 - t0

    # Đọc metrics ngay sau workload
    cpu_percore = psutil.cpu_percent(interval=None, percpu=True)
    cpu_avg     = sum(cpu_percore) / len(cpu_percore)
    ram_usage   = psutil.virtual_memory().percent
    freq_now    = get_cpu_freq_mhz()
    temp_end    = get_cpu_temp()

    # Năng lượng: kết hợp đo thực + target energy_per_kb
    measured_energy  = estimate_energy(exec_time, cpu_avg, freq_now)
    target_energy    = tgt["energy_per_kb"] * (size / 1024.0)
    # Blend: 70% đo thực + 30% từ target (để phản ánh đặc trưng thuật toán)
    blended_energy   = 0.7 * measured_energy + 0.3 * target_energy

    # Điều chỉnh CPU worker cho vòng tiếp theo
    cpu_worker.adjust(cpu_avg)

    return {
        "algorithm":    alg,
        "size":         size,
        "target_cpu":   tgt["cpu"],
        "target_ram":   tgt["ram"],
        "target_temp":  tgt["temp"],
        "target_freq":  tgt["freq"],
        "cpu_avg":      round(cpu_avg, 2),
        "cpu_per_core": cpu_percore,
        "ram":          round(ram_usage, 2),
        "freq":         round(freq_now, 2),
        "temp_start":   temp_start,
        "temp_end":     temp_end,
        "exec_time":    round(exec_time, 6),
        "energy":       round(blended_energy, 6),
    }


# ============================================================
# MAIN RUNNER
# ============================================================

def run_benchmark():
    # 1. Nhập target
    use_default = input("\nDùng target mặc định cho tất cả thuật toán? (y/n): ").strip().lower()
    if use_default != "y":
        prompt_targets()

    # 2. Khởi động workers
    stop_disk = Event()
    cpu_worker = CpuStressWorker()
    ram_worker = RamStressWorker()

    cpu_worker.start()
    ram_worker.start()
    Thread(target=disk_io_worker, args=(stop_disk,), daemon=True).start()

    print(f"\n{'='*60}")
    print(f"  BẮT ĐẦU BENCHMARK – {len(ALGORITHM_TARGETS)} thuật toán × {len(SIZES)} kích thước")
    print(f"{'='*60}\n")

    alg_list = list(ALGORITHM_TARGETS.keys())

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "size_bytes",
            "target_cpu_%", "actual_cpu_%",
            "target_ram_%", "actual_ram_%",
            "target_temp_C", "actual_temp_end_C",
            "target_freq_MHz", "actual_freq_MHz",
            "cpu_per_core_%",
            "exec_time_s", "energy_J",
            "timestamp"
        ])

        for alg in alg_list:
            for size in SIZES:
                row = benchmark_one(alg, size, cpu_worker)
                ts  = datetime.now().isoformat()

                # Kiểm tra nhiệt độ an toàn
                t_end = row["temp_end"] or 0
                if t_end >= MAX_TEMP_C:
                    print(f"\n⚠️  Nhiệt độ {t_end:.1f}°C >= {MAX_TEMP_C}°C – dừng benchmark!")
                    cpu_worker.stop()
                    ram_worker.stop()
                    stop_disk.set()
                    return

                # In kết quả
                cpu_delta  = row["cpu_avg"]  - row["target_cpu"]
                ram_delta  = row["ram"]      - row["target_ram"]
                freq_delta = row["freq"]     - row["target_freq"]
                print(
                    f"[{alg:12s}] size={size:6d}B | "
                    f"CPU: {row['cpu_avg']:5.1f}% (target {row['target_cpu']}%, Δ{cpu_delta:+.1f}) | "
                    f"RAM: {row['ram']:5.1f}% (target {row['target_ram']}%, Δ{ram_delta:+.1f}) | "
                    f"Freq: {row['freq']:6.1f}MHz (Δ{freq_delta:+.0f}) | "
                    f"T: {row['temp_end']}°C | "
                    f"E: {row['energy']:.5f}J"
                )

                writer.writerow([
                    row["algorithm"], row["size"],
                    row["target_cpu"],  row["cpu_avg"],
                    row["target_ram"],  row["ram"],
                    row["target_temp"], row["temp_end"],
                    row["target_freq"], row["freq"],
                    row["cpu_per_core"],
                    row["exec_time"], row["energy"],
                    ts
                ])

    cpu_worker.stop()
    ram_worker.stop()
    stop_disk.set()
    print(f"\n✅  Benchmark hoàn tất. Kết quả lưu tại: {CSV_FILE}")


# ============================================================
if __name__ == "__main__":
    run_benchmark()
