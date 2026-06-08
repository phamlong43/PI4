import os, time, psutil, subprocess
from threading import Thread, Event, Lock

# ============================================================
#  NHẬP TARGET TỪ NGƯỜI DÙNG
# ============================================================

def prompt_float(label, unit, default=None, min_val=None, max_val=None):
    while True:
        hint = f" [default: {default}{unit}]" if default is not None else ""
        raw = input(f"  {label}{hint}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"    ⚠ Tối thiểu {min_val}{unit}")
                continue
            if max_val is not None and val > max_val:
                print(f"    ⚠ Tối đa {max_val}{unit}")
                continue
            return val
        except ValueError:
            print("    ⚠ Nhập số hợp lệ")

def get_input():
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    print("\n" + "="*50)
    print("  NHẬP TARGET HỆ THỐNG")
    print("="*50)
    print(f"  (RAM vật lý hiện có: {total_ram_gb:.1f} GB)\n")

    targets = {}
    targets["cpu"]    = prompt_float("CPU usage",             "%",   default=50,  min_val=0,   max_val=95)
    targets["ram_gb"] = prompt_float("RAM usage",             " GB", default=1.0, min_val=0.1, max_val=round(total_ram_gb * 0.85, 1))
    targets["temp"]   = prompt_float("Nhiệt độ CPU target",   "°C",  default=50,  min_val=30,  max_val=75)

    print()
    return targets

# ============================================================
#  ĐỌC NHIỆT ĐỘ CPU
# ============================================================

def get_cpu_temp():
    # Raspberry Pi
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL).decode()
        return float(out.replace("temp=", "").replace("'C", "").strip())
    except Exception:
        pass
    # Linux chung (psutil)
    try:
        temps = psutil.sensors_temperatures()
        for key in ("cpu_thermal", "coretemp", "k10temp", "acpitz"):
            if key in temps and temps[key]:
                return temps[key][0].current
    except Exception:
        pass
    return None

# ============================================================
#  CPU STRESS WORKER  (busy-loop có điều chỉnh duty cycle)
# ============================================================

class CpuWorker:
    """
    Mỗi logical core có 1 thread.
    Duty cycle (0.0–1.0) điều chỉnh tỷ lệ thời gian "bận" / "nghỉ"
    để ép CPU% bám target.

    FIX: Giảm SLOT từ 50ms → 20ms để giảm jitter,
         đặc biệt trên máy nhiều core.
    """
    SLOT = 0.02     # FIX: 20ms thay vì 50ms → điều chỉnh mượt hơn

    def __init__(self):
        self._duty   = 0.0
        self._lock   = Lock()
        self._stop   = Event()
        self._n_core = psutil.cpu_count(logical=True)

    def _core_loop(self):
        while not self._stop.is_set():
            with self._lock:
                d = self._duty
            busy = self.SLOT * d
            idle = self.SLOT * (1.0 - d)
            if busy > 0:
                end = time.perf_counter() + busy
                while time.perf_counter() < end:
                    pass            # spin – tạo tải CPU
            if idle > 0:
                time.sleep(idle)

    def start(self):
        for _ in range(self._n_core):
            Thread(target=self._core_loop, daemon=True).start()

    def set_duty(self, d):
        with self._lock:
            self._duty = max(0.0, min(1.0, d))

    def stop(self):
        self._stop.set()

# ============================================================
#  RAM STRESS WORKER  (giữ số GB đã cấp phát)
# ============================================================

class RamWorker:
    CHUNK = 64 * 1024 * 1024        # cấp / giải phóng từng 64 MB

    def __init__(self):
        self._blocks      = []
        self._target_gb   = 0.0
        self._lock        = Lock()
        self._stop        = Event()
        self._baseline_gb = psutil.virtual_memory().used / 1024**3
        print(f"  [RamWorker] RAM nền (baseline): {self._baseline_gb:.2f} GB")

    def set_target(self, gb):
        with self._lock:
            self._target_gb = gb

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                tgt_gb = self._target_gb

            need_gb    = max(0.0, tgt_gb - self._baseline_gb)
            need_bytes = int(need_gb * 1024**3)
            current    = sum(len(b) for b in self._blocks)
            diff       = need_bytes - current

            if diff > self.CHUNK // 2:
                alloc = min(diff, self.CHUNK)
                try:
                    buf = bytearray(alloc)
                    for i in range(0, len(buf), 4096):
                        buf[i] = 0xFF
                    self._blocks.append(buf)
                except MemoryError:
                    pass

            elif diff < -(self.CHUNK // 2) and self._blocks:
                self._blocks.pop(0)

            time.sleep(0.1)

    def start(self):
        Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        self._blocks.clear()

# ============================================================
#  FEEDBACK CONTROLLER  (vòng điều khiển chính)
# ============================================================

class Controller:
    # FIX: Tăng hệ số PID để hội tụ nhanh hơn
    KP_CPU = 0.05   # tăng từ 0.02
    KI_CPU = 0.02   # tăng từ 0.005

    def __init__(self, targets, cpu_worker, ram_worker):
        self.targets    = targets
        self.cpu_worker = cpu_worker
        self.ram_worker = ram_worker
        self._stop      = Event()
        self._i_cpu     = 0.0

        # FIX: Bỏ * 0.8 — khởi tạo duty đúng bằng target
        self.cpu_worker.set_duty(targets["cpu"] / 100.0)
        self.ram_worker.set_target(targets["ram_gb"])

    def _loop(self):
        # FIX: Warm-up — discard lần đọc đầu tiên (luôn trả về 0.0)
        psutil.cpu_percent()
        time.sleep(0.5)

        while not self._stop.is_set():
            # FIX: interval=None (non-blocking) thay vì interval=0.5
            #      Tránh block thread controller 500ms mỗi vòng
            cur_cpu  = psutil.cpu_percent(interval=None)
            cur_ram  = psutil.virtual_memory()
            cur_temp = get_cpu_temp()
            cur_freq = psutil.cpu_freq()

            # ---- CPU PID ----
            err_cpu     = self.targets["cpu"] - cur_cpu
            self._i_cpu = max(-1.0, min(1.0, self._i_cpu + err_cpu * self.KI_CPU))
            delta       = err_cpu * self.KP_CPU + self._i_cpu

            with self.cpu_worker._lock:
                old_duty = self.cpu_worker._duty
            self.cpu_worker.set_duty(old_duty + delta)

            # ---- In trạng thái ----
            ram_gb_used  = cur_ram.used / 1024**3
            ram_total    = cur_ram.total / 1024**3
            temp_str     = f"{cur_temp:.1f}°C" if cur_temp else "N/A"
            freq_str     = f"{cur_freq.current:.0f}MHz" if cur_freq else "N/A"

            tgt_cpu  = self.targets["cpu"]
            tgt_ram  = self.targets["ram_gb"]
            tgt_temp = self.targets["temp"]

            allocated_gb = sum(len(b) for b in self.ram_worker._blocks) / 1024**3
            need_gb      = max(0.0, tgt_ram - self.ram_worker._baseline_gb)

            cpu_ok  = abs(cur_cpu - tgt_cpu)      <= 3.0
            ram_ok  = abs(allocated_gb - need_gb) <= 0.2
            status  = "✅" if (cpu_ok and ram_ok) else "⏳"

            print(
                f"\r{status} "
                f"CPU: {cur_cpu:5.1f}% / target {tgt_cpu}%  |  "
                f"RAM total: {ram_gb_used:.2f}/{ram_total:.1f}GB  "
                f"(script: +{allocated_gb:.2f}GB / cần +{need_gb:.2f}GB)  |  "
                f"Temp: {temp_str} / target {tgt_temp}°C  |  "
                f"Freq: {freq_str}   ",
                end="", flush=True
            )

            # Cảnh báo nếu quá nhiệt
            if cur_temp and cur_temp >= 75.0:
                print(f"\n⚠️  Nhiệt độ {cur_temp:.1f}°C quá cao! Đang giảm tải...")
                self.cpu_worker.set_duty(0.1)
                self._i_cpu = 0.0   # reset tích phân tránh windup

            # FIX: Sleep ngắn để vòng lặp chạy ~10 lần/giây
            time.sleep(0.1)

    def start(self):
        Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

# ============================================================
#  MAIN
# ============================================================

def main():
    targets = get_input()

    print("\n" + "="*50)
    print("  TARGET ĐÃ ĐẶT:")
    print(f"    CPU  : {targets['cpu']}%")
    print(f"    RAM  : {targets['ram_gb']} GB")
    print(f"    Temp : {targets['temp']}°C (theo dõi, không thể ép trực tiếp)")
    print("="*50)
    print("\nĐang khởi động workers...")
    print("Nhấn  Ctrl+C  để dừng.\n")

    cpu_worker = CpuWorker()
    ram_worker = RamWorker()
    controller = Controller(targets, cpu_worker, ram_worker)

    cpu_worker.start()
    ram_worker.start()
    controller.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nĐang dừng...")
        controller.stop()
        cpu_worker.stop()
        ram_worker.stop()
        time.sleep(0.5)
        print("Đã dừng. Bye!")

if __name__ == "__main__":
    main()
