import time, psutil, subprocess
from collections import deque
from threading import Thread, Event, Lock
from multiprocessing import Process, Value, Event as MpEvent

# ============================================================
#  NHẬP TARGET
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
                print(f"    ⚠ Tối thiểu {min_val}{unit}"); continue
            if max_val is not None and val > max_val:
                print(f"    ⚠ Tối đa {max_val}{unit}"); continue
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
    targets["cpu"]    = prompt_float("CPU usage",           "%",   default=50,  min_val=0,   max_val=95)
    targets["ram_gb"] = prompt_float("RAM usage",           " GB", default=1.0, min_val=0.1, max_val=round(total_ram_gb * 0.85, 1))
    targets["temp"]   = prompt_float("Nhiệt độ CPU target", "°C",  default=50,  min_val=30,  max_val=75)
    print()
    return targets

# ============================================================
#  ĐỌC NHIỆT ĐỘ
# ============================================================

def get_cpu_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL).decode()
        return float(out.replace("temp=", "").replace("'C", "").strip())
    except Exception:
        pass
    try:
        temps = psutil.sensors_temperatures()
        for key in ("cpu_thermal", "coretemp", "k10temp", "acpitz"):
            if key in temps and temps[key]:
                return temps[key][0].current
    except Exception:
        pass
    return None

# ============================================================
#  CPU WORKER  (multiprocessing — bypass GIL)
# ============================================================

def _cpu_worker_proc(shared_duty, stop_event):
    SLOT = 0.02
    while not stop_event.is_set():
        d    = shared_duty.value
        busy = SLOT * d
        idle = SLOT * (1.0 - d)
        if busy > 0:
            end = time.perf_counter() + busy
            while time.perf_counter() < end:
                pass
        if idle > 0:
            time.sleep(idle)

class CpuWorker:
    def __init__(self):
        self._n_core    = psutil.cpu_count(logical=True)
        self._duty      = Value('d', 0.0)
        self._stop      = MpEvent()
        self._processes = []

    def start(self):
        for _ in range(self._n_core):
            p = Process(target=_cpu_worker_proc,
                        args=(self._duty, self._stop), daemon=True)
            p.start()
            self._processes.append(p)

    def set_duty(self, d):
        self._duty.value = max(0.0, min(1.0, d))

    def get_duty(self):
        return self._duty.value

    def stop(self):
        self._stop.set()
        for p in self._processes:
            p.terminate()

# ============================================================
#  RAM WORKER
# ============================================================

class RamWorker:
    CHUNK = 64 * 1024 * 1024

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
            need_bytes = int(max(0.0, tgt_gb - self._baseline_gb) * 1024**3)
            current    = sum(len(b) for b in self._blocks)
            diff       = need_bytes - current
            if diff > self.CHUNK // 2:
                try:
                    buf = bytearray(min(diff, self.CHUNK))
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
#  CONTROLLER
#
#  Hai fix chính so với version trước:
#
#  FIX 1 — MOVING AVERAGE (smoothing)
#    psutil.cpu_percent() rất nhiễu khi đọc mỗi 100ms.
#    Lấy trung bình 5 mẫu gần nhất trước khi đưa vào PID
#    → loại bỏ spike ngẫu nhiên, PID không phản ứng với nhiễu.
#
#  FIX 2 — PID điều chỉnh DUTY trực tiếp, không cộng delta
#    Thay vì: duty += delta  (tích lũy sai số, dễ overshoot)
#    Dùng:    duty  = clamp(target_cpu/100 + P + I)
#    → duty luôn bám sát giá trị hợp lý, không drift.
#    KP nhỏ hơn (0.008) để tránh phản ứng quá mạnh.
# ============================================================

class Controller:
    KP      = 0.008   # nhỏ hơn để tránh overshoot
    KI      = 0.003   # tích phân chậm, chỉ bù steady-state error
    SMOOTH  = 8       # số mẫu moving average

    def __init__(self, targets, cpu_worker, ram_worker):
        self.targets    = targets
        self.cpu_worker = cpu_worker
        self.ram_worker = ram_worker
        self._stop      = Event()
        self._integral  = 0.0
        self._samples   = deque(maxlen=self.SMOOTH)

        self.cpu_worker.set_duty(targets["cpu"] / 100.0)
        self.ram_worker.set_target(targets["ram_gb"])

    def _smooth_cpu(self, raw):
        """Thêm mẫu mới, trả về trung bình SMOOTH mẫu gần nhất."""
        self._samples.append(raw)
        return sum(self._samples) / len(self._samples)

    def _loop(self):
        # Warm-up: nạp đủ SMOOTH mẫu trước khi PID bắt đầu
        psutil.cpu_percent()
        for _ in range(self.SMOOTH):
            self._samples.append(psutil.cpu_percent(interval=0.2))

        while not self._stop.is_set():
            raw_cpu  = psutil.cpu_percent(interval=0.2)   # interval=0.2: ít nhiễu hơn None
            cur_cpu  = self._smooth_cpu(raw_cpu)           # FIX 1: làm mượt

            cur_ram  = psutil.virtual_memory()
            cur_temp = get_cpu_temp()
            cur_freq = psutil.cpu_freq()

            tgt_cpu = self.targets["cpu"]

            # ---- PID ----
            err = tgt_cpu - cur_cpu
            self._integral = max(-5.0, min(5.0,           # anti-windup
                                self._integral + err * self.KI))

            # FIX 2: duty tính trực tiếp từ feedforward + correction
            # feedforward = target/100 (điểm xuất phát hợp lý)
            # correction  = P + I      (bù sai lệch)
            duty = (tgt_cpu / 100.0) + err * self.KP + self._integral / 100.0
            self.cpu_worker.set_duty(duty)

            # ---- Hiển thị ----
            ram_used_gb  = cur_ram.used / 1024**3
            ram_total_gb = cur_ram.total / 1024**3
            alloc_gb     = sum(len(b) for b in self.ram_worker._blocks) / 1024**3
            need_gb      = max(0.0, self.targets["ram_gb"] - self.ram_worker._baseline_gb)
            temp_str     = f"{cur_temp:.1f}°C" if cur_temp else "N/A"
            freq_str     = f"{cur_freq.current:.0f}MHz" if cur_freq else "N/A"

            cpu_ok = abs(cur_cpu - tgt_cpu) <= 3.0
            ram_ok = abs(alloc_gb - need_gb) <= 0.2
            status = "✅" if (cpu_ok and ram_ok) else "⏳"

            print(
                f"\r{status} "
                f"CPU: {cur_cpu:5.1f}% (raw:{raw_cpu:5.1f}%) / {tgt_cpu}%  |  "
                f"duty: {self.cpu_worker.get_duty():.3f}  |  "
                f"RAM: {ram_used_gb:.2f}/{ram_total_gb:.1f}GB (+{alloc_gb:.2f}/{need_gb:.2f}GB)  |  "
                f"Temp: {temp_str}/{self.targets['temp']}°C  Freq: {freq_str}   ",
                end="", flush=True
            )

            if cur_temp and cur_temp >= 75.0:
                print(f"\n⚠️  Nhiệt độ {cur_temp:.1f}°C quá cao! Giảm tải...")
                self.cpu_worker.set_duty(0.1)
                self._integral = 0.0

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
    print(f"    Temp : {targets['temp']}°C (theo dõi, không ép trực tiếp)")
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
