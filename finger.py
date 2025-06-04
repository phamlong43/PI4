import json
import time
import adafruit_fingerprint
import serial

DB_FILE = "finger_db.json"

# Kết nối cảm biến vân tay
uart = serial.Serial("/dev/ttyS0", baudrate=57600, timeout=1)
finger = adafruit_fingerprint.Adafruit_Fingerprint(uart)

# Đọc cơ sở dữ liệu JSON
def load_db():
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

# Lưu cơ sở dữ liệu JSON
def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

def read_templates():
    return finger.read_templates()

def enroll_finger(location):
    for fingerimg in range(1, 4):  # 3 lần đặt ngón tay
        if fingerimg == 1:
            print("Place finger on sensor...")
        else:
            print("Place same finger again...")
        start_time = time.time()
        while True:
            i = finger.get_image()
            if i == adafruit_fingerprint.OK:
                print("Image taken")
                break
            elif i == adafruit_fingerprint.NOFINGER:
                # Nếu quá 10s không có ngón tay đặt, báo lỗi
                if time.time() - start_time > 10:
                    return False, "Timeout: No finger detected"
            elif i == adafruit_fingerprint.IMAGEFAIL:
                return False, "Imaging error"
            else:
                return False, f"Error code: {i}"
        print("Templating...")
        i = finger.image_2_tz(fingerimg)
        if i != adafruit_fingerprint.OK:
            return False, f"Failed to template (code {i})"
        if fingerimg < 3:
            print("Remove finger")
            time.sleep(1)
            # Đợi tới khi ngón tay được rút ra
            while finger.get_image() != adafruit_fingerprint.NOFINGER:
                pass
    print("Creating model...")
    i = finger.create_model()
    if i != adafruit_fingerprint.OK:
        return False, f"Failed to create model (code {i})"
    print(f"Storing model #{location}...")
    i = finger.store_model(location)
    if i != adafruit_fingerprint.OK:
        return False, f"Failed to store model (code {i})"
    return True, "Enrollment successful"


def delete_finger(location):
    return finger.delete_model(location)

def search_finger():
    i = finger.get_image()
    if i != adafruit_fingerprint.OK:
        return None
    i = finger.image_2_tz(1)
    if i != adafruit_fingerprint.OK:
        return None
    i = finger.finger_fast_search()
    if i == adafruit_fingerprint.OK:
        return finger.finger_id
    return None
