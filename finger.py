import json
import time
import serial
import adafruit_fingerprint

USER_DB_FILE = "user_db.json"

# Khởi tạo module vân tay
uart = serial.Serial("/dev/ttyS0", baudrate=57600, timeout=1)
finger = adafruit_fingerprint.Adafruit_Fingerprint(uart)

def load_user_db():
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_user_db(user_db):
    with open(USER_DB_FILE, "w") as f:
        json.dump(user_db, f, indent=4)

def enroll_finger(location, username):
    """Đăng ký vân tay tại vị trí location và lưu tên"""
    print(f"Đăng ký vân tay cho {username} tại ID {location}")
    for fingerimg in range(1, 3):
        if fingerimg == 1:
            print("Đặt ngón tay lên cảm biến...")
        else:
            print("Đặt lại cùng ngón tay...")

        while True:
            i = finger.get_image()
            if i == adafruit_fingerprint.OK:
                print("Đã chụp ảnh")
                break
            elif i == adafruit_fingerprint.NOFINGER:
                pass
            else:
                print(f"Lỗi khi lấy ảnh: {i}")
                return False

        i = finger.image_2_tz(fingerimg)
        if i != adafruit_fingerprint.OK:
            print(f"Lỗi khi tạo mẫu ảnh: {i}")
            return False

        if fingerimg == 1:
            print("Bỏ tay ra cảm biến")
            time.sleep(1)
            while finger.get_image() != adafruit_fingerprint.NOFINGER:
                pass

    i = finger.create_model()
    if i != adafruit_fingerprint.OK:
        print(f"Lỗi khi tạo model: {i}")
        return False

    i = finger.store_model(location)
    if i != adafruit_fingerprint.OK:
        print(f"Lỗi khi lưu model: {i}")
        return False

    # Lưu tên người dùng
    user_db = load_user_db()
    user_db[str(location)] = username
    save_user_db(user_db)
    print(f"Đăng ký vân tay thành công cho {username}")
    return True

def delete_finger(location):
    """Xóa mẫu vân tay theo location"""
    i = finger.delete_model(location)
    if i == adafruit_fingerprint.OK:
        user_db = load_user_db()
        if str(location) in user_db:
            del user_db[str(location)]
            save_user_db(user_db)
        print(f"Xóa mẫu vân tay ID {location} thành công")
        return True
    else:
        print(f"Xóa mẫu vân tay thất bại: {i}")
        return False

def update_username(location, new_username):
    """Cập nhật tên người dùng"""
    user_db = load_user_db()
    if str(location) in user_db:
        user_db[str(location)] = new_username
        save_user_db(user_db)
        print(f"Cập nhật tên ID {location} thành công")
        return True
    else:
        print(f"Không tìm thấy ID {location} để cập nhật")
        return False

def authenticate_finger():
    """Xác thực vân tay và trả về (True, tên người) hoặc (False, None)"""
    print("Đặt ngón tay lên cảm biến...")
    while True:
        i = finger.get_image()
        if i == adafruit_fingerprint.OK:
            break
        elif i == adafruit_fingerprint.NOFINGER:
            pass
        else:
            print(f"Lỗi khi lấy ảnh: {i}")
            return False, None

    i = finger.image_2_tz(1)
    if i != adafruit_fingerprint.OK:
        print(f"Lỗi tạo mẫu ảnh: {i}")
        return False, None

    i = finger.finger_fast_search()
    if i == adafruit_fingerprint.OK:
        location = finger.finger_id
        confidence = finger.confidence
        user_db = load_user_db()
        username = user_db.get(str(location), "Người dùng không xác định")
        print(f"Xác thực thành công: {username} (ID {location}) - Độ tin cậy: {confidence}")
        return True, username
    else:
        print("Không tìm thấy vân tay phù hợp")
        return False, None
