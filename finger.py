import time
import serial
import adafruit_fingerprint
from digitalio import DigitalInOut, Direction
import board

led = DigitalInOut(board.D13)
led.direction = Direction.OUTPUT

# Khai báo UART kết nối cảm biến vân tay
uart = serial.Serial("/dev/ttyS0", baudrate=57600, timeout=1)
finger = adafruit_fingerprint.Adafruit_Fingerprint(uart)

def get_fingerprint():
    print("Chờ đặt ngón tay...")
    while finger.get_image() != adafruit_fingerprint.OK:
        pass

    print("Templating...")
    if finger.image_2_tz(1) != adafruit_fingerprint.OK:
        return False

    print("Tìm kiếm...")
    if finger.finger_search() != adafruit_fingerprint.OK:
        return False

    return True

def enroll_finger(location):
    for img in range(1, 3):
        if img == 1:
            print("Đặt ngón tay lên cảm biến...")
        else:
            print("Đặt lại ngón tay...")

        while True:
            i = finger.get_image()
            if i == adafruit_fingerprint.OK:
                print("Đã chụp ảnh")
                break
            elif i == adafruit_fingerprint.NOFINGER:
                print(".", end="")
                time.sleep(0.5)
            else:
                print("Lỗi khi chụp ảnh")
                return False

        print("\nChuyển đổi thành template...")
        i = finger.image_2_tz(img)
        if i != adafruit_fingerprint.OK:
            print("Lỗi khi chuyển đổi")
            return False

        if img == 1:
            print("Hãy nhấc ngón tay ra")
            time.sleep(1)
            while finger.get_image() != adafruit_fingerprint.NOFINGER:
                time.sleep(0.5)

    print("Ghép 2 mẫu thành 1...")
    if finger.create_model() != adafruit_fingerprint.OK:
        print("Mẫu không khớp")
        return False

    print("Lưu mẫu tại ID #%d..." % location)
    if finger.store_model(location) == adafruit_fingerprint.OK:
        print("Lưu thành công")
        return True
    else:
        print("Lỗi khi lưu")
        return False
