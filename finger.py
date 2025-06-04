# fingerprint_functions.py
import json
import os

DATA_FILE = "fingerprint_data.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def enroll_finger_auto(name):
    """
    Thêm vân tay với tên, tự cấp ID (int tăng dần).
    Trả về True nếu thành công, False nếu tên đã tồn tại.
    """
    data = load_data()
    # Kiểm tra trùng tên
    for id_, info in data.items():
        if info['name'] == name:
            return False  # Tên đã tồn tại

    # Tự cấp ID
    if data:
        new_id = str(int(max(data.keys(), key=int)) + 1)
    else:
        new_id = "1"

    # TODO: Thêm code gọi phần cứng để enroll vân tay ở đây
    # Giả sử enroll thành công:

    data[new_id] = {"name": name}
    save_data(data)
    return True

def update_username_by_name(old_name, new_name):
    """
    Sửa tên theo tên cũ.
    Trả về True nếu thành công, False nếu không tìm thấy hoặc tên mới đã tồn tại.
    """
    data = load_data()

    # Kiểm tra tên mới có trùng không
    for info in data.values():
        if info['name'] == new_name:
            return False  # Tên mới đã tồn tại

    for id_, info in data.items():
        if info['name'] == old_name:
            data[id_]['name'] = new_name
            save_data(data)
            return True
    return False

def delete_finger_by_name(name):
    """
    Xóa vân tay theo tên.
    Trả về True nếu thành công, False nếu không tìm thấy.
    """
    data = load_data()
    for id_, info in list(data.items()):
        if info['name'] == name:
            # TODO: Xóa vân tay phần cứng nếu cần
            del data[id_]
            save_data(data)
            return True
    return False

def authenticate_finger():
    """
    Giả lập xác thực vân tay.
    Trả về (True, username) nếu thành công, (False, None) nếu thất bại.
    Thực tế bạn phải gọi phần cứng để lấy ID vân tay, rồi tra tên.
    """
    # TODO: Gọi phần cứng lấy id vân tay
    # Giả lập: hỏi người dùng nhập ID (chỉ dùng để test)
    id_input = input("Nhập ID vân tay để giả lập xác thực: ").strip()
    data = load_data()
    if id_input in data:
        return True, data[id_input]['name']
    else:
        return False, None
