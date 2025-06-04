import tkinter as tk
from tkinter import messagebox
import json
from fingerprint import enroll_finger, get_fingerprint, finger

# Quản lý file user (tên <-> id vân tay)
def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

users = load_users()  # {"Tên người dùng": id mẫu vân tay}

def find_free_id():
    used_ids = list(users.values())
    for i in range(1, 128):
        if i not in used_ids:
            return i
    return None

def enroll():
    name = entry_name.get().strip()
    if not name:
        messagebox.showerror("Lỗi", "Vui lòng nhập tên")
        return

    if name in users:
        messagebox.showwarning("Cảnh báo", f"Tên '{name}' đã tồn tại!")
        return

    new_id = find_free_id()
    if new_id is None:
        messagebox.showerror("Lỗi", "Bộ nhớ vân tay đã đầy!")
        return

    if enroll_finger(new_id):
        users[name] = new_id
        save_users(users)
        messagebox.showinfo("Thành công", f"Đã lưu vân tay cho {name}")
        entry_name.delete(0, tk.END)
    else:
        messagebox.showerror("Thất bại", "Không thể đăng ký vân tay")

def verify():
    if get_fingerprint():
        uid = finger.finger_id
        name = next((k for k, v in users.items() if v == uid), None)
        if not name:
            name = "Không xác định"
        conf = finger.confidence
        messagebox.showinfo("Xác thực", f"Đã nhận diện: {name}\nĐộ tin cậy: {conf}")
    else:
        messagebox.showwarning("Thất bại", "Không tìm thấy vân tay")

root = tk.Tk()
root.title("Hệ thống Vân tay")

tk.Label(root, text="Nhập tên người dùng:").pack(pady=5)
entry_name = tk.Entry(root, width=30)
entry_name.pack()

tk.Button(root, text="Đăng ký vân tay", command=enroll, width=25).pack(pady=10)
tk.Button(root, text="Xác thực vân tay", command=verify, width=25).pack()

root.mainloop()
