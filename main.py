import tkinter as tk
from tkinter import messagebox
import json
from fingerprint import enroll_finger, get_fingerprint
import adafruit_fingerprint

# Đọc danh sách người dùng
def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

users = load_users()

def enroll():
    name = entry_name.get().strip()
    if not name:
        messagebox.showerror("Lỗi", "Vui lòng nhập tên")
        return
    # Tìm ID trống
    used_ids = list(map(int, users.keys()))
    next_id = 1
    while next_id in used_ids:
        next_id += 1
    if enroll_finger(next_id):
        users[str(next_id)] = name
        save_users(users)
        messagebox.showinfo("Thành công", f"Đã lưu '{name}' với ID {next_id}")
    else:
        messagebox.showerror("Thất bại", "Không thể đăng ký")

def verify():
    if get_fingerprint():
        user_id = str(finger.finger_id)
        name = users.get(user_id, "Không xác định")
        confidence = finger.confidence
        messagebox.showinfo("Xác thực", f"Đã nhận diện: {name} (ID {user_id})\nĐộ tin cậy: {confidence}")
    else:
        messagebox.showwarning("Thất bại", "Không tìm thấy vân tay")

# Giao diện
root = tk.Tk()
root.title("Hệ thống Vân tay")

tk.Label(root, text="Tên người dùng:").pack()
entry_name = tk.Entry(root)
entry_name.pack()

tk.Button(root, text="Đăng ký vân tay", command=enroll).pack(pady=5)
tk.Button(root, text="Xác thực vân tay", command=verify).pack(pady=5)

root.mainloop()
