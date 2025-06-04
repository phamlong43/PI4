import tkinter as tk
from tkinter import ttk
import fingerprints as fp

class FingerprintApp:
    def __init__(self, root):
        self.root = root
        root.title("Quản lý và xác thực vân tay")

        frame_inputs = ttk.Frame(root, padding=10)
        frame_inputs.pack()

        ttk.Label(frame_inputs, text="Tên:").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(frame_inputs, width=25)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        frame_buttons = ttk.Frame(root, padding=10)
        frame_buttons.pack()

        self.enroll_btn = ttk.Button(frame_buttons, text="Thêm vân tay", command=self.enroll)
        self.enroll_btn.grid(row=0, column=0, padx=5, pady=5)

        self.edit_btn = ttk.Button(frame_buttons, text="Sửa tên", command=self.edit_name)
        self.edit_btn.grid(row=0, column=1, padx=5, pady=5)

        self.delete_btn = ttk.Button(frame_buttons, text="Xóa vân tay", command=self.delete)
        self.delete_btn.grid(row=0, column=2, padx=5, pady=5)

        self.auth_btn = ttk.Button(root, text="Xác thực vân tay", command=self.authenticate)
        self.auth_btn.pack(pady=10, fill='x')

        self.status_label = ttk.Label(root, text="Sẵn sàng", font=("Arial", 14))
        self.status_label.pack(pady=5)

        self.log_text = tk.Text(root, height=10, width=60, state='disabled', bg="#f0f0f0")
        self.log_text.pack(pady=5)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def get_name(self):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.config(text="Lỗi: Vui lòng nhập tên", foreground="red")
            return None
        return name

    def enroll(self):
        name = self.get_name()
        if name is None:
            return
        self.status_label.config(text="Đang đăng ký vân tay...", foreground="blue")
        self.root.update()

        # Tự động cấp ID mới, giả sử hàm enroll_finger trả về True nếu thành công
        success = fp.enroll_finger_auto(name)
        if success:
            self.status_label.config(text=f"Đăng ký thành công cho {name}", foreground="green")
            self.log(f"Đã đăng ký vân tay cho {name}")
        else:
            self.status_label.config(text="Đăng ký thất bại hoặc tên đã tồn tại", foreground="red")
            self.log(f"Lỗi khi đăng ký vân tay cho {name}")

    def edit_name(self):
        old_name = self.get_name()
        if old_name is None:
            return
        # Yêu cầu nhập tên mới trong 1 popup
        new_name = tk.simpledialog.askstring("Sửa tên", "Nhập tên mới:")
        if not new_name:
            self.status_label.config(text="Hủy sửa tên", foreground="orange")
            return
        success = fp.update_username_by_name(old_name, new_name)
        if success:
            self.status_label.config(text=f"Đã đổi tên {old_name} thành {new_name}", foreground="green")
            self.log(f"Đổi tên {old_name} → {new_name}")
        else:
            self.status_label.config(text=f"Không tìm thấy tên {old_name}", foreground="red")
            self.log(f"Lỗi: Không tìm thấy tên {old_name}")

    def delete(self):
        name = self.get_name()
        if name is None:
            return
        confirm = tk.simpledialog.askstring("Xác nhận xóa", f"Nhập 'YES' để xóa vân tay của {name}:")
        if confirm != "YES":
            self.status_label.config(text="Hủy xóa", foreground="orange")
            return
        success = fp.delete_finger_by_name(name)
        if success:
            self.status_label.config(text=f"Đã xóa vân tay của {name}", foreground="green")
            self.log(f"Xóa vân tay {name} thành công")
        else:
            self.status_label.config(text="Xóa thất bại hoặc không tìm thấy tên", foreground="red")
            self.log(f"Lỗi khi xóa vân tay {name}")

    def authenticate(self):
        self.status_label.config(text="Vui lòng đặt ngón tay lên cảm biến...", foreground="blue")
        self.root.update()

        success, username = fp.authenticate_finger()
        if success:
            self.status_label.config(text=f"Xác thực thành công - Chào mừng {username}", foreground="green")
            self.log(f"Xác thực thành công: {username}")
        else:
            self.status_label.config(text="Xác thực thất bại", foreground="red")
            self.log("Xác thực thất bại")

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()
