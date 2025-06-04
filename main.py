import tkinter as tk
from tkinter import simpledialog
import fingerprint_functions as fp

class FingerprintApp:
    def __init__(self, root):
        self.root = root
        root.title("Quản lý và xác thực vân tay")

        # Label trạng thái / thông báo
        self.status_label = tk.Label(root, text="Sẵn sàng", font=("Arial", 14), fg="blue")
        self.status_label.pack(pady=10)

        # Frame chứa các nút
        frame = tk.Frame(root)
        frame.pack(pady=10)

        self.enroll_btn = tk.Button(frame, text="Thêm vân tay", width=15, command=self.enroll)
        self.enroll_btn.grid(row=0, column=0, padx=5, pady=5)

        self.edit_btn = tk.Button(frame, text="Sửa tên", width=15, command=self.edit_name)
        self.edit_btn.grid(row=0, column=1, padx=5, pady=5)

        self.delete_btn = tk.Button(frame, text="Xóa vân tay", width=15, command=self.delete)
        self.delete_btn.grid(row=0, column=2, padx=5, pady=5)

        self.auth_btn = tk.Button(root, text="Xác thực vân tay", width=50, command=self.authenticate)
        self.auth_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), fg="green")
        self.result_label.pack(pady=10)

        # Text widget để hiển thị log chi tiết, có thể cuộn
        self.log_text = tk.Text(root, height=10, width=60, state='disabled', bg="#f0f0f0")
        self.log_text.pack(pady=5)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Tự cuộn xuống cuối
        self.log_text.config(state='disabled')

    def enroll(self):
        location = simpledialog.askinteger("Nhập ID", "Nhập ID từ 1 đến 127:")
        if location is None:
            self.status_label.config(text="Hủy đăng ký vân tay")
            return
        if not (1 <= location <= 127):
            self.status_label.config(text="Lỗi: ID phải từ 1 đến 127", fg="red")
            return

        username = simpledialog.askstring("Nhập tên", "Nhập tên người dùng:")
        if not username:
            self.status_label.config(text="Lỗi: Tên không được để trống", fg="red")
            return

        self.status_label.config(text="Đang đăng ký vân tay...", fg="blue")
        self.result_label.config(text="")
        self.root.update()

        success = fp.enroll_finger(location, username)
        if success:
            self.status_label.config(text="Đăng ký thành công", fg="green")
            self.log(f"Đã đăng ký vân tay ID {location} cho {username}")
        else:
            self.status_label.config(text="Đăng ký thất bại", fg="red")
            self.log(f"Lỗi khi đăng ký vân tay cho ID {location}")

    def edit_name(self):
        location = simpledialog.askinteger("Nhập ID", "Nhập ID cần sửa tên:")
        if location is None:
            self.status_label.config(text="Hủy sửa tên")
            return

        username = simpledialog.askstring("Nhập tên mới", "Nhập tên mới:")
        if not username:
            self.status_label.config(text="Lỗi: Tên không được để trống", fg="red")
            return

        success = fp.update_username(location, username)
        if success:
            self.status_label.config(text=f"Đã cập nhật tên ID {location}", fg="green")
            self.log(f"Cập nhật tên thành công: ID {location} → {username}")
        else:
            self.status_label.config(text=f"Không tìm thấy ID {location}", fg="red")
            self.log(f"Lỗi: Không tìm thấy ID {location} để cập nhật")

    def delete(self):
        location = simpledialog.askinteger("Nhập ID", "Nhập ID cần xóa:")
        if location is None:
            self.status_label.config(text="Hủy xóa vân tay")
            return

        # Thay vì messagebox xác nhận, ta xác nhận bằng dialog đơn giản (nếu cần, có thể tạo nút Confirm trên GUI)
        confirm = simpledialog.askstring("Xác nhận xóa", f"Nhập 'YES' để xóa ID {location}:")
        if confirm != "YES":
            self.status_label.config(text="Hủy xóa vân tay")
            return

        success = fp.delete_finger(location)
        if success:
            self.status_label.config(text=f"Đã xóa ID {location}", fg="green")
            self.log(f"Xóa vân tay ID {location} thành công")
        else:
            self.status_label.config(text="Xóa thất bại", fg="red")
            self.log(f"Lỗi khi xóa ID {location}")

    def authenticate(self):
        self.status_label.config(text="Vui lòng đặt ngón tay lên cảm biến...", fg="blue")
        self.result_label.config(text="")
        self.root.update()

        success, username = fp.authenticate_finger()
        if success:
            self.status_label.config(text="Xác thực thành công", fg="green")
            self.result_label.config(text=f"Chào mừng, {username}", fg="green")
            self.log(f"Xác thực thành công: {username}")
        else:
            self.status_label.config(text="Xác thực thất bại", fg="red")
            self.result_label.config(text="", fg="red")
            self.log("Xác thực thất bại")

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()
