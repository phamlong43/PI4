import tkinter as tk
from tkinter import messagebox, simpledialog
import fingerprint as fp

class FingerprintApp:
    def __init__(self, root):
        self.root = root
        root.title("Quản lý và xác thực vân tay")

        # Status
        self.status_label = tk.Label(root, text="Sẵn sàng", font=("Arial", 14))
        self.status_label.pack(pady=10)

        # Buttons frame
        frame = tk.Frame(root)
        frame.pack(pady=10)

        # Thêm vân tay
        self.enroll_btn = tk.Button(frame, text="Thêm vân tay", width=15, command=self.enroll)
        self.enroll_btn.grid(row=0, column=0, padx=5, pady=5)

        # Sửa tên
        self.edit_btn = tk.Button(frame, text="Sửa tên", width=15, command=self.edit_name)
        self.edit_btn.grid(row=0, column=1, padx=5, pady=5)

        # Xóa vân tay
        self.delete_btn = tk.Button(frame, text="Xóa vân tay", width=15, command=self.delete)
        self.delete_btn.grid(row=0, column=2, padx=5, pady=5)

        # Xác thực
        self.auth_btn = tk.Button(root, text="Xác thực vân tay", width=50, command=self.authenticate)
        self.auth_btn.pack(pady=10)

        # Hiển thị tên người dùng khi xác thực
        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)

    def enroll(self):
        try:
            location = simpledialog.askinteger("Nhập ID", "Nhập ID từ 1 đến 127:")
            if location is None:
                return
            if not (1 <= location <= 127):
                messagebox.showerror("Lỗi", "ID phải từ 1 đến 127")
                return
            username = simpledialog.askstring("Nhập tên", "Nhập tên người dùng:")
            if not username:
                messagebox.showerror("Lỗi", "Tên không được để trống")
                return
            self.status_label.config(text="Đang đăng ký vân tay...")
            self.root.update()
            success = fp.enroll_finger(location, username)
            if success:
                messagebox.showinfo("Thành công", f"Đã đăng ký vân tay cho {username}")
                self.status_label.config(text="Đăng ký thành công")
            else:
                messagebox.showerror("Lỗi", "Đăng ký vân tay thất bại")
                self.status_label.config(text="Đăng ký thất bại")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            self.status_label.config(text="Lỗi xảy ra")

    def edit_name(self):
        try:
            location = simpledialog.askinteger("Nhập ID", "Nhập ID cần sửa tên:")
            if location is None:
                return
            username = simpledialog.askstring("Nhập tên mới", "Nhập tên mới:")
            if not username:
                messagebox.showerror("Lỗi", "Tên không được để trống")
                return
            success = fp.update_username(location, username)
            if success:
                messagebox.showinfo("Thành công", f"Đã cập nhật tên ID {location}")
            else:
                messagebox.showerror("Lỗi", f"Không tìm thấy ID {location}")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def delete(self):
        try:
            location = simpledialog.askinteger("Nhập ID", "Nhập ID cần xóa:")
            if location is None:
                return
            if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa vân tay ID {location}?"):
                success = fp.delete_finger(location)
                if success:
                    messagebox.showinfo("Thành công", f"Đã xóa ID {location}")
                else:
                    messagebox.showerror("Lỗi", "Xóa thất bại")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def authenticate(self):
        self.status_label.config(text="Vui lòng đặt ngón tay lên cảm biến...")
        self.result_label.config(text="")
        self.root.update()

        success, username = fp.authenticate_finger()
        if success:
            self.status_label.config(text="Xác thực thành công")
            self.result_label.config(text=f"Chào mừng, {username}")
        else:
            self.status_label.config(text="Xác thực thất bại")
            self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()
