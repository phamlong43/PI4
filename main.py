import tkinter as tk
from tkinter import messagebox, scrolledtext
import fingerprint_app as fapp
import adafruit_fingerprint

class FingerprintGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Fingerprint Auth System")

        self.db = fapp.load_db()

        # Label và Entry nhập tên
        tk.Label(master, text="Name:").grid(row=0, column=0)
        self.name_entry = tk.Entry(master)
        self.name_entry.grid(row=0, column=1)

        # Nút chức năng
        tk.Button(master, text="Add", command=self.add_user).grid(row=1, column=0)
        tk.Button(master, text="Edit", command=self.edit_user).grid(row=1, column=1)
        tk.Button(master, text="Delete", command=self.delete_user).grid(row=1, column=2)
        tk.Button(master, text="Authenticate", command=self.authenticate_user).grid(row=1, column=3)

        # Vùng log hiển thị trạng thái
        self.log = scrolledtext.ScrolledText(master, width=60, height=20)
        self.log.grid(row=2, column=0, columnspan=4)

        self.log_insert("System initialized.")

    def log_insert(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.master.update()

    def add_user(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Name cannot be empty")
            return
        if name in self.db.values():
            self.log_insert(f"Name '{name}' already exists.")
            return
    
        used_ids = set(map(int, self.db.keys()))
        for i in range(1, 128):
            if i not in used_ids:
                free_id = i
                break
        else:
            self.log_insert("No free ID slot available.")
            return
    
        self.log_insert(f"Register fingerprint for {name} at ID {free_id}")
    
        self.log_insert("Please place your finger to check if fingerprint exists...")
        fid = fapp.search_finger()
        if fid is not None:
            existing_name = self.db.get(str(fid), None)
            self.log_insert(f"Fingerprint already exists for '{existing_name}'. Registration aborted.")
            return
    
        self.log_insert("Place finger 3 times to register.")
        success, msg = fapp.enroll_finger(free_id)
        if success:
            self.db[str(free_id)] = name
            fapp.save_db(self.db)
            self.log_insert(f"User '{name}' registered successfully with ID {free_id}.")
        else:
            self.log_insert(f"Enrollment failed: {msg}")

    def edit_user(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Name cannot be empty")
            return
        # Tìm ID theo tên
        found_id = None
        for k, v in self.db.items():
            if v == name:
                found_id = int(k)
                break
        if not found_id:
            self.log_insert(f"User '{name}' not found.")
            return
        self.log_insert(f"Editing fingerprint for '{name}' with ID {found_id}")
        # Xóa mẫu cũ
        if fapp.delete_finger(found_id) != adafruit_fingerprint.OK:
            self.log_insert("Failed to delete old fingerprint template.")
            return
        # Enroll lại
        self.log_insert("Place finger 3 times to re-register.")
        success = fapp.enroll_finger(found_id)
        if success:
            self.log_insert(f"Fingerprint for '{name}' updated successfully.")
        else:
            self.log_insert("Failed to update fingerprint.")

    def delete_user(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Name cannot be empty")
            return
        found_id = None
        for k, v in self.db.items():
            if v == name:
                found_id = int(k)
                break
        if not found_id:
            self.log_insert(f"User '{name}' not found.")
            return
        if fapp.delete_finger(found_id) == adafruit_fingerprint.OK:
            del self.db[str(found_id)]
            fapp.save_db(self.db)
            self.log_insert(f"User '{name}' deleted successfully.")
        else:
            self.log_insert("Failed to delete fingerprint template.")

    def authenticate_user(self):
        self.log_insert("Please place your finger to authenticate...")
        fid, confidence = fapp.search_finger()
        if fid is None:
            # Nếu muốn, có thể phân biệt thêm để không báo lỗi ngay
            self.log_insert("No finger detected yet.")
            return
        name = self.db.get(str(fid), None)
        if name:
            self.log_insert(f"Authenticated: {name} (Confidence: {confidence})")
        else:
            self.log_insert(f"Fingerprint ID {fid} found but no associated name.")


def main():
    root = tk.Tk()
    app = FingerprintGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
