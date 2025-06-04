import tkinter as tk
from tkinter import messagebox
import json
import fingerprint as fp

DATA_FILE = "finger_names.json"

def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

class FingerprintApp:
    def __init__(self, root):
        self.root = root
        root.title("Fingerprint Auth System")

        self.data = load_data()

        tk.Label(root, text="Name:").grid(row=0, column=0)
        self.name_var = tk.StringVar()
        tk.Entry(root, textvariable=self.name_var).grid(row=0, column=1)

        self.log_text = tk.Text(root, width=50, height=15)
        self.log_text.grid(row=3, column=0, columnspan=3)

        tk.Button(root, text="Add / Enroll", command=self.enroll).grid(row=1, column=0)
        tk.Button(root, text="Delete", command=self.delete).grid(row=1, column=1)
        tk.Button(root, text="Authenticate", command=self.authenticate).grid(row=1, column=2)

        self.status_label = tk.Label(root, text="Ready")
        self.status_label.grid(row=2, column=0, columnspan=3)

        fp.read_templates()

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def find_next_id(self):
        used_ids = set(int(k) for k in self.data.keys())
        for i in range(1, 128):
            if i not in used_ids:
                return i
        return None

    def enroll(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Input error", "Please enter a name.")
            return

        # Nếu tên đã có, hỏi có muốn ghi đè (sửa)
        existing_id = None
        for k, v in self.data.items():
            if v == name:
                existing_id = int(k)
                break

        if existing_id:
            if not messagebox.askyesno("Confirm", f"Name '{name}' exists. Do you want to overwrite?"):
                return
            # Xóa mẫu cũ
            if fp.delete_finger(existing_id):
                self.log(f"Deleted old fingerprint ID {existing_id} for {name}")
            else:
                self.log(f"Failed to delete old fingerprint ID {existing_id}")

            use_id = existing_id
        else:
            use_id = self.find_next_id()
            if not use_id:
                messagebox.showerror("Error", "No available fingerprint ID slots!")
                return

        self.status_label.config(text=f"Please place finger twice to enroll '{name}' (ID {use_id})", fg="blue")
        self.root.update()

        success = fp.enroll_finger(use_id)
        if success:
            self.data[str(use_id)] = name
            save_data(self.data)
            self.status_label.config(text=f"Enrollment successful for '{name}' (ID {use_id})", fg="green")
            self.log(f"Enrolled fingerprint for '{name}' with ID {use_id}")
        else:
            self.status_label.config(text="Enrollment failed", fg="red")
            self.log("Enrollment failed")

    def delete(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Input error", "Please enter a name.")
            return

        delete_id = None
        for k, v in self.data.items():
            if v == name:
                delete_id = int(k)
                break

        if not delete_id:
            messagebox.showinfo("Info", f"No fingerprint found for '{name}'")
            return

        if messagebox.askyesno("Confirm", f"Are you sure to delete fingerprint for '{name}'?"):
            if fp.delete_finger(delete_id):
                self.log(f"Deleted fingerprint ID {delete_id} for '{name}'")
                del self.data[str(delete_id)]
                save_data(self.data)
                self.status_label.config(text=f"Deleted fingerprint for '{name}'", fg="green")
            else:
                self.status_label.config(text=f"Failed to delete fingerprint for '{name}'", fg="red")
                self.log(f"Failed to delete fingerprint ID {delete_id}")

    def authenticate(self):
        self.status_label.config(text="Place finger for authentication...", fg="blue")
        self.root.update()
        result = fp.search_finger()
        if result:
            fid, confidence = result
            name = self.data.get(str(fid), "Unknown")
            self.status_label.config(text=f"Fingerprint matched: {name} (ID {fid}), confidence {confidence}", fg="green")
            self.log(f"Authenticated: {name} (ID {fid}) confidence={confidence}")
        else:
            self.status_label.config(text="No match found", fg="red")
            self.log("Authentication failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()
