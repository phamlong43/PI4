import numpy as np
import os

POSES = ["frontal", "looking left", "looking right", "looking up", "looking down"]

def get_pose_filename(pose):
    return f"face_db_{pose.replace(' ', '_')}.npz"

def load_pose_data(pose):
    filename = get_pose_filename(pose)
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        return list(data["embeddings"]), list(data["labels"])
    return [], []

def save_pose_data(pose, embeddings, labels):
    filename = get_pose_filename(pose)
    np.savez(filename, embeddings=embeddings, labels=labels)

def list_users():
    all_users = set()
    print("=== DANH SÁCH NGƯỜI DÙNG THEO TỪNG GÓC ===")
    for pose in POSES:
        embeddings, labels = load_pose_data(pose)
        unique_labels = set(labels)
        all_users.update(unique_labels)
        print(f"[{pose.upper()}] ({len(unique_labels)} người): {', '.join(unique_labels) if unique_labels else 'Không có'}")
    print("\n===> Tổng số người dùng: ", len(all_users))

def delete_user(name):
    found = False
    for pose in POSES:
        embeddings, labels = load_pose_data(pose)
        new_embeddings = []
        new_labels = []
        for emb, lbl in zip(embeddings, labels):
            if lbl != name:
                new_embeddings.append(emb)
                new_labels.append(lbl)
            else:
                found = True
        save_pose_data(pose, new_embeddings, new_labels)
    if found:
        print(f"[✓] Đã xóa người dùng '{name}' khỏi hệ thống.")
    else:
        print(f"[!] Không tìm thấy người dùng '{name}' trong bất kỳ góc nào.")

def main():
    while True:
        print("\n=== QUẢN LÝ NGƯỜI DÙNG ===")
        print("1. Xem danh sách người dùng")
        print("2. Xóa người dùng")
        print("3. Thoát")

        choice = input("Chọn chức năng (1/2/3): ").strip()

        if choice == '1':
            list_users()
        elif choice == '2':
            name = input("Nhập tên người dùng cần xóa: ").strip()
            if name:
                confirm = input(f"Bạn có chắc chắn muốn xóa '{name}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    delete_user(name)
                else:
                    print("[-] Hủy thao tác.")
            else:
                print("[-] Tên không hợp lệ.")
        elif choice == '3':
            print("Thoát.")
            break
        else:
            print("[-] Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()
