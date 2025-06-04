# fingerprint_app.py

import json
import os
from fingerprint import (
    finger,
    get_fingerprint,
    enroll_finger,
    delete_finger,
    read_templates,
    get_finger_id,
    get_confidence
)

USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def get_num():
    """Use input() to get a valid number from 1 to 127. Retry till success!"""
    i = 0
    while (i > 127) or (i < 1):
        try:
            i = int(input("Enter ID # from 1-127: "))
        except ValueError:
            pass
    return i

def enroll_or_update_finger():
    location = get_num()
    users = load_users()
    
    if str(location) in users:
        print(f"User ID {location} already exists with name: {users[str(location)]}")
        update = input("Do you want to update fingerprint and/or name? (y/n): ")
        if update.lower() != 'y':
            print("Cancelled.")
            return

    # Tiến hành enroll như bình thường
    if enroll_finger(location):
        # Sau enroll thành công, nhập tên mới
        new_name = input("Enter name for user ID {}: ".format(location))
        users[str(location)] = new_name
        save_users(users)
        print("Fingerprint and name stored successfully.")
    else:
        print("Failed to enroll fingerprint.")

def edit_name():
    users = load_users()
    user_id = get_num()
    if str(user_id) not in users:
        print("User ID not found.")
        return
    new_name = input("Enter new name for user ID {}: ".format(user_id))
    users[str(user_id)] = new_name
    save_users(users)
    print(f"Name updated for ID {user_id} -> {new_name}")

def delete_finger_and_name():
    user_id = get_num()
    if delete_finger(user_id) == finger.OK:
        users = load_users()
        user_id_str = str(user_id)
        if user_id_str in users:
            users.pop(user_id_str)
            save_users(users)
        print("Deleted fingerprint and user name for ID:", user_id)
    else:
        print("Failed to delete fingerprint.")

def find_finger():
    if get_fingerprint():
        users = load_users()
        user_name = users.get(str(get_finger_id()), "Unknown User")
        print(f"Detected user: {user_name} (ID: {get_finger_id()}) with confidence {get_confidence()}")
    else:
        print("Finger not found")

def main():
    while True:
        if read_templates() != finger.OK:
            raise RuntimeError("Failed to read templates")
        print("----------------")
        print("Fingerprint templates:", finger.templates)
        print("e) enroll or update fingerprint")
        print("f) find fingerprint")
        print("d) delete fingerprint")
        print("n) edit user name")
        print("q) quit")
        print("----------------")
        c = input("> ")

        if c == "e":
            enroll_or_update_finger()
        elif c == "f":
            find_finger()
        elif c == "d":
            delete_finger_and_name()
        elif c == "n":
            edit_name()
        elif c == "q":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
