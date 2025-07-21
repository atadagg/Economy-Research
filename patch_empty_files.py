import os

OLD3_DIR = "older/3"

for root, dirs, files in os.walk(OLD3_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.getsize(file_path) == 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(" ")
            print(f"Patched: {file_path}")