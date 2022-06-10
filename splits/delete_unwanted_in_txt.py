
file_path = "./eigen_zhou/val_files.txt"
target_path = "./eigen_zhou_test/val_files.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    with open(target_path, "w", encoding='utf-8') as f:
        for line in lines:
            if line.startswith("2011_09_30/2011_09_30_drive_0020_sync"):
                f.write(line)

