import os
import random


def process_file(in_file, cand, test):
    num_lines = cand + test
    with open(in_file, "r") as file:
        lines = file.readlines()
        selected_lines = random.sample(lines, num_lines)
        cand_lines = selected_lines[:cand]
        test_lines = selected_lines[cand:]
    target_dir = os.path.dirname(in_file).replace("test_origin", "cand")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(in_file.replace("test_origin", "cand"), "w") as file:
        file.writelines(cand_lines)
    target_dir = os.path.dirname(in_file).replace("test_origin", "test")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(in_file.replace("test_origin", "test"), "w") as file:
        file.writelines(test_lines)
    print(f"{in_file} finished")


def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path + "/test_origin"):
        for file in files:
            if file.endswith(".txt"):
                folder_name = os.path.basename(root)
                if folder_name.startswith("sgf_"):
                    parts = folder_name.split("_")
                    if len(parts) == 3:
                        try:
                            x = int(parts[1])
                            if x < 1000 or x >= 2600:
                                process_file(os.path.join(root, file), 5, 50)
                            else:
                                process_file(os.path.join(root, file), 20, 200)
                        except ValueError:
                            pass


def main():
    folder_path = "rank_50000_1000_2600_200interval"
    process_folder(folder_path)


if __name__ == "__main__":
    main()
