import os
import random
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

min_elo = 1000
max_elo = 2600
interval = 200

# sample numbers of line in each file
lines_per_file = 50000

input_dir = "training_sgf"


def getRank(elo):
    return (elo - min_elo) // interval


def process_file(file_name):
    filtered_lines = []
    for i in range(min_elo, max_elo, interval):
        output_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/train/sgf_{i}_{i + interval}"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/test_origin/sgf_{i}_{i + interval}"
        os.makedirs(output_dir, exist_ok=True)
        filtered_lines.append([])
    input_file = os.path.join(input_dir, file_name)
    print(f"------start {file_name} sample {min_elo}~{max_elo}------")
    with open(input_file, "r", encoding="utf-8") as f_in:

        for line in tqdm(f_in):
            wr_match = re.search(r"WR\[(\d+)\]", line)
            br_match = re.search(r"BR\[(\d+)\]", line)
            if wr_match and br_match:
                wr_rating = int(wr_match.group(1))
                br_rating = int(br_match.group(1))
                if wr_rating >= min_elo and wr_rating < max_elo:
                    if getRank(wr_rating) == getRank(br_rating):

                        filtered_lines[getRank(wr_rating)].append(line)
    for i in range(min_elo, max_elo, interval):
        print(
            f"{file_name} sample {i}~{i+interval}: {len(filtered_lines[getRank(i)])} lines"
        )
        random_lines = random.sample(
            filtered_lines[getRank(i)],
            min(lines_per_file, len(filtered_lines[getRank(i)])),
        )
        sep = int(len(random_lines) * 0.8)
        output_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/train/sgf_{i}_{i + interval}"
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(random_lines[:sep])
        output_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/test_origin/sgf_{i}_{i + interval}"
        output_file = os.path.join(output_dir, file_name)

        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(random_lines[sep:])
    print(f"------finish {file_name} sample {min_elo}~{max_elo}------")


with ThreadPoolExecutor() as executor:
    executor.map(process_file, os.listdir(input_dir))
print("complete!")
