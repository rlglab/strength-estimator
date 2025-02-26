import os
import random


input_folder = "after_filtered_"
query_folder = "split_files/query"
cand_folder = "split_files/cand"

os.makedirs(query_folder, exist_ok=True)
os.makedirs(cand_folder, exist_ok=True)

# 處理每個檔案
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # 確保是檔案
    if not os.path.isfile(file_path):
        continue

    # 讀取檔案內容並打亂順序
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    random.shuffle(lines)  # 打亂順序

    # 檢查行數是否足夠
    if len(lines) < 1220:
        print(f"檔案 {filename} 行數不足 1000 行，跳過...")
        continue

    # 分割成兩部分
    query_lines = lines[:1100]
    cand_lines = lines[1100:1220]

    # 保存到 query 資料夾
    query_output_path = os.path.join(query_folder, filename)
    with open(query_output_path, "w", encoding="utf-8") as query_file:
        query_file.writelines(query_lines)

    # 保存到 cand 資料夾
    cand_output_path = os.path.join(cand_folder, filename)
    with open(cand_output_path, "w", encoding="utf-8") as cand_file:
        cand_file.writelines(cand_lines)

    print(f"檔案 {filename} 已成功分割到 query 和 cand 資料夾。")
