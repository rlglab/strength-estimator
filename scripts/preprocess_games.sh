#!/bin/bash

cp ./scripts/data.py download_chess_game/
cp ./scripts/board.py download_chess_game/
cd download_chess_game/

for year in 2024 2023; do
    for month in 01 02 09 10 11 12; do
        if [ "$year" = "2024" ] || ([ "$year" = "2023" ] && [ "$month" -ge 9 ]); then
            mkdir -p "database${year}/${year}${month}/"
            python3 data.py $year $month -u > "database${year}/${year}${month}/${year}-${month}-convert.txt"
        fi
    done
done
cd ../

mkdir -p training_sgf
for year in 2024 2023; do
    for month in 01 02 09 10 11 12; do
        if [ "$year" = "2024" ] || ([ "$year" = "2023" ] && [ "$month" -ge 9 ]); then
            mv "download_chess_game/database${year}/${year}${month}/${year}-${month}-convert.txt" training_sgf/
        fi
    done
done


cp ./scripts/sgf_filter_random_sample.py ./
python3 sgf_filter_random_sample.py

cp ./scripts/random_sample.py ./
python3 random_sample.py




declare -A folder_map
folder_map["rank_50000_1000_2600_200interval/train"]="training_sgf_chess"
folder_map["rank_50000_1000_2600_200interval/test"]="query_sgf_chess"
folder_map["rank_50000_1000_2600_200interval/cand"]="candidate_sgf_chess"


for source_dir in "${!folder_map[@]}"; do
    target_dir="${folder_map[$source_dir]}"  

    
    mkdir -p "$target_dir"

    echo "Processing $source_dir -> $target_dir"

    for folder in "$source_dir"/sgf_*; do
        if [[ -d "$folder" ]]; then  
            base_name=$(basename "$folder")
            new_name="${base_name#sgf_}.txt"
            cat "$folder"/*.txt > "$target_dir/$new_name"

            echo "Merged $folder/*.txt -> $target_dir/$new_name"
        fi
    done
done

echo "All files have been merged and moved to their respective directories."
