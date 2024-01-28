#!/bin/bash

# This script is used to run the program.

# To run this script, open a terminal and navigate to the directory where the script is located.
# Then, execute the following command:
# ./script.sh

# Make sure the script has execute permissions. If not, run the following command:
# chmod +x script.sh

# The script will then be executed and the program will run accordingly.
#!/bin/bash

loop_num=$1
save_dir=$2
data_config=$3

for ((i=0; i<$loop_num; i++))
do
    python3 scripts/train_vrlpap_baseline.py --data_config_path "$data_config" --b 128 --save_dir "$save_dir"  --epochs 200
done

