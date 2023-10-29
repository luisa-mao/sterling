import os
import glob
import sys

# Specify the base directory to search in
search_directory = '/robodata/eyang/data'

# Read the input file
with open('haresh_example_config.txt', 'r') as file:
    lines = file.readlines()

# Function to search for and process a file
def process_file(line):
    terrain_label = line.split('/')[-2]
    bag_name = line.split('/')[-1]  # Remove the directory path if present

    # Construct the file name to search for
    file_name = bag_name + '.bag'

    # Search for the file in subdirectories with date names
    for subdir in os.listdir(search_directory):
        if os.path.isdir(os.path.join(search_directory, subdir)):
            file_path = os.path.join(search_directory, subdir, file_name)
            if os.path.exists(file_path):
                print(f'Found: {file_path}')
                output_dir = ''
                if 'train' in line:
                    output_dir = os.path.join('spot_data', 'train', terrain_label, bag_name)
                elif 'test' in line:
                    output_dir = os.path.join('spot_data', 'test', terrain_label, bag_name)
                command = f'python3 scripts/spot_extract_data.py -b {file_path} -o {output_dir}'
                print(f'Running: {command}')
                os.system(command)
                break

# Loop through the lines in the input file
for og_line in lines:
    if not og_line.__contains__('-'):
        continue
    line = og_line.strip()  # Remove leading/trailing whitespace
    process_file(line)

    # Check for user input to stop the script
    # user_input = input("Press 'q' and Enter to quit or Enter to continue: ")
    # if user_input == 'q':
    #     sys.exit(0)
