import os
import glob

# Specify the base directory to search in
search_directory = '/robodata/eyang/data'

# Read the input file
with open('haresh_example_config.txt', 'r') as file:
    lines = file.readlines()

# Loop through the lines in the input file
for og_line in lines:
    if not og_line.__contains__('-'):
        continue
    line = og_line.strip()  # Remove leading/trailing whitespace
    #     # Remove the '- ' prefix if present
    #     line = line[2:]

    terrain_label = line.split('/')[-2]
    line = line.split('/')[-1]  # Remove the directory path if present

    # Construct the file name to search for
    file_name = line + '.bag'

    # Search for the file in subdirectories with date names
    found = False
    # Search for the file in subdirectories with date names
    for subdir in os.listdir(search_directory):
        if os.path.isdir(os.path.join(search_directory, subdir)):
            file_path = os.path.join(search_directory, subdir, file_name)
            if os.path.exists(file_path):
                print(f'Found: {file_path}')
                output_dir = ''
                if 'train' in og_line:
                    output_dir = os.path.join('spot_data', 'train', terrain_label, line)
                elif 'test' in og_line:
                    output_dir = os.path.join('spot_data', 'test', terrain_label, line)
                # os.system(f'scripts/spot_extract_data.py -b {file_name} -o {output_dir}')
                print(f'scripts/spot_extract_data.py -b {file_name} -o {output_dir}')
                break
