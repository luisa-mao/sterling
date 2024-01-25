#!/bin/bash

# Function to move contents of inner directory one level up and delete inner directory
move_and_delete_inner() {
    local dir="$1"

    # Iterate through subdirectories of the given directory
    for sub_dir in "$dir"/*; do
        if [ -d "$sub_dir" ]; then
            # Get the inner directory name
            inner_dir=$(basename "$sub_dir")

            # Check if the inner directory exists
            if [ -d "$sub_dir/$inner_dir" ]; then
                # Move the contents of the inner directory one level up
                mv "$sub_dir/$inner_dir"/* "$sub_dir/"

                # Remove the inner directory
                rm -r "$sub_dir/$inner_dir"
            fi
        fi
    done
}

# Iterate through all directories in the current directory
for dir in */; do
    if [ -d "$dir" ]; then
        # Call the function to move and delete inner directories
        move_and_delete_inner "$dir"
    fi
done
