#!/bin/bash
# process_directory.sh — Run both basic.py and efficient.py script
# Usage:
#   ./process_directory.sh input_dir output_basic_dir output_efficient_dir    # Process directory

# If arguments are provided, use them as before
if [ $# -eq 3 ]; then
    input_dir="$1"
    output_basic_dir="$2"
    output_efficient_dir="$3"
    # Create output directory if it doesn't exist
    mkdir -p "$output_basic_dir"
    mkdir -p "$output_efficient_dir"
    # Process all *in*.txt files
    for input_file in "$input_dir"/*in*.txt; do
        if [ -f "$input_file" ]; then
            # Extract the filename (e.g., input1.txt)
            filename=$(basename "$input_file")
            # Convert input1.txt -> output1.txt or in1.txt -> out1.txt
            output_filename=$(echo "$filename" | sed 's/in/out/')
            output_basic_file="$output_basic_dir/$output_filename"
            output_efficient_file="$output_efficient_dir/$output_filename"
            echo "Processing $input_file -> $output_basic_file"
            python3 basic.py "$input_file" "$output_basic_file"
            echo "Processing $input_file -> $output_efficient_file"
            python3 efficient.py "$input_file" "$output_efficient_file"
        fi
    done
    exit 0
fi
