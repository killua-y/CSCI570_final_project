#!/bin/bash
# efficient.sh — Run efficient.py script
# Usage:
#   ./efficient.sh input.txt output.txt    # Process single file
#   ./efficient.sh                         # Process all inputs in SampleTestCases/

# If arguments are provided, use them as before
if [ $# -eq 2 ]; then
    python3 efficient.py "$1" "$2"
    exit 0
fi

# If no arguments, process all input files in SampleTestCases/
INPUT_DIR="SampleTestCases"
OUTPUT_DIR="SampleTestOutput"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process all input*.txt files
for input_file in "$INPUT_DIR"/input*.txt; do
    if [ -f "$input_file" ]; then
        # Extract the filename (e.g., input1.txt)
        filename=$(basename "$input_file")
        # Convert input1.txt -> output1.txt
        output_filename=$(echo "$filename" | sed 's/input/output/')
        output_file="$OUTPUT_DIR/$output_filename"
        
        echo "Processing $input_file -> $output_file"
        python3 efficient.py "$input_file" "$output_file"
    fi
done

echo "Done processing all input files."