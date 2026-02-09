#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert bead design CSV from 1D to 2D format
"""

import csv
import os

def convert_csv_to_2d(input_csv_path, output_csv_path):
    """
    Convert 1D bead design CSV to 2D format
    
    Args:
        input_csv_path: Path to the input CSV file (1D format)
        output_csv_path: Path to save the output CSV file (2D format)
    """
    # Read the input CSV file
    data = []
    max_x = 0
    max_y = 0
    
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x = int(row['x'])
            y = int(row['y'])
            color = row['color']
            
            data.append((x, y, color))
            
            # Update max coordinates
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
    
    print(f"Found {len(data)} data points")
    print(f"Maximum x: {max_x}, Maximum y: {max_y}")
    
    # Create a 2D grid filled with TRANSPARENT
    grid = [['TRANSPARENT' for _ in range(max_x)] for _ in range(max_y)]
    
    # Fill the grid with colors
    for x, y, color in data:
        # Convert to 0-based indices
        grid_x = x - 1
        grid_y = y - 1
        
        # Make sure indices are within bounds
        if 0 <= grid_x < max_x and 0 <= grid_y < max_y:
            grid[grid_y][grid_x] = color
    
    print(f"Created 2D grid with size: {max_y} rows x {max_x} columns")
    
    # Write the 2D grid to output CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row (optional)
        # writer.writerow([''] + [str(x) for x in range(1, max_x + 1)])
        
        # Write grid rows
        for row_idx, row in enumerate(grid):
            # Write row with optional row index (y-coordinate)
            # writer.writerow([row_idx + 1] + row)
            writer.writerow(row)
    
    print(f"2D CSV file saved to: {output_csv_path}")
    
    return max_x, max_y

if __name__ == "__main__":
    # Get the current directory (where this script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the csv directory path
    csv_dir = os.path.join(current_dir, "csv")
    
    # Create csv directory if it doesn't exist
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        print(f"Created csv directory: {csv_dir}")
    
    print(f"\n=== CSV 2D Conversion Tool ===")
    print(f"Processing all CSV files in: {csv_dir}")
    
    # Get all CSV files in the csv directory
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and not f.endswith('_2D.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        print("Please place your 1D CSV files in this directory and run the script again.")
    else:
        print(f"\nFound {len(csv_files)} CSV file(s) to convert:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        
        # Convert each CSV file
        for file in csv_files:
            print(f"\n{'='*60}")
            print(f"Converting file: {file}")
            
            # Build input and output paths
            input_csv = os.path.join(csv_dir, file)
            filename_without_ext = os.path.splitext(file)[0]
            output_csv = os.path.join(csv_dir, f"{filename_without_ext}_2D.csv")
            
            print(f"Input file: {input_csv}")
            print(f"Output file: {output_csv}")
            
            try:
                max_x, max_y = convert_csv_to_2d(input_csv, output_csv)
                print(f"\n✓ Conversion completed successfully!")
                print(f"  2D grid size: {max_y} rows x {max_x} columns")
            except Exception as e:
                print(f"\n✗ Conversion failed with error: {e}")
                print(f"  Skipping file: {file}")
        
        print(f"\n{'='*60}")
        print(f"All conversions completed!")
        print(f"2D CSV files are saved in: {csv_dir}")