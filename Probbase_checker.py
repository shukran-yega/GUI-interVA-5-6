#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converter for InterVA2022 R data files (.rda) to CSV format
Handles both probbase2022 and causetext2022 files
"""

import pyreadr
import pandas as pd
import os
import sys


def convert_rda_to_csv(rda_file_path: str, output_csv_path: str = None) -> str:
    """
    Convert R data file (.rda) to CSV format.
    
    Parameters:
    -----------
    rda_file_path : str
        Path to the input .rda file
    output_csv_path : str, optional
        Path for output CSV file. If not provided, uses same name as input with .csv extension
    
    Returns:
    --------
    str : Path to the created CSV file
    """
    
    if not os.path.exists(rda_file_path):
        raise FileNotFoundError(f"File not found: {rda_file_path}")
    
    # Determine output path
    if output_csv_path is None:
        output_csv_path = os.path.splitext(rda_file_path)[0] + ".csv"
    
    try:
        # Load the R object(s)
        result = pyreadr.read_r(rda_file_path)
        
        # Get the object name and data
        object_names = list(result.keys())
        
        if len(object_names) == 0:
            raise ValueError("No data objects found in the .rda file")
        
        # Extract the data (usually the first/only object)
        object_name = object_names[0]
        df = result[object_name]
        
        print(f"Found object: {object_name}")
        print(f"Shape: {df.shape}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)[:10]}...")
        
        # Save to CSV without modifying data
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Successfully converted {rda_file_path} to {output_csv_path}")
        
        return output_csv_path
        
    except Exception as e:
        print(f"❌ Error converting {rda_file_path}: {e}")
        raise


def validate_probbase(csv_path: str) -> bool:
    """
    Validate that the probbase CSV has the expected structure.
    
    Expected structure:
    - 343 rows (including header)
    - 71 columns
    - Last column should be related to symptom i446o
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check dimensions
        expected_rows = 343
        expected_cols = 71
        
        if df.shape[0] != expected_rows:
            print(f"⚠️ Warning: Expected {expected_rows} rows, found {df.shape[0]}")
        
        if df.shape[1] != expected_cols:
            print(f"⚠️ Warning: Expected {expected_cols} columns, found {df.shape[1]}")
        
        # Check for version info (should be in position [0, 2])
        if df.shape[1] > 2:
            version = df.iloc[0, 2]
            print(f"Probbase version found: {version}")
        
        return True
        
    except Exception as e:
        print(f"Error validating probbase: {e}")
        return False


def validate_causetext(csv_path: str) -> bool:
    """
    Validate that the causetext CSV has the expected structure.
    
    Expected structure:
    - 67 rows (3 pregnancy states + 64 causes)
    - At least 2 columns (code and description)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check dimensions
        expected_rows = 67
        
        if df.shape[0] != expected_rows:
            print(f"⚠️ Warning: Expected {expected_rows} rows, found {df.shape[0]}")
        
        if df.shape[1] < 2:
            print(f"⚠️ Warning: Expected at least 2 columns, found {df.shape[1]}")
        
        print(f"Causetext structure: {df.shape[0]} causes, {df.shape[1]} columns")
        
        return True
        
    except Exception as e:
        print(f"Error validating causetext: {e}")
        return False


def main():
    """
    Main function to convert InterVA2022 data files.
    """
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python probbase_checker.py <path_to_rda_file> [output_csv_path]")
        print("\nExample:")
        print("  python probbase_checker.py probbase2022.rda")
        print("  python probbase_checker.py causetext2022.rda causetext.csv")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert the file
    try:
        output_path = convert_rda_to_csv(input_file, output_file)
        
        # Validate based on file type
        if 'probbase' in os.path.basename(input_file).lower():
            print("\nValidating probbase structure...")
            validate_probbase(output_path)
        elif 'causetext' in os.path.basename(input_file).lower():
            print("\nValidating causetext structure...")
            validate_causetext(output_path)
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)


# Additional utility function for batch conversion
def convert_all_rda_files(directory: str) -> None:
    """
    Convert all .rda files in a directory to CSV format.
    
    Parameters:
    -----------
    directory : str
        Path to directory containing .rda files
    """
    
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    # Find all .rda files
    rda_files = [f for f in os.listdir(directory) if f.endswith('.rda')]
    
    if not rda_files:
        print(f"No .rda files found in {directory}")
        return
    
    print(f"Found {len(rda_files)} .rda files to convert:")
    
    for rda_file in rda_files:
        rda_path = os.path.join(directory, rda_file)
        print(f"\nConverting: {rda_file}")
        
        try:
            output_path = convert_rda_to_csv(rda_path)
            
            # Validate if it's a known file type
            if 'probbase' in rda_file.lower():
                validate_probbase(output_path)
            elif 'causetext' in rda_file.lower():
                validate_causetext(output_path)
            
        except Exception as e:
            print(f"Failed to convert {rda_file}: {e}")
    
    print("\nConversion complete!")


if __name__ == "__main__":
    # If running as a script with specific paths, you can modify these
    if len(sys.argv) == 1:
        # Default behavior - convert known files if they exist
        default_files = [
            r"C:\Users\Shukurani\Desktop\Project X\probbase2022.rda",
            r"C:\Users\Shukurani\Desktop\Project X\causetext2022.rda"
        ]
        
        for file_path in default_files:
            if os.path.exists(file_path):
                print(f"Converting {os.path.basename(file_path)}...")
                try:
                    output = convert_rda_to_csv(file_path)
                    
                    if 'probbase' in file_path.lower():
                        validate_probbase(output)
                    elif 'causetext' in file_path.lower():
                        validate_causetext(output)
                        
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print(f"File not found: {file_path}")
    else:
        main()