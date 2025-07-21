#!/usr/bin/env python3
"""
Script to clean text files by removing lines starting with '#' and '='
"""

import os
import glob
from pathlib import Path


def clean_file(file_path):
    """
    Clean a single file by removing lines starting with '#' or '='
    
    Args:
        file_path (str): Path to the file to clean
    
    Returns:
        tuple: (original_line_count, cleaned_line_count, lines_removed)
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_count = len(lines)
        
        # Filter out lines starting with '#' or '='
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            if not (stripped_line.startswith('#') or stripped_line.startswith('=')):
                cleaned_lines.append(line)
        
        cleaned_count = len(cleaned_lines)
        lines_removed = original_count - cleaned_count
        
        # Only write back if there were changes
        if lines_removed > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            
            print(f"✓ {file_path}: Removed {lines_removed} lines ({original_count} → {cleaned_count})")
        else:
            print(f"• {file_path}: No changes needed")
        
        return original_count, cleaned_count, lines_removed
    
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return 0, 0, 0


def find_text_files(root_dir="."):
    """
    Find all .txt files in the directory tree
    
    Args:
        root_dir (str): Root directory to search from
    
    Returns:
        list: List of .txt file paths
    """
    txt_files = []
    
    # Use glob to find all .txt files recursively
    pattern = os.path.join(root_dir, "**", "*.txt")
    txt_files = glob.glob(pattern, recursive=True)
    
    return sorted(txt_files)


def main():
    """
    Main function to clean all text files
    """
    print("🧹 Cleaning text files - removing lines starting with '#' and '='")
    print("=" * 60)
    
    # Find all text files
    txt_files = find_text_files()
    
    if not txt_files:
        print("No .txt files found in the current directory tree.")
        return
    
    print(f"Found {len(txt_files)} text files to process:")
    print()
    
    # Process each file
    total_original = 0
    total_cleaned = 0
    total_removed = 0
    files_modified = 0
    
    for file_path in txt_files:
        original, cleaned, removed = clean_file(file_path)
        total_original += original
        total_cleaned += cleaned
        total_removed += removed
        
        if removed > 0:
            files_modified += 1
    
    # Summary
    print()
    print("=" * 60)
    print("📊 SUMMARY:")
    print(f"Files processed: {len(txt_files)}")
    print(f"Files modified: {files_modified}")
    print(f"Total lines removed: {total_removed}")
    print(f"Total lines: {total_original} → {total_cleaned}")
    
    if files_modified > 0:
        print(f"\n✅ Successfully cleaned {files_modified} files!")
    else:
        print(f"\n✅ All files were already clean!")


if __name__ == "__main__":
    main() 