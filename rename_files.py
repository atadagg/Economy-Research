#!/usr/bin/env python3
"""
Script to remove "_economic" from the end of filenames
"""

import os
import glob
from pathlib import Path


def rename_file(file_path):
    """
    Rename a single file by removing "_economic" from the filename
    
    Args:
        file_path (str): Path to the file to rename
    
    Returns:
        tuple: (success, old_name, new_name)
    """
    try:
        # Get the directory and filename
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Check if filename contains "_economic"
        if "_economic" not in filename:
            return False, filename, filename  # No change needed
        
        # Remove "_economic" from the filename
        new_filename = filename.replace("_economic", "")
        
        # Create the new file path
        new_file_path = os.path.join(directory, new_filename)
        
        # Check if the new filename already exists
        if os.path.exists(new_file_path):
            print(f"⚠️  Warning: {new_file_path} already exists, skipping {file_path}")
            return False, filename, new_filename
        
        # Rename the file
        os.rename(file_path, new_file_path)
        
        print(f"✓ Renamed: {file_path} → {new_file_path}")
        return True, filename, new_filename
    
    except Exception as e:
        print(f"✗ Error renaming {file_path}: {e}")
        return False, filename, filename


def find_economic_files(root_dir="."):
    """
    Find all files containing "_economic" in their filename
    
    Args:
        root_dir (str): Root directory to search from
    
    Returns:
        list: List of file paths containing "_economic"
    """
    economic_files = []
    
    # Use glob to find all files with "_economic" in the name recursively
    pattern = os.path.join(root_dir, "**", "*_economic*")
    economic_files = glob.glob(pattern, recursive=True)
    
    # Filter to only include files (not directories)
    economic_files = [f for f in economic_files if os.path.isfile(f)]
    
    return sorted(economic_files)


def main():
    """
    Main function to rename all files containing "_economic"
    """
    print("📝 Renaming files - removing '_economic' from filenames")
    print("=" * 60)
    
    # Find all files with "_economic" in the name
    economic_files = find_economic_files()
    
    if not economic_files:
        print("No files containing '_economic' found in the current directory tree.")
        return
    
    print(f"Found {len(economic_files)} files to rename:")
    print()
    
    # Process each file
    renamed_count = 0
    skipped_count = 0
    
    for file_path in economic_files:
        success, old_name, new_name = rename_file(file_path)
        
        if success:
            renamed_count += 1
        else:
            if old_name != new_name:  # Was supposed to be renamed but failed
                skipped_count += 1
    
    # Summary
    print()
    print("=" * 60)
    print("📊 SUMMARY:")
    print(f"Files found: {len(economic_files)}")
    print(f"Files renamed: {renamed_count}")
    print(f"Files skipped: {skipped_count}")
    
    if renamed_count > 0:
        print(f"\n✅ Successfully renamed {renamed_count} files!")
    elif len(economic_files) == 0:
        print(f"\n✅ No files with '_economic' found!")
    else:
        print(f"\n⚠️  No files were renamed (possibly due to conflicts)!")


if __name__ == "__main__":
    main() 