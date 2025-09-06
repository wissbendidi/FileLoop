#!/usr/bin/env python3
"""
Demo script showing how to use the quarantine feature.

This script demonstrates the quarantine functionality outside of the Streamlit UI.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils import FileUtils

def demo_quarantine():
    """Demonstrate quarantine functionality."""
    print("🛡️ FileLoop Quarantine Feature Demo")
    print("=" * 50)
    
    # Initialize the file utils
    utils = FileUtils()
    
    # Create some test files
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    test_files = [
        "test1.txt",
        "test2.jpg", 
        "test3.pdf",
        "duplicate1.txt",
        "duplicate2.txt"
    ]
    
    print("📁 Creating test files...")
    for filename in test_files:
        file_path = os.path.join(test_dir, filename)
        with open(file_path, 'w') as f:
            f.write(f"Test content for {filename}")
        print(f"  Created: {file_path}")
    
    print(f"\n🗑️ Moving files to quarantine...")
    
    # Move files to quarantine
    quarantined_files = []
    for filename in test_files:
        file_path = os.path.join(test_dir, filename)
        try:
            # Move to quarantine with subdirectory based on file type
            ext = Path(filename).suffix.lower()
            subdir = "images" if ext in ['.jpg', '.png'] else "documents" if ext in ['.pdf', '.txt'] else "other"
            
            quarantined_path = utils.safe_move_to_quarantine(file_path, subdir)
            quarantined_files.append(quarantined_path)
            print(f"  ✅ Moved {filename} to {quarantined_path}")
        except Exception as e:
            print(f"  ❌ Error moving {filename}: {e}")
    
    print(f"\n📊 Quarantine Status:")
    print(f"  Files quarantined: {len(quarantined_files)}")
    print(f"  Quarantine directory: {utils.quarantine_dir}")
    
    # List quarantined files
    print(f"\n📋 Quarantined Files:")
    for root, dirs, files in os.walk(utils.quarantine_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, utils.quarantine_dir)
            size = os.path.getsize(file_path)
            print(f"  📁 {relative_path} ({size} bytes)")
    
    # Demonstrate restore functionality
    if quarantined_files:
        print(f"\n🔄 Restoring a file...")
        try:
            # Restore the first file
            file_to_restore = quarantined_files[0]
            restored_path = os.path.join("restored", os.path.basename(file_to_restore))
            os.makedirs(os.path.dirname(restored_path), exist_ok=True)
            
            utils.restore_from_quarantine(file_to_restore, restored_path)
            print(f"  ✅ Restored {file_to_restore} to {restored_path}")
        except Exception as e:
            print(f"  ❌ Error restoring file: {e}")
    
    # Generate quarantine report
    print(f"\n📊 Generating quarantine report...")
    try:
        report = utils.create_quarantine_report(quarantined_files)
        print("Quarantine Report:")
        print("-" * 30)
        print(report)
    except Exception as e:
        print(f"  ❌ Error generating report: {e}")
    
    print(f"\n🎯 Quarantine Demo Complete!")
    print(f"  Check the 'quarantine/' directory to see quarantined files")
    print(f"  Check the 'restored/' directory to see restored files")

if __name__ == "__main__":
    demo_quarantine()
