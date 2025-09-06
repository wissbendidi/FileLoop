"""
Utility functions for file operations, hashing, and duplicate detection.

This module provides helper functions for various file operations,
including duplicate detection, file hashing, and safe file operations.
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUtils:
    """Utility class for file operations and duplicate detection."""
    
    def __init__(self, quarantine_dir: str = "quarantine"):
        """
        Initialize the file utilities.
        
        Args:
            quarantine_dir: Directory name for quarantined files
        """
        self.quarantine_dir = quarantine_dir
        self._ensure_quarantine_dir()
    
    def _ensure_quarantine_dir(self) -> None:
        """Ensure the quarantine directory exists."""
        if not os.path.exists(self.quarantine_dir):
            os.makedirs(self.quarantine_dir)
            logger.info(f"Created quarantine directory: {self.quarantine_dir}")
    
    def calculate_file_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm ('sha256', 'md5', 'sha1')
            
        Returns:
            Hexadecimal hash string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {file_path}")
        
        return hash_obj.hexdigest()
    
    def find_duplicate_files(self, file_metadata: List[Dict[str, Any]], 
                           algorithm: str = 'sha256') -> Dict[str, List[Dict[str, Any]]]:
        """
        Find duplicate files based on their hash values.
        
        Args:
            file_metadata: List of file metadata dictionaries
            algorithm: Hash algorithm to use
            
        Returns:
            Dictionary mapping hash values to lists of duplicate files
        """
        hash_to_files = {}
        duplicates = {}
        
        logger.info(f"Calculating {algorithm} hashes for {len(file_metadata)} files")
        
        for i, file_info in enumerate(file_metadata):
            file_path = file_info.get('path', '')
            
            try:
                file_hash = self.calculate_file_hash(file_path, algorithm)
                
                if file_hash in hash_to_files:
                    # Found a duplicate
                    if file_hash not in duplicates:
                        duplicates[file_hash] = [hash_to_files[file_hash]]
                    duplicates[file_hash].append(file_info)
                else:
                    hash_to_files[file_hash] = file_info
                    
            except (FileNotFoundError, PermissionError) as e:
                logger.warning(f"Could not hash file {file_path}: {e}")
                continue
        
        # Filter out single files (keep only actual duplicates)
        duplicates = {h: files for h, files in duplicates.items() if len(files) > 1}
        
        logger.info(f"Found {len(duplicates)} groups of duplicate files")
        return duplicates
    
    def safe_move_to_quarantine(self, file_path: str, 
                              quarantine_subdir: Optional[str] = None) -> str:
        """
        Safely move a file to the quarantine directory.
        
        Args:
            file_path: Path to the file to move
            quarantine_subdir: Optional subdirectory within quarantine
            
        Returns:
            Path where the file was moved
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If file can't be moved
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create quarantine subdirectory if specified
        if quarantine_subdir:
            quarantine_path = os.path.join(self.quarantine_dir, quarantine_subdir)
        else:
            quarantine_path = self.quarantine_dir
        
        os.makedirs(quarantine_path, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        source_path = Path(file_path)
        target_filename = source_path.name
        target_path = os.path.join(quarantine_path, target_filename)
        
        # If file already exists, add timestamp
        counter = 1
        while os.path.exists(target_path):
            name, ext = os.path.splitext(target_filename)
            target_filename = f"{name}_{counter}{ext}"
            target_path = os.path.join(quarantine_path, target_filename)
            counter += 1
        
        try:
            shutil.move(file_path, target_path)
            logger.info(f"Moved {file_path} to {target_path}")
            return target_path
        except PermissionError:
            raise PermissionError(f"Permission denied moving file: {file_path}")
    
    def restore_from_quarantine(self, quarantined_path: str, 
                              target_path: str) -> str:
        """
        Restore a file from quarantine to its original location.
        
        Args:
            quarantined_path: Path to the quarantined file
            target_path: Target path to restore the file
            
        Returns:
            Path where the file was restored
            
        Raises:
            FileNotFoundError: If quarantined file doesn't exist
            PermissionError: If file can't be moved
        """
        if not os.path.exists(quarantined_path):
            raise FileNotFoundError(f"Quarantined file not found: {quarantined_path}")
        
        # Create target directory if it doesn't exist
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            shutil.move(quarantined_path, target_path)
            logger.info(f"Restored {quarantined_path} to {target_path}")
            return target_path
        except PermissionError:
            raise PermissionError(f"Permission denied restoring file: {quarantined_path}")
    
    def get_file_size_category(self, size_bytes: int) -> str:
        """
        Categorize file size into human-readable categories.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Size category string
        """
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb < 1:
            return "tiny"
        elif size_mb < 10:
            return "small"
        elif size_mb < 100:
            return "medium"
        elif size_mb < 1000:
            return "large"
        else:
            return "huge"
    
    def format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def get_file_age_category(self, last_modified: float) -> str:
        """
        Categorize file age based on last modified timestamp.
        
        Args:
            last_modified: Last modified timestamp
            
        Returns:
            Age category string
        """
        current_time = datetime.now().timestamp()
        days_old = (current_time - last_modified) / (24 * 3600)
        
        if days_old < 7:
            return "recent"
        elif days_old < 30:
            return "monthly"
        elif days_old < 365:
            return "yearly"
        else:
            return "ancient"
    
    def create_quarantine_report(self, quarantined_files: List[str]) -> str:
        """
        Create a report of quarantined files.
        
        Args:
            quarantined_files: List of quarantined file paths
            
        Returns:
            Report content as string
        """
        report_lines = [
            f"Quarantine Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            f"Total files quarantined: {len(quarantined_files)}",
            "",
            "Files:"
        ]
        
        for file_path in quarantined_files:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                size = self.format_file_size(stat.st_size)
                modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                report_lines.append(f"  {file_path} ({size}, modified: {modified})")
            else:
                report_lines.append(f"  {file_path} (file not found)")
        
        return "\n".join(report_lines)


# Example usage
if __name__ == "__main__":
    utils = FileUtils()
    
    # Example: Calculate file hash
    try:
        # Create a test file
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("This is a test file for hashing.")
        
        file_hash = utils.calculate_file_hash(test_file)
        print(f"File hash (SHA256): {file_hash}")
        
        # Example: Move to quarantine
        quarantined_path = utils.safe_move_to_quarantine(test_file, "test_quarantine")
        print(f"File moved to: {quarantined_path}")
        
        # Clean up
        if os.path.exists(quarantined_path):
            os.remove(quarantined_path)
            os.rmdir(os.path.dirname(quarantined_path))
        
    except Exception as e:
        print(f"Error: {e}")
