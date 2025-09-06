"""
File scanner module for collecting file metadata.

This module handles scanning directories and collecting comprehensive
metadata about files including path, size, timestamps, and basic properties.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FileMetadata:
    """Data class to store file metadata."""
    path: str
    name: str
    extension: str
    size: int
    last_accessed: float
    last_modified: float
    is_directory: bool
    parent_dir: str


class FileScanner:
    """Scanner class for collecting file metadata from directories."""
    
    def __init__(self, excluded_dirs: Optional[List[str]] = None):
        """
        Initialize the file scanner.
        
        Args:
            excluded_dirs: List of directory names to exclude from scanning
        """
        self.excluded_dirs = excluded_dirs or ['.git', '__pycache__', 'node_modules', '.venv']
    
    def scan_directory(self, directory_path: str) -> List[FileMetadata]:
        """
        Scan a directory and collect metadata for all files.
        
        Args:
            directory_path: Path to the directory to scan
            
        Returns:
            List of FileMetadata objects for all files found
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
            PermissionError: If access to the directory is denied
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        file_metadata = []
        directory_path = Path(directory_path).resolve()
        
        for root, dirs, files in os.walk(directory_path):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            # Process files in current directory
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    metadata = self._get_file_metadata(file_path)
                    file_metadata.append(metadata)
                except (OSError, PermissionError) as e:
                    # Skip files that can't be accessed
                    print(f"Warning: Could not access {file_path}: {e}")
                    continue
        
        return file_metadata
    
    def _get_file_metadata(self, file_path: str) -> FileMetadata:
        """
        Get metadata for a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileMetadata object with file information
        """
        path_obj = Path(file_path)
        stat_info = path_obj.stat()
        
        return FileMetadata(
            path=str(path_obj.absolute()),
            name=path_obj.name,
            extension=path_obj.suffix.lower(),
            size=stat_info.st_size,
            last_accessed=stat_info.st_atime,
            last_modified=stat_info.st_mtime,
            is_directory=path_obj.is_dir(),
            parent_dir=str(path_obj.parent)
        )
    
    def get_file_categories(self, files: List[FileMetadata]) -> Dict[str, List[FileMetadata]]:
        """
        Categorize files by their extensions.
        
        Args:
            files: List of FileMetadata objects
            
        Returns:
            Dictionary mapping category names to lists of files
        """
        categories = {
            'images': [],
            'videos': [],
            'documents': [],
            'code': [],
            'audio': [],
            'archives': [],
            'other': []
        }
        
        # Extension mappings
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
        doc_extensions = {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.pages'}
        code_extensions = {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.php', '.rb', '.go', '.rs'}
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
        archive_extensions = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'}
        
        for file in files:
            ext = file.extension
            if ext in image_extensions:
                categories['images'].append(file)
            elif ext in video_extensions:
                categories['videos'].append(file)
            elif ext in doc_extensions:
                categories['documents'].append(file)
            elif ext in code_extensions:
                categories['code'].append(file)
            elif ext in audio_extensions:
                categories['audio'].append(file)
            elif ext in archive_extensions:
                categories['archives'].append(file)
            else:
                categories['other'].append(file)
        
        return categories


# Example usage
if __name__ == "__main__":
    scanner = FileScanner()
    
    # Example: Scan current directory
    try:
        files = scanner.scan_directory(".")
        print(f"Found {len(files)} files")
        
        # Show file categories
        categories = scanner.get_file_categories(files)
        for category, file_list in categories.items():
            if file_list:
                print(f"{category.capitalize()}: {len(file_list)} files")
                
    except Exception as e:
        print(f"Error scanning directory: {e}")
