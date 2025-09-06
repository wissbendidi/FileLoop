"""
Tests for the scanner module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.scanner import FileScanner, FileMetadata


class TestFileScanner:
    """Test cases for FileScanner class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.scanner = FileScanner()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        files = self.scanner.scan_directory(self.temp_dir)
        assert len(files) == 0
    
    def test_scan_directory_with_files(self):
        """Test scanning a directory with files."""
        # Create test files
        test_files = ['test1.txt', 'test2.jpg', 'test3.pdf']
        for filename in test_files:
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(f"Test content for {filename}")
        
        files = self.scanner.scan_directory(self.temp_dir)
        assert len(files) == 3
        
        # Check file names
        file_names = [f.name for f in files]
        for test_file in test_files:
            assert test_file in file_names
    
    def test_excluded_directories(self):
        """Test that excluded directories are not scanned."""
        # Create excluded directory
        excluded_dir = os.path.join(self.temp_dir, '.git')
        os.makedirs(excluded_dir)
        
        # Create file in excluded directory
        with open(os.path.join(excluded_dir, 'test.txt'), 'w') as f:
            f.write("This should not be found")
        
        files = self.scanner.scan_directory(self.temp_dir)
        assert len(files) == 0
    
    def test_file_categories(self):
        """Test file categorization."""
        # Create files of different types
        test_files = {
            'image.jpg': 'images',
            'video.mp4': 'videos',
            'document.pdf': 'documents',
            'script.py': 'code',
            'music.mp3': 'audio',
            'archive.zip': 'archives'
        }
        
        for filename, expected_category in test_files.items():
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write("Test content")
        
        files = self.scanner.scan_directory(self.temp_dir)
        categories = self.scanner.get_file_categories(files)
        
        # Check that each expected category has files
        for filename, expected_category in test_files.items():
            assert expected_category in categories
            assert len(categories[expected_category]) > 0
            
            # Check that the file is in the correct category
            file_found = False
            for file_metadata in categories[expected_category]:
                if file_metadata.name == filename:
                    file_found = True
                    break
            assert file_found, f"File {filename} not found in category {expected_category}"
    
    def test_nonexistent_directory(self):
        """Test scanning a non-existent directory."""
        with pytest.raises(FileNotFoundError):
            self.scanner.scan_directory("/nonexistent/directory")
    
    def test_file_not_directory(self):
        """Test scanning a file instead of directory."""
        # Create a file
        file_path = os.path.join(self.temp_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write("Test content")
        
        with pytest.raises(ValueError):
            self.scanner.scan_directory(file_path)


if __name__ == "__main__":
    pytest.main([__file__])
