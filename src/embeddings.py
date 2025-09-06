"""
Embedding generation module using local SentenceTransformers models.

This module handles generating embeddings for files using local AI models,
ensuring complete privacy as no data is sent to external services.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Class for generating embeddings using local SentenceTransformers models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the SentenceTransformers model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_file_embeddings(self, file_metadata: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for a list of files based on their metadata.
        
        Args:
            file_metadata: List of file metadata dictionaries
            
        Returns:
            Numpy array of embeddings with shape (n_files, embedding_dim)
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Create text representations for each file
        file_texts = []
        for file_info in file_metadata:
            text_representation = self._create_file_text_representation(file_info)
            file_texts.append(text_representation)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(file_texts)} files")
        embeddings = self.model.encode(file_texts, show_progress_bar=True)
        
        return embeddings
    
    def _create_file_text_representation(self, file_info: Dict[str, Any]) -> str:
        """
        Create a text representation of a file for embedding generation.
        
        Args:
            file_info: Dictionary containing file metadata
            
        Returns:
            Text representation of the file
        """
        # Extract basic information
        name = file_info.get('name', '')
        extension = file_info.get('extension', '')
        size = file_info.get('size', 0)
        parent_dir = file_info.get('parent_dir', '')
        
        # Create a descriptive text representation
        text_parts = []
        
        # Add file name and extension
        if name:
            text_parts.append(f"file name: {name}")
        
        if extension:
            text_parts.append(f"file type: {extension}")
        
        # Add size information
        if size > 0:
            size_mb = size / (1024 * 1024)
            if size_mb > 1000:
                size_gb = size_mb / 1024
                text_parts.append(f"large file: {size_gb:.1f} GB")
            elif size_mb > 10:
                text_parts.append(f"medium file: {size_mb:.1f} MB")
            else:
                text_parts.append(f"small file: {size_mb:.1f} MB")
        
        # Add directory context
        if parent_dir:
            # Extract meaningful parts of the path
            path_parts = Path(parent_dir).parts
            if len(path_parts) > 1:
                # Include last few directory parts for context
                relevant_parts = path_parts[-3:]
                text_parts.append(f"location: {'/'.join(relevant_parts)}")
        
        # Add file category based on extension
        category = self._get_file_category(extension)
        if category:
            text_parts.append(f"category: {category}")
        
        return " ".join(text_parts)
    
    def _get_file_category(self, extension: str) -> Optional[str]:
        """
        Get a human-readable category for a file extension.
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            Category name or None if unknown
        """
        ext = extension.lower().lstrip('.')
        
        categories = {
            'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'svg', 'webp'],
            'videos': ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'm4v'],
            'documents': ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt', 'pages'],
            'code': ['py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'h', 'php', 'rb', 'go', 'rs'],
            'audio': ['mp3', 'wav', 'flac', 'aac', 'ogg', 'wma', 'm4a'],
            'archives': ['zip', 'rar', '7z', 'tar', 'gz', 'bz2', 'xz'],
            'data': ['csv', 'json', 'xml', 'yaml', 'yml', 'sql', 'db'],
            'presentations': ['ppt', 'pptx', 'odp', 'key'],
            'spreadsheets': ['xls', 'xlsx', 'ods', 'numbers']
        }
        
        for category, extensions in categories.items():
            if ext in extensions:
                return category
        
        return None
    
    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        return self.model.encode(texts, show_progress_bar=True)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        return self.model.get_sentence_embedding_dimension()


# Example usage
if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator()
    
    # Sample file metadata
    sample_files = [
        {
            'name': 'vacation_photos.jpg',
            'extension': '.jpg',
            'size': 2048000,  # 2MB
            'parent_dir': '/home/user/Pictures/Vacation'
        },
        {
            'name': 'project_report.pdf',
            'extension': '.pdf',
            'size': 1024000,  # 1MB
            'parent_dir': '/home/user/Documents/Work'
        }
    ]
    
    try:
        embeddings = generator.generate_file_embeddings(sample_files)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
