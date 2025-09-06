"""
Visualization module for creating interactive charts and plots.

This module provides functions for creating treemap, sunburst, and other
visualizations using Plotly for the Streamlit interface.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    color_scheme: str = "viridis"
    width: int = 800
    height: int = 600
    show_legend: bool = True


class FileVisualizer:
    """Class for creating file visualizations."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the file visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
    
    def create_treemap(self, 
                      file_metadata: List[Dict[str, Any]], 
                      cluster_labels: Optional[np.ndarray] = None,
                      title: str = "File Organization Treemap") -> go.Figure:
        """
        Create a treemap visualization of files.
        
        Args:
            file_metadata: List of file metadata dictionaries
            cluster_labels: Optional cluster labels for coloring
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Prepare data for treemap
        df = self._prepare_treemap_data(file_metadata, cluster_labels)
        
        # Create treemap
        fig = px.treemap(
            df,
            path=['category', 'cluster', 'name'],
            values='size',
            color='size',
            color_continuous_scale=self.config.color_scheme,
            title=title,
            width=self.config.width,
            height=self.config.height
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            font_size=12,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_sunburst(self, 
                       file_metadata: List[Dict[str, Any]], 
                       cluster_labels: Optional[np.ndarray] = None,
                       title: str = "File Organization Sunburst") -> go.Figure:
        """
        Create a sunburst visualization of files.
        
        Args:
            file_metadata: List of file metadata dictionaries
            cluster_labels: Optional cluster labels for coloring
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Prepare data for sunburst
        df = self._prepare_sunburst_data(file_metadata, cluster_labels)
        
        # Create sunburst
        fig = px.sunburst(
            df,
            path=['category', 'cluster', 'name'],
            values='size',
            color='size',
            color_continuous_scale=self.config.color_scheme,
            title=title,
            width=self.config.width,
            height=self.config.height
        )
        
        # Update layout
        fig.update_layout(
            title_x=0.5,
            font_size=12,
            showlegend=self.config.show_legend
        )
        
        return fig
    
    def create_size_distribution_chart(self, 
                                     file_metadata: List[Dict[str, Any]],
                                     title: str = "File Size Distribution") -> go.Figure:
        """
        Create a histogram of file size distribution.
        
        Args:
            file_metadata: List of file metadata dictionaries
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        sizes = [f.get('size', 0) for f in file_metadata]
        sizes_mb = [s / (1024 * 1024) for s in sizes]
        
        fig = px.histogram(
            x=sizes_mb,
            nbins=50,
            title=title,
            labels={'x': 'File Size (MB)', 'y': 'Count'},
            width=self.config.width,
            height=self.config.height
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="File Size (MB)",
            yaxis_title="Number of Files"
        )
        
        return fig
    
    def create_category_pie_chart(self, 
                                 file_metadata: List[Dict[str, Any]],
                                 title: str = "Files by Category") -> go.Figure:
        """
        Create a pie chart showing file distribution by category.
        
        Args:
            file_metadata: List of file metadata dictionaries
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Categorize files
        categories = self._categorize_files(file_metadata)
        
        # Create pie chart data
        category_counts = {cat: len(files) for cat, files in categories.items() if files}
        
        fig = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title=title,
            width=self.config.width,
            height=self.config.height
        )
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    def create_cluster_scatter_plot(self, 
                                  embeddings: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  file_metadata: List[Dict[str, Any]],
                                  title: str = "File Clusters (2D Projection)") -> go.Figure:
        """
        Create a 2D scatter plot of file clusters using t-SNE or PCA.
        
        Args:
            embeddings: File embeddings
            cluster_labels: Cluster labels
            file_metadata: List of file metadata dictionaries
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Reduce dimensionality to 2D
        if embeddings.shape[1] > 2:
            if embeddings.shape[0] > 50:
                # Use PCA for large datasets
                reducer = PCA(n_components=2, random_state=42)
            else:
                # Use t-SNE for smaller datasets
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            
            coords_2d = reducer.fit_transform(embeddings)
        else:
            coords_2d = embeddings
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': cluster_labels,
            'name': [f.get('name', '') for f in file_metadata],
            'size': [f.get('size', 0) for f in file_metadata]
        })
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            size='size',
            hover_data=['name', 'size'],
            title=title,
            width=self.config.width,
            height=self.config.height
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2"
        )
        
        return fig
    
    def create_usage_patterns_chart(self, 
                                  file_metadata: List[Dict[str, Any]],
                                  title: str = "File Usage Patterns") -> go.Figure:
        """
        Create a chart showing file usage patterns (size vs age).
        
        Args:
            file_metadata: List of file metadata dictionaries
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        import time
        
        current_time = time.time()
        
        # Prepare data
        data = []
        for file_info in file_metadata:
            size_mb = file_info.get('size', 0) / (1024 * 1024)
            days_old = (current_time - file_info.get('last_modified', current_time)) / (24 * 3600)
            
            data.append({
                'size_mb': size_mb,
                'days_old': days_old,
                'name': file_info.get('name', ''),
                'extension': file_info.get('extension', ''),
                'category': self._get_file_category(file_info.get('extension', ''))
            })
        
        df = pd.DataFrame(data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='days_old',
            y='size_mb',
            color='category',
            hover_data=['name', 'extension'],
            title=title,
            width=self.config.width,
            height=self.config.height,
            log_y=True  # Log scale for size
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title="Days Since Last Modified",
            yaxis_title="File Size (MB, log scale)"
        )
        
        return fig
    
    def _prepare_treemap_data(self, 
                            file_metadata: List[Dict[str, Any]], 
                            cluster_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prepare data for treemap visualization."""
        data = []
        
        for i, file_info in enumerate(file_metadata):
            category = self._get_file_category(file_info.get('extension', ''))
            cluster = f"Cluster {cluster_labels[i]}" if cluster_labels is not None else "No Cluster"
            
            data.append({
                'name': file_info.get('name', ''),
                'size': file_info.get('size', 0),
                'category': category,
                'cluster': cluster,
                'extension': file_info.get('extension', '')
            })
        
        return pd.DataFrame(data)
    
    def _prepare_sunburst_data(self, 
                             file_metadata: List[Dict[str, Any]], 
                             cluster_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Prepare data for sunburst visualization."""
        return self._prepare_treemap_data(file_metadata, cluster_labels)
    
    def _categorize_files(self, file_metadata: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize files by their extensions."""
        categories = {
            'images': [],
            'videos': [],
            'documents': [],
            'code': [],
            'audio': [],
            'archives': [],
            'other': []
        }
        
        for file_info in file_metadata:
            category = self._get_file_category(file_info.get('extension', ''))
            if category in categories:
                categories[category].append(file_info)
            else:
                categories['other'].append(file_info)
        
        return categories
    
    def _get_file_category(self, extension: str) -> str:
        """Get file category based on extension."""
        ext = extension.lower().lstrip('.')
        
        categories = {
            'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'svg', 'webp'],
            'videos': ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'm4v'],
            'documents': ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt', 'pages'],
            'code': ['py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'h', 'php', 'rb', 'go', 'rs'],
            'audio': ['mp3', 'wav', 'flac', 'aac', 'ogg', 'wma', 'm4a'],
            'archives': ['zip', 'rar', '7z', 'tar', 'gz', 'bz2', 'xz']
        }
        
        for category, extensions in categories.items():
            if ext in extensions:
                return category
        
        return 'other'


# Example usage
if __name__ == "__main__":
    # Example usage
    visualizer = FileVisualizer()
    
    # Sample data
    sample_files = [
        {'name': 'photo1.jpg', 'size': 2048000, 'extension': '.jpg', 'last_modified': 1000000000},
        {'name': 'document.pdf', 'size': 1024000, 'extension': '.pdf', 'last_modified': 1000000000},
        {'name': 'video.mp4', 'size': 50000000, 'extension': '.mp4', 'last_modified': 1000000000}
    ]
    
    try:
        # Create treemap
        treemap_fig = visualizer.create_treemap(sample_files)
        print("Treemap created successfully")
        
        # Create pie chart
        pie_fig = visualizer.create_category_pie_chart(sample_files)
        print("Pie chart created successfully")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
