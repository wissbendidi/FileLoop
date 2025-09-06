"""
File classification and clustering module.

This module handles clustering and classifying files using scikit-learn
algorithms based on their embeddings and metadata.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Data class to store clustering results."""
    labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    cluster_centers: Optional[np.ndarray] = None


class FileClassifier:
    """Class for clustering and classifying files based on embeddings and metadata."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the file classifier.
        
        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def cluster_files(self, 
                     embeddings: np.ndarray, 
                     file_metadata: List[Dict[str, Any]], 
                     method: str = 'kmeans',
                     n_clusters: Optional[int] = None,
                     eps: float = 0.5,
                     min_samples: int = 5) -> ClusterResult:
        """
        Cluster files based on their embeddings and metadata.
        
        Args:
            embeddings: Array of file embeddings
            file_metadata: List of file metadata dictionaries
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters for KMeans (auto-determined if None)
            eps: Epsilon parameter for DBSCAN
            min_samples: Minimum samples parameter for DBSCAN
            
        Returns:
            ClusterResult object with clustering information
        """
        if method == 'kmeans':
            return self._kmeans_clustering(embeddings, file_metadata, n_clusters)
        elif method == 'dbscan':
            return self._dbscan_clustering(embeddings, file_metadata, eps, min_samples)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _kmeans_clustering(self, 
                          embeddings: np.ndarray, 
                          file_metadata: List[Dict[str, Any]], 
                          n_clusters: Optional[int] = None) -> ClusterResult:
        """Perform KMeans clustering."""
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(embeddings)
        
        logger.info(f"Performing KMeans clustering with {n_clusters} clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(embeddings, labels)
        else:
            silhouette_avg = 0.0
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            silhouette_score=silhouette_avg,
            cluster_centers=kmeans.cluster_centers_
        )
    
    def _dbscan_clustering(self, 
                          embeddings: np.ndarray, 
                          file_metadata: List[Dict[str, Any]], 
                          eps: float, 
                          min_samples: int) -> ClusterResult:
        """Perform DBSCAN clustering."""
        logger.info(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)
        
        # Calculate number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate silhouette score (only if we have more than one cluster)
        if n_clusters > 1:
            # Remove noise points for silhouette calculation
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
            else:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            silhouette_score=silhouette_avg
        )
    
    def _determine_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """
        Determine optimal number of clusters using silhouette analysis.
        
        Args:
            embeddings: Array of embeddings
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        if len(embeddings) < 2:
            return 1
        
        max_clusters = min(max_clusters, len(embeddings) - 1)
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, labels)
            silhouette_scores.append(silhouette_avg)
        
        # Return the number of clusters with the highest silhouette score
        optimal_clusters = np.argmax(silhouette_scores) + 2
        logger.info(f"Optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters
    
    def classify_by_usage_patterns(self, file_metadata: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Classify files by usage patterns (size, age, access frequency).
        
        Args:
            file_metadata: List of file metadata dictionaries
            
        Returns:
            Dictionary mapping pattern names to lists of file indices
        """
        patterns = {
            'large_files': [],
            'old_files': [],
            'recent_files': [],
            'small_files': [],
            'medium_files': []
        }
        
        current_time = pd.Timestamp.now().timestamp()
        
        for i, file_info in enumerate(file_metadata):
            size = file_info.get('size', 0)
            last_modified = file_info.get('last_modified', current_time)
            
            # Size-based classification
            size_mb = size / (1024 * 1024)
            if size_mb > 100:  # > 100MB
                patterns['large_files'].append(i)
            elif size_mb < 1:  # < 1MB
                patterns['small_files'].append(i)
            else:
                patterns['medium_files'].append(i)
            
            # Age-based classification
            days_old = (current_time - last_modified) / (24 * 3600)
            if days_old > 365:  # > 1 year
                patterns['old_files'].append(i)
            elif days_old < 30:  # < 30 days
                patterns['recent_files'].append(i)
        
        return patterns
    
    def get_cluster_summary(self, 
                           cluster_result: ClusterResult, 
                           file_metadata: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Get a summary of each cluster including file counts and characteristics.
        
        Args:
            cluster_result: Result from clustering
            file_metadata: List of file metadata dictionaries
            
        Returns:
            Dictionary mapping cluster IDs to summary information
        """
        summary = {}
        
        for cluster_id in range(cluster_result.n_clusters):
            cluster_mask = cluster_result.labels == cluster_id
            cluster_files = [file_metadata[i] for i in range(len(file_metadata)) if cluster_mask[i]]
            
            if not cluster_files:
                continue
            
            # Calculate cluster statistics
            total_size = sum(f.get('size', 0) for f in cluster_files)
            extensions = [f.get('extension', '') for f in cluster_files]
            extension_counts = pd.Series(extensions).value_counts()
            
            summary[cluster_id] = {
                'file_count': len(cluster_files),
                'total_size': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'most_common_extension': extension_counts.index[0] if len(extension_counts) > 0 else '',
                'extension_diversity': len(extension_counts),
                'files': cluster_files
            }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Example usage
    classifier = FileClassifier()
    
    # Sample data
    sample_embeddings = np.random.rand(10, 384)  # 384-dim embeddings
    sample_metadata = [
        {'name': f'file_{i}.txt', 'size': i * 1000, 'extension': '.txt', 'last_modified': 1000000000 + i}
        for i in range(10)
    ]
    
    try:
        # Perform clustering
        result = classifier.cluster_files(sample_embeddings, sample_metadata, method='kmeans')
        print(f"Clustering result: {result.n_clusters} clusters, silhouette score: {result.silhouette_score:.3f}")
        
        # Get cluster summary
        summary = classifier.get_cluster_summary(result, sample_metadata)
        for cluster_id, info in summary.items():
            print(f"Cluster {cluster_id}: {info['file_count']} files, {info['total_size_mb']:.1f} MB")
        
        # Classify by usage patterns
        patterns = classifier.classify_by_usage_patterns(sample_metadata)
        for pattern, indices in patterns.items():
            print(f"{pattern}: {len(indices)} files")
            
    except Exception as e:
        print(f"Error in classification: {e}")
