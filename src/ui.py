"""
Main Streamlit UI for the FileLoop application.

This module provides the interactive web interface for file scanning,
analysis, and cleanup operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import os
from typing import List, Dict, Any, Optional

# Import our modules
from scanner import FileScanner, FileMetadata
from embeddings import EmbeddingGenerator
from classifier import FileClassifier
from visualization import FileVisualizer
from utils import FileUtils

# Page configuration
st.set_page_config(
    page_title="FileLoop - AI File Cleaner",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🗂️ FileLoop</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Local File Cleaner")
    st.markdown("---")
    
    # Initialize session state
    if 'file_metadata' not in st.session_state:
        st.session_state.file_metadata = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'cluster_result' not in st.session_state:
        st.session_state.cluster_result = None
    if 'duplicates' not in st.session_state:
        st.session_state.duplicates = {}
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Directory selection
        st.subheader("📁 Select Directory")
        directory_path = st.text_input(
            "Enter directory path:",
            value=os.getcwd(),
            help="Enter the full path to the directory you want to scan"
        )
        
        # Scan options
        st.subheader("⚙️ Scan Options")
        excluded_dirs = st.multiselect(
            "Exclude directories:",
            options=['.git', '__pycache__', 'node_modules', '.venv', '.env'],
            default=['.git', '__pycache__', 'node_modules', '.venv']
        )
        
        # Clustering options
        st.subheader("🤖 AI Analysis")
        clustering_method = st.selectbox(
            "Clustering method:",
            options=['kmeans', 'dbscan'],
            help="KMeans: Fixed number of clusters, DBSCAN: Automatic cluster detection"
        )
        
        if clustering_method == 'kmeans':
            n_clusters = st.slider("Number of clusters:", 2, 20, 5)
        else:
            eps = st.slider("DBSCAN eps:", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min samples:", 2, 20, 5)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 File Analysis")
        
        # Scan button
        if st.button("🔍 Scan Directory", type="primary"):
            if not os.path.exists(directory_path):
                st.error("❌ Directory not found. Please check the path.")
            else:
                with st.spinner("Scanning directory..."):
                    try:
                        # Initialize scanner
                        scanner = FileScanner(excluded_dirs=excluded_dirs)
                        
                        # Scan directory
                        file_metadata = scanner.scan_directory(directory_path)
                        st.session_state.file_metadata = [
                            {
                                'path': f.path,
                                'name': f.name,
                                'extension': f.extension,
                                'size': f.size,
                                'last_accessed': f.last_accessed,
                                'last_modified': f.last_modified,
                                'is_directory': f.is_directory,
                                'parent_dir': f.parent_dir
                            }
                            for f in file_metadata
                        ]
                        
                        st.success(f"✅ Found {len(file_metadata)} files!")
                        
                    except Exception as e:
                        st.error(f"❌ Error scanning directory: {e}")
        
        # Display file statistics
        if st.session_state.file_metadata:
            display_file_statistics(st.session_state.file_metadata)
    
    with col2:
        st.header("🎯 Quick Actions")
        
        # AI Analysis button
        if st.button("🤖 Run AI Analysis") and st.session_state.file_metadata:
            with st.spinner("Running AI analysis..."):
                try:
                    # Generate embeddings
                    embedding_gen = EmbeddingGenerator()
                    embeddings = embedding_gen.generate_file_embeddings(st.session_state.file_metadata)
                    st.session_state.embeddings = embeddings
                    
                    # Cluster files
                    classifier = FileClassifier()
                    if clustering_method == 'kmeans':
                        cluster_result = classifier.cluster_files(
                            embeddings, 
                            st.session_state.file_metadata, 
                            method='kmeans',
                            n_clusters=n_clusters
                        )
                    else:
                        cluster_result = classifier.cluster_files(
                            embeddings, 
                            st.session_state.file_metadata, 
                            method='dbscan',
                            eps=eps,
                            min_samples=min_samples
                        )
                    
                    st.session_state.cluster_result = cluster_result
                    
                    st.success(f"✅ AI analysis complete! Found {cluster_result.n_clusters} clusters.")
                    
                except Exception as e:
                    st.error(f"❌ Error in AI analysis: {e}")
        
        # Find duplicates button
        if st.button("🔍 Find Duplicates") and st.session_state.file_metadata:
            with st.spinner("Finding duplicate files..."):
                try:
                    utils = FileUtils()
                    duplicates = utils.find_duplicate_files(st.session_state.file_metadata)
                    st.session_state.duplicates = duplicates
                    
                    total_duplicates = sum(len(files) for files in duplicates.values())
                    st.success(f"✅ Found {len(duplicates)} groups of duplicates ({total_duplicates} files total)")
                    
                except Exception as e:
                    st.error(f"❌ Error finding duplicates: {e}")
    
    # Visualizations
    if st.session_state.file_metadata:
        st.header("📈 Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Treemap", "☀️ Sunburst", "🔍 Clusters"])
        
        with tab1:
            display_overview_charts()
        
        with tab2:
            display_treemap()
        
        with tab3:
            display_sunburst()
        
        with tab4:
            display_cluster_analysis()
    
    # Duplicates section
    if st.session_state.duplicates:
        st.header("🔄 Duplicate Files")
        display_duplicates()
    
    # Footer
    st.markdown("---")
    st.markdown("🔒 **Privacy First**: All processing happens locally on your machine. No data is sent to external services.")


def display_file_statistics(file_metadata: List[Dict[str, Any]]):
    """Display file statistics."""
    total_files = len(file_metadata)
    total_size = sum(f.get('size', 0) for f in file_metadata)
    total_size_gb = total_size / (1024**3)
    
    # File categories
    categories = {}
    for file_info in file_metadata:
        ext = file_info.get('extension', '').lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            categories['Images'] = categories.get('Images', 0) + 1
        elif ext in ['.mp4', '.avi', '.mkv', '.mov']:
            categories['Videos'] = categories.get('Videos', 0) + 1
        elif ext in ['.pdf', '.doc', '.docx', '.txt']:
            categories['Documents'] = categories.get('Documents', 0) + 1
        elif ext in ['.py', '.js', '.html', '.css']:
            categories['Code'] = categories.get('Code', 0) + 1
        else:
            categories['Other'] = categories.get('Other', 0) + 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", f"{total_files:,}")
    
    with col2:
        st.metric("Total Size", f"{total_size_gb:.2f} GB")
    
    with col3:
        avg_size = total_size / total_files if total_files > 0 else 0
        st.metric("Avg Size", f"{avg_size / (1024*1024):.1f} MB")
    
    with col4:
        most_common = max(categories.items(), key=lambda x: x[1]) if categories else ("None", 0)
        st.metric("Most Common", f"{most_common[0]} ({most_common[1]})")


def display_overview_charts():
    """Display overview charts."""
    if not st.session_state.file_metadata:
        return
    
    visualizer = FileVisualizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Size distribution
        fig_size = visualizer.create_size_distribution_chart(st.session_state.file_metadata)
        st.plotly_chart(fig_size, use_container_width=True)
    
    with col2:
        # Category pie chart
        fig_pie = visualizer.create_category_pie_chart(st.session_state.file_metadata)
        st.plotly_chart(fig_pie, use_container_width=True)


def display_treemap():
    """Display treemap visualization."""
    if not st.session_state.file_metadata:
        return
    
    visualizer = FileVisualizer()
    cluster_labels = st.session_state.cluster_result.labels if st.session_state.cluster_result else None
    
    fig = visualizer.create_treemap(
        st.session_state.file_metadata, 
        cluster_labels,
        "File Organization Treemap"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_sunburst():
    """Display sunburst visualization."""
    if not st.session_state.file_metadata:
        return
    
    visualizer = FileVisualizer()
    cluster_labels = st.session_state.cluster_result.labels if st.session_state.cluster_result else None
    
    fig = visualizer.create_sunburst(
        st.session_state.file_metadata, 
        cluster_labels,
        "File Organization Sunburst"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_cluster_analysis():
    """Display cluster analysis."""
    if not st.session_state.cluster_result or not st.session_state.embeddings is not None:
        st.info("Run AI analysis first to see cluster information.")
        return
    
    visualizer = FileVisualizer()
    
    # 2D cluster scatter plot
    fig_scatter = visualizer.create_cluster_scatter_plot(
        st.session_state.embeddings,
        st.session_state.cluster_result.labels,
        st.session_state.file_metadata
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Cluster summary
    st.subheader("📋 Cluster Summary")
    classifier = FileClassifier()
    summary = classifier.get_cluster_summary(st.session_state.cluster_result, st.session_state.file_metadata)
    
    for cluster_id, info in summary.items():
        with st.expander(f"Cluster {cluster_id} ({info['file_count']} files, {info['total_size_mb']:.1f} MB)"):
            st.write(f"**Most common extension:** {info['most_common_extension']}")
            st.write(f"**Extension diversity:** {info['extension_diversity']}")
            
            # Show sample files
            sample_files = info['files'][:10]  # Show first 10 files
            file_df = pd.DataFrame([
                {
                    'Name': f['name'],
                    'Size (MB)': f['size'] / (1024*1024),
                    'Extension': f['extension']
                }
                for f in sample_files
            ])
            st.dataframe(file_df, use_container_width=True)


def display_duplicates():
    """Display duplicate files."""
    if not st.session_state.duplicates:
        return
    
    st.subheader("🔄 Duplicate Files Found")
    
    for hash_value, files in st.session_state.duplicates.items():
        with st.expander(f"Duplicate Group ({len(files)} files) - {hash_value[:16]}..."):
            file_df = pd.DataFrame([
                {
                    'Path': f['path'],
                    'Name': f['name'],
                    'Size (MB)': f['size'] / (1024*1024),
                    'Modified': pd.to_datetime(f['last_modified'], unit='s').strftime('%Y-%m-%d %H:%M')
                }
                for f in files
            ])
            st.dataframe(file_df, use_container_width=True)
            
            # Action buttons for each duplicate group
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🗑️ Move to Quarantine", key=f"quarantine_{hash_value}"):
                    st.info("Quarantine functionality will be implemented in the next version.")
            with col2:
                if st.button(f"📋 Select Files", key=f"select_{hash_value}"):
                    st.info("File selection functionality will be implemented in the next version.")


if __name__ == "__main__":
    main()
