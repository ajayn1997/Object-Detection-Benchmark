"""
Object Detection Benchmark - Interactive Streamlit Application
A comprehensive web interface for testing and comparing object detection models
"""

import streamlit as st
import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the models directory to the path
sys.path.append('../models')
from object_detection_models import (
    YOLOv8Detector, YOLOv5Detector, SSDDetector, 
    FasterRCNNDetector, DETRDetector, RetinaNetDetector
)

# Configure Streamlit page
st.set_page_config(
    page_title="Object Detection Benchmark",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .detection-box {
        border: 2px solid #ff6b6b;
        background-color: rgba(255, 107, 107, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = {}
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = None

# Model information
MODEL_INFO = {
    "YOLOv8": {
        "description": "Latest YOLO variant with anchor-free design",
        "strengths": ["Fast inference", "Good accuracy", "Easy deployment"],
        "use_cases": ["Real-time applications", "Video processing", "General purpose"]
    },
    "YOLOv5": {
        "description": "Mature YOLO variant with extensive ecosystem",
        "strengths": ["Production ready", "Great documentation", "Transfer learning"],
        "use_cases": ["Commercial applications", "Custom training", "Production systems"]
    },
    "SSD MobileNet": {
        "description": "Lightweight detector for mobile deployment",
        "strengths": ["Small model size", "Mobile optimized", "Low memory usage"],
        "use_cases": ["Mobile apps", "Edge devices", "IoT applications"]
    },
    "Faster R-CNN": {
        "description": "Two-stage detector with high accuracy",
        "strengths": ["High precision", "Good for small objects", "Research proven"],
        "use_cases": ["Medical imaging", "Quality control", "High-precision tasks"]
    },
    "DETR": {
        "description": "Transformer-based end-to-end detector",
        "strengths": ["No hand-crafted components", "Set prediction", "Research frontier"],
        "use_cases": ["Research applications", "Complex scenes", "Novel architectures"]
    },
    "RetinaNet": {
        "description": "Single-stage detector with Focal Loss",
        "strengths": ["Handles class imbalance", "Good accuracy", "Dense detection"],
        "use_cases": ["Challenging datasets", "Dense objects", "Balanced performance"]
    }
}

def load_models():
    """Load all object detection models"""
    if not st.session_state.models_loaded:
        with st.spinner("Loading object detection models... This may take a few minutes."):
            try:
                models = {
                    "YOLOv8": YOLOv8Detector(),
                    "YOLOv5": YOLOv5Detector(),
                    "SSD MobileNet": SSDDetector(),
                    "Faster R-CNN": FasterRCNNDetector(),
                    "DETR": DETRDetector(),
                    "RetinaNet": RetinaNetDetector()
                }
                
                # Load each model
                for name, model in models.items():
                    model.load_model()
                
                st.session_state.models = models
                st.session_state.models_loaded = True
                st.success("✅ All models loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                return False
    return True

def run_detection(image, selected_models):
    """Run object detection on the uploaded image"""
    results = {}
    
    # Save uploaded image temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(selected_models):
        status_text.text(f"Running {model_name}...")
        
        model = st.session_state.models[model_name]
        
        # Measure inference time
        start_time = time.time()
        detections = model.predict(temp_path)
        inference_time = time.time() - start_time
        
        results[model_name] = {
            "detections": detections,
            "inference_time": inference_time * 1000,  # Convert to milliseconds
            "num_detections": len(detections)
        }
        
        progress_bar.progress((i + 1) / len(selected_models))
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    status_text.text("✅ Detection complete!")
    return results

def visualize_detections(image, results):
    """Create visualization of detection results"""
    num_models = len(results)
    if num_models == 0:
        return None
    
    # Calculate grid dimensions
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    for i, (model_name, result) in enumerate(results.items()):
        ax = axes[i] if num_models > 1 else axes[0]
        ax.imshow(img_array)
        ax.set_title(f"{model_name}\n{result['num_detections']} detections, {result['inference_time']:.1f}ms")
        ax.axis('off')
        
        # Draw bounding boxes
        for detection in result['detections']:
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Create rectangle
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            ax.text(x, y-5, label, color='red', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Hide unused subplots
    for i in range(num_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def create_performance_chart(results):
    """Create interactive performance comparison chart"""
    if not results:
        return None
    
    models = list(results.keys())
    inference_times = [results[model]['inference_time'] for model in models]
    num_detections = [results[model]['num_detections'] for model in models]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for inference time
    fig.add_trace(
        go.Bar(x=models, y=inference_times, name="Inference Time (ms)", 
               marker_color='lightblue', opacity=0.8),
        secondary_y=False,
    )
    
    # Add line chart for number of detections
    fig.add_trace(
        go.Scatter(x=models, y=num_detections, mode='lines+markers', 
                  name="Number of Detections", line=dict(color='red', width=3),
                  marker=dict(size=8)),
        secondary_y=True,
    )
    
    # Set axis titles
    fig.update_xaxes(title_text="Models")
    fig.update_yaxes(title_text="Inference Time (ms)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Detections", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title="Performance Comparison",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def display_model_info():
    """Display information about available models"""
    st.markdown("## 🤖 Available Models")
    
    for model_name, info in MODEL_INFO.items():
        with st.expander(f"📊 {model_name}"):
            st.markdown(f"**Description:** {info['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Key Strengths:**")
                for strength in info['strengths']:
                    st.markdown(f"• {strength}")
            
            with col2:
                st.markdown("**Best Use Cases:**")
                for use_case in info['use_cases']:
                    st.markdown(f"• {use_case}")

def load_benchmark_results():
    """Load pre-computed benchmark results"""
    try:
        results_path = "../results/benchmark_results.json"
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading benchmark results: {e}")
    return None

def display_benchmark_results():
    """Display comprehensive benchmark results"""
    st.markdown("## 📊 Comprehensive Benchmark Results")
    
    if st.session_state.benchmark_results is None:
        st.session_state.benchmark_results = load_benchmark_results()
    
    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        # Summary metrics
        st.markdown("### 🏆 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fastest Model", results['summary']['fastest_model'])
        with col2:
            st.metric("Most Detections", results['summary']['most_detections'])
        with col3:
            st.metric("Total Images", results['dataset_info']['total_images'])
        with col4:
            st.metric("Models Tested", len(results['model_results']))
        
        # Detailed results table
        st.markdown("### 📈 Detailed Performance Metrics")
        
        performance_data = []
        for model_name, model_results in results['model_results'].items():
            performance_data.append({
                'Model': model_name,
                'Avg Inference Time (ms)': f"{model_results['avg_inference_time'] * 1000:.1f}",
                'Total Detections': model_results['total_detections'],
                'Images Processed': model_results['total_images'],
                'Detections per Image': f"{model_results['total_detections'] / model_results['total_images']:.1f}"
            })
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
        
        # Interactive charts
        st.markdown("### 📊 Interactive Performance Analysis")
        
        # Prepare data for plotting
        models = list(results['model_results'].keys())
        avg_times = [results['model_results'][model]['avg_inference_time'] * 1000 for model in models]
        total_detections = [results['model_results'][model]['total_detections'] for model in models]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Inference time chart
            fig_time = px.bar(
                x=models, y=avg_times,
                title="Average Inference Time by Model",
                labels={'x': 'Model', 'y': 'Inference Time (ms)'},
                color=avg_times,
                color_continuous_scale='viridis'
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Detection count chart
            fig_det = px.bar(
                x=models, y=total_detections,
                title="Total Detections by Model",
                labels={'x': 'Model', 'y': 'Total Detections'},
                color=total_detections,
                color_continuous_scale='plasma'
            )
            fig_det.update_layout(height=400)
            st.plotly_chart(fig_det, use_container_width=True)
        
        # Speed vs Accuracy scatter plot
        st.markdown("### ⚡ Speed vs Performance Trade-off")
        fig_scatter = px.scatter(
            x=avg_times, y=total_detections,
            text=models,
            title="Speed vs Detection Performance",
            labels={'x': 'Average Inference Time (ms)', 'y': 'Total Detections'},
            size=[10] * len(models),
            color=models
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.warning("⚠️ Benchmark results not available. Please run the benchmark first.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Object Detection Benchmark</h1>', unsafe_allow_html=True)
    st.markdown("**Compare state-of-the-art object detection models interactively**")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Interactive Detection", "📊 Benchmark Results", "📚 Model Information", "ℹ️ About"]
    )
    
    if page == "🏠 Interactive Detection":
        st.markdown("## 🖼️ Upload Image for Detection")
        
        # Load models
        if not load_models():
            st.stop()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to test object detection models"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Model selection
                st.markdown("### 🎯 Select Models to Test")
                available_models = list(MODEL_INFO.keys())
                selected_models = st.multiselect(
                    "Choose models for comparison:",
                    available_models,
                    default=["YOLOv8", "Faster R-CNN"],
                    help="Select one or more models to compare their performance"
                )
                
                if st.button("🚀 Run Detection", type="primary"):
                    if selected_models:
                        st.session_state.detection_results = run_detection(image, selected_models)
                    else:
                        st.warning("Please select at least one model.")
            
            with col2:
                if st.session_state.detection_results:
                    st.markdown("### 🎯 Detection Results")
                    
                    # Display visualization
                    fig = visualize_detections(image, st.session_state.detection_results)
                    if fig:
                        st.pyplot(fig)
                    
                    # Performance comparison
                    st.markdown("### ⚡ Performance Comparison")
                    perf_fig = create_performance_chart(st.session_state.detection_results)
                    if perf_fig:
                        st.plotly_chart(perf_fig, use_container_width=True)
                    
                    # Detailed results
                    st.markdown("### 📋 Detailed Results")
                    for model_name, result in st.session_state.detection_results.items():
                        with st.expander(f"📊 {model_name} Results"):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Inference Time", f"{result['inference_time']:.1f} ms")
                            with col_b:
                                st.metric("Detections", result['num_detections'])
                            with col_c:
                                avg_conf = np.mean([d['confidence'] for d in result['detections']]) if result['detections'] else 0
                                st.metric("Avg Confidence", f"{avg_conf:.3f}")
                            
                            if result['detections']:
                                st.markdown("**Detected Objects:**")
                                for i, detection in enumerate(result['detections']):
                                    st.markdown(
                                        f"**{i+1}.** {detection['class']} "
                                        f"(confidence: {detection['confidence']:.3f}, "
                                        f"bbox: {detection['bbox']})"
                                    )
                            else:
                                st.info("No objects detected")
        
        else:
            st.info("👆 Please upload an image to start testing object detection models")
            
            # Show sample images
            st.markdown("### 📸 Sample Images")
            sample_dir = Path("../data/images")
            if sample_dir.exists():
                sample_images = list(sample_dir.glob("*.jpg"))
                if sample_images:
                    cols = st.columns(min(3, len(sample_images)))
                    for i, img_path in enumerate(sample_images[:3]):
                        with cols[i]:
                            sample_img = Image.open(img_path)
                            st.image(sample_img, caption=img_path.name, use_column_width=True)
    
    elif page == "📊 Benchmark Results":
        display_benchmark_results()
    
    elif page == "📚 Model Information":
        display_model_info()
    
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This interactive web application provides a comprehensive platform for testing and comparing 
        state-of-the-art object detection models. It enables users to:
        
        - **Upload custom images** and test them with multiple detection models
        - **Compare performance** across different architectures and approaches
        - **Analyze results** with interactive visualizations and detailed metrics
        - **Explore model characteristics** to make informed decisions for specific use cases
        
        ### 🤖 Included Models
        The application benchmarks six representative object detection models:
        
        1. **YOLOv8** - Latest YOLO variant with anchor-free design
        2. **YOLOv5** - Mature YOLO variant with extensive ecosystem support
        3. **SSD MobileNet** - Lightweight detector optimized for mobile deployment
        4. **Faster R-CNN** - Two-stage detector emphasizing high accuracy
        5. **DETR** - Transformer-based end-to-end detector
        6. **RetinaNet** - Single-stage detector with Focal Loss
        
        ### 📊 Evaluation Metrics
        The benchmark evaluates models across multiple dimensions:
        
        - **Inference Speed** - Time required to process each image
        - **Detection Accuracy** - Number and quality of detected objects
        - **Confidence Scores** - Model certainty in its predictions
        - **Resource Usage** - Memory and computational requirements
        
        ### 🛠️ Technical Implementation
        - **Framework**: Streamlit for interactive web interface
        - **Models**: PyTorch and Ultralytics implementations
        - **Visualization**: Matplotlib and Plotly for charts and graphs
        - **Deployment**: Containerized for easy deployment and scaling
        
        ### 📚 Use Cases
        This tool is designed for:
        - **Researchers** comparing model architectures and performance
        - **Engineers** selecting models for production deployment
        - **Students** learning about object detection technologies
        - **Practitioners** evaluating models for specific applications
        
        ### 🔗 Resources
        - [GitHub Repository](https://github.com/your-repo/object-detection-benchmark)
        - [Technical Documentation](https://docs.your-site.com)
        - [Research Paper](https://arxiv.org/your-paper)
        - [Blog Post](https://medium.com/your-article)
        """)
        
        st.markdown("---")
        st.markdown("**Created by Manus AI** | *Advancing practical AI applications*")

if __name__ == "__main__":
    main()

