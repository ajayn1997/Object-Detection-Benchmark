#!/bin/bash

# Object Detection Benchmark - Deployment Script
# This script sets up the complete environment and runs the benchmark

set -e  # Exit on any error

echo "🚀 Object Detection Benchmark - Deployment Script"
echo "=================================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Download dataset if not exists
if [ ! -d "data/images" ]; then
    echo "📥 Downloading dataset..."
    python download_dataset.py
else
    echo "✅ Dataset already exists"
fi

# Run benchmark if results don't exist
if [ ! -f "results/benchmark_results.json" ]; then
    echo "🔬 Running benchmark..."
    python models/object_detection_models.py
else
    echo "✅ Benchmark results already exist"
fi

# Function to start Streamlit app
start_streamlit() {
    echo "🌐 Starting Streamlit application..."
    cd streamlit_app
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
}

# Function to run Jupyter notebook
start_jupyter() {
    echo "📓 Starting Jupyter notebook..."
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
}

# Check command line arguments
case "${1:-streamlit}" in
    "streamlit")
        start_streamlit
        ;;
    "jupyter")
        start_jupyter
        ;;
    "benchmark")
        echo "🔬 Running benchmark only..."
        python models/object_detection_models.py
        ;;
    "setup")
        echo "✅ Setup complete! Use './deploy.sh streamlit' to start the app"
        ;;
    *)
        echo "Usage: $0 [streamlit|jupyter|benchmark|setup]"
        echo "  streamlit  - Start Streamlit web application (default)"
        echo "  jupyter    - Start Jupyter notebook server"
        echo "  benchmark  - Run benchmark only"
        echo "  setup      - Setup environment only"
        exit 1
        ;;
esac

echo "✅ Deployment complete!"

