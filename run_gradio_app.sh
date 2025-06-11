#!/bin/bash

# Loop Trajectory Visualizer Gradio App Launcher
echo "🔄 Loop Trajectory Visualizer"
echo "================================"

# Check if requirements are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements_gradio.txt
fi

# Launch with default settings
echo "🚀 Launching Gradio app..."
echo "   Access at: http://localhost:7860"
echo "   Press Ctrl+C to stop"
echo ""

python launch_gradio_app.py --debug

echo "👋 App stopped." 