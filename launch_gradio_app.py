#!/usr/bin/env python
"""
Simple launcher for the Loop Trajectory Visualizer Gradio app.
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch the Loop Trajectory Visualizer Gradio app")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--share', action='store_true', help='Create public shareable link')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Import the app after parsing args to avoid import issues
    try:
        from gradio_loop_trajectory_app import create_gradio_app
        
        print("🚀 Starting Loop Trajectory Visualizer...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        print(f"   Debug: {args.debug}")
        print("\n📖 Make sure you have:")
        print("   - A trained GPT model checkpoint (.pt file)")
        print("   - The corresponding tokenizer meta.pkl file")
        print("   - model.py in the same directory")
        print("\n🔄 Starting server...")
        
        app = create_gradio_app()
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all requirements:")
        print("   pip install -r requirements_gradio.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 