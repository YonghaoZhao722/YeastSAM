"""
YeastSAM: Yeast cell analysis and mask processing tools.

A comprehensive toolkit for yeast cell segmentation, mask processing,
and analysis using micro-sam and custom deep learning models.
"""

__version__ = "0.1.0"
__author__ = "Yonghao Zhao"

def ensure_microsam():
    """Check if micro-sam is available and provide helpful error message if not."""
    try:
        import micro_sam  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "YeastSAM requires `micro-sam` (only available via conda).\n"
            "Please run:\n"
            "  conda install -c conda-forge micro-sam\n"
            f"Original error: {e}"
        )

def launch_gui():
    """Launch the YeastSAM GUI launcher."""
    ensure_microsam()
    
    # Import and run the main GUI
    import sys
    import os
    
    # Add the project root to Python path to find launch.py
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from launch import main
    main()

def get_tools():
    """Get available tools in the YeastSAM toolkit."""
    return {
        "napari": "Interactive mask generation and editing",
        "shift_analyzer": "Analyze and detect shifts in image data", 
        "registration": "Apply image registration corrections",
        "mask2outline": "Convert mask files to FISH-Quant compatible format",
        "mask_editor": "Advanced mask editing with CNN & U-Net separation"
    }

# Make key functions available at package level
__all__ = ["launch_gui", "get_tools", "ensure_microsam"]
