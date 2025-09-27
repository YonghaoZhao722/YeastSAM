"""
CLI entry point for YeastSAM package.

This module provides the command-line interface for YeastSAM.
Users can run 'yeastsam' command after installing the package.
"""

import sys
import argparse
from . import launch_gui, get_tools, __version__

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YeastSAM: Yeast cell analysis and mask processing tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yeastsam                    # Launch GUI
  yeastsam --version         # Show version
  yeastsam --tools           # List available tools
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"YeastSAM {__version__}"
    )
    
    parser.add_argument(
        "--tools",
        action="store_true",
        help="List available tools and exit"
    )
    
    args = parser.parse_args()
    
    if args.tools:
        print("Available YeastSAM tools:")
        print("-" * 40)
        for tool, description in get_tools().items():
            print(f"{tool:20} - {description}")
        return
    
    # Default action: launch GUI
    try:
        launch_gui()
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
