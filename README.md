# YeastSAM

<div align="center">
  <img src="src/yeastsam.png" alt="YeastSAM Logo" width="200"/>
</div>

YeastSAM is a model and a framework for yeast cell analysis and mask processing. It provides an intuitive GUI launcher and various tools for generating masks, image registration, and outline conversion. For detailed documentation, visit [YeastSAM](https://yeastsamdoc.readthedocs.io/en/latest/).

## Quick Start

If you just want to run YeastSAM, you can refer to [µSAM](https://github.com/computational-cell-analytics/micro-sam) to install the framework. We provide custom weights for better yeast cell segmentation that can be downloaded from our [GitHub Releases](https://github.com/YonghaoZhao722/YeastSAM/releases). The custom weights can be used with napari, [BAND](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-micro_sam-on-band), [CLI](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#using-the-command-line-interface-cli), and [QuPath](https://github.com/ksugar/qupath-extension-sam).

## Installation

### With Conda

run the following commands in your terminal:

```bash
git clone https://github.com/YonghaoZhao722/YeastSAM.git
conda create -n yeastsam -c conda-forge -c defaults python=3.10 micro_sam
conda activate yeastsam
```

## Usage

### Launch the GUI

To start the YeastSAM tools launcher:

```bash
python launch.py
```

This will open a GUI with four main sections:

### 1. Generate Masks
- **napari**: Opens napari viewer for interactive mask generation and editing. You can load our custom weight YeastSAM for better accuracy in budding yeast. 

### 2. Optional Tools
This section is to fix the offset between smFISH image and DIC (as masks are generated from DIC).
- **Shift Analyzer**: Analyze and detect shifts in your image data
- **Apply Registration**: Apply image registration corrections

### 3. Convert to Outline File
- **Mask2Outline**: Convert mask files to FISH-Quant compatible outline format

### 4. Separation Module
- **Mask Editor**: You can annotate mask images manually or with CNN & U-Net separation module. Download models at [GitHub Releases](https://github.com/YonghaoZhao722/YeastSAM/releases).

## Tools Overview

The `tools/` directory contains the following utilities:

- `shift.py`: Shift detection and analysis
- `registration.py`: Image registration functionality
- `Mask2Outline.py`: Mask to outline conversion
- `mask_editor.py`: Advanced mask editing interface

**Note**: On macOS these tools are not compatible with Python 3.13. If you follow the installation command we provide above, everything should work correctly. However, if you directly use the conda instructions from the [MicroSAM tutorial page](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html), you may run into errors. In that case, please downgrade your Python version to 3.10 and recreate the environment.


## Acknowledgement

We acknowledge the following reference for inspiring our work:

Archit, A., Freckmann, L., Nair, S. et al. *Segment Anything for Microscopy*. Nat Methods 22, 579–591 (2025). https://doi.org/10.1038/s41592-024-02580-4

See the full [COPYRIGHT](src/COPYRIGHT) notice for details.
