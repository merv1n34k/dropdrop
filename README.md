# DropDrop

Automated pipeline for detecting droplets and inclusions (beads) in microscopy z-stacks using Cellpose segmentation and morphological analysis.

Tailored for the EVOS M5000 Imaging System.

## Installation

```bash
# Using uv (recommended)
uv pip install dropdrop

# Using pip
pip install dropdrop
```

### GPU support (CUDA)

On Linux/Windows with an NVIDIA GPU, install CUDA-enabled PyTorch **before** installing DropDrop:

```bash
# Install PyTorch with CUDA 12.6 (adjust version for your driver)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Then install DropDrop
uv pip install dropdrop
```

On macOS, GPU acceleration via Metal (MPS) is included in the default PyTorch build — no extra steps needed.

### From source

```bash
git clone https://github.com/yourusername/dropdrop.git
cd dropdrop
uv pip install -e .
```

## Quick Start

```bash
# Single directory — interactive prompts for settings
dropdrop ./images

# Process only first 5 frames (for testing)
dropdrop ./images -n 5

# With interactive editor
dropdrop ./images -e

# Multiplex mode — batch process subdirectories
dropdrop -m ./samples
```

## Usage

### Single Mode

```bash
# Basic run (prompts for settings interactively)
dropdrop ./images

# Custom output directory
dropdrop ./images ./results/my_project

# With editor and archive
dropdrop ./images -e -z
```

### Multiplex Mode

Process multiple sample directories at once. Each subdirectory is labeled interactively and processed as a separate sample, then combined into a multiplexed report.

```bash
dropdrop -m ./samples_parent_dir
dropdrop -m ./samples_parent_dir -z    # Archive result
```

### Resume (Resurrect)

If a multiplex run is interrupted, resume from where it left off:

```bash
dropdrop -r
```

### Cache Control

```bash
dropdrop ./images --no-cache       # Disable caching
dropdrop ./images --clear-cache    # Clear cache before run
```

## Interactive Editor

The editor (`-e`) allows manual correction of detected inclusions:

| Key | Action |
|-----|--------|
| Left-click | Add inclusion |
| Right-click (hold) | Remove inclusions |
| `s` | Toggle droplet selection (hover over droplet) |
| `u` | Undo last action |
| `c` | Clear all inclusions in frame |
| `d` | Toggle droplet visibility |
| Arrow keys / Space | Navigate frames |
| `q` / Esc | Exit |

Disabled droplets (gray with X) are excluded from the final results.

## Output Structure

### Single mode

```
results/<YYYYMMDD>_<label>/
  data.csv                  # Raw detection data
  summary.txt               # Settings and statistics
  report.png                # Combined report with sample frames
  size_distribution.png     # Droplet diameter histogram
  poisson_comparison.png    # Bead distribution vs theoretical
```

### Multiplex mode

```
results/<YYYYMMDD>_multiplex/
  data.csv                  # Merged data with sample column
  summary.txt               # Per-sample statistics
  summary_report.png        # Comparison table, overlaid plots, sample collage
  size_distribution.png     # Overlaid diameter histograms
  poisson_comparison.png    # Overlaid inclusion distributions
```

### data.csv columns

| Column | Description |
|--------|-------------|
| `sample` | Sample label (multiplex only) |
| `frame` | Frame index |
| `droplet_id` | Droplet ID within frame |
| `center_x`, `center_y` | Droplet center coordinates (px) |
| `diameter_px`, `diameter_um` | Droplet diameter |
| `area_px`, `area_um2` | Droplet area |
| `inclusions` | Number of inclusions detected |

## Architecture

```
CLI
  -> Detection (per sample) -> .tmp_<label>/data.csv + sample_*.png
  -> Analysis.run(output_dir)  -- auto-discovers .tmp_* dirs
       1 sample  -> single report
       2+ samples -> multiplex report
  -> Cleanup .tmp_* dirs
  -> Archive (optional)
```

## Configuration

Create `config.json` in your working directory to customize detection parameters:

```json
{
  "cellpose_flow_threshold": 0.4,
  "cellpose_cellprob_threshold": 0.0,
  "erosion_pixels": 5,
  "kernel_size": 7,
  "tophat_threshold": 30,
  "min_inclusion_area": 7,
  "max_inclusion_area": 50,
  "edge_buffer": 5,
  "min_droplet_diameter": 80,
  "max_droplet_diameter": 200,
  "px_to_um": 1.14,
  "cache": {
    "enabled": true,
    "max_frames": 100
  }
}
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `cellpose_flow_threshold` | Cellpose flow threshold for segmentation |
| `cellpose_cellprob_threshold` | Cellpose cell probability threshold |
| `erosion_pixels` | Pixels to erode droplet mask before inclusion detection |
| `kernel_size` | Morphological kernel size for black-hat transform |
| `tophat_threshold` | Threshold for inclusion detection |
| `min/max_inclusion_area` | Inclusion size constraints (px) |
| `edge_buffer` | Buffer from image edge to ignore inclusions |
| `min/max_droplet_diameter` | Droplet size constraints (px) |
| `px_to_um` | Pixel to micrometer conversion factor |

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended for Cellpose)

## License

MIT
