# DropDrop

Automated Python pipeline for detecting droplets and inclusions (beads) in microscopy z-stacks using Cellpose segmentation and morphological analysis.

Tailored for the EVOS M5000 Imaging System.

## Installation

```bash
# Using uv (recommended)
uv pip install dropdrop

# Using pip
pip install dropdrop
```

### From source

```bash
git clone https://github.com/yourusername/dropdrop.git
cd dropdrop
uv pip install -e .
```

## Quick Start

```bash
# Run with interactive prompts
dropdrop ./images

# Run with settings
dropdrop ./images --settings "d=1000,p=on,l=experiment1"

# Process only first 5 frames (for testing)
dropdrop ./images -n 5
```

## Usage

### Basic Commands

```bash
# Run pipeline with compact settings
dropdrop ./images --settings "d=1000,p=on,c=6.5e5,l=experiment1"

# Custom output directory
dropdrop ./images ./results/my_project --settings "d=500"

# Interactive viewer (view results after processing)
dropdrop ./images --view

# Interactive editor (manually correct inclusions)
dropdrop ./images --interactive

# Archive output as tar.gz
dropdrop ./images -z
```

### Settings Format

Compact settings string: `d=dilution,p=poisson,c=count,l=label`

| Key | Full name | Description | Default |
|-----|-----------|-------------|---------|
| `d` | `dilution` | Dilution factor | 500 |
| `p` | `poisson` | Enable Poisson analysis (on/off) | on |
| `c` | `count` | Stock bead count per uL | 6.5e5 |
| `l` | `label` | Project label for output naming | None |

### Cache Control

```bash
dropdrop ./images --no-cache       # Disable caching
dropdrop ./images --clear-cache    # Clear cache before run
```

## Interactive Editor

The editor allows manual correction of detected inclusions:

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

```
results/<YYYYMMDD>_<label>/
  data.csv                  # Raw detection data
  summary.txt               # Settings and statistics
  size_distribution.png     # Droplet diameter histogram
  poisson_comparison.png    # Bead distribution vs theoretical
```

### data.csv columns

| Column | Description |
|--------|-------------|
| `frame` | Frame index |
| `droplet_id` | Droplet ID within frame |
| `center_x`, `center_y` | Droplet center coordinates (px) |
| `diameter_px`, `diameter_um` | Droplet diameter |
| `area_px`, `area_um2` | Droplet area |
| `inclusions` | Number of inclusions detected |

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
