# DropDrop

This is an automated script for detection of droplets and inclusions in microscopy, works with z-stacks using Cellpose and morphological analysis. (Tailored to EVOS™ M5000 Imaging System)

## Installation
```bash
pip install cellpose opencv-python numpy pandas matplotlib scipy seaborn tqdm
```

## Usage
```bash
# Basic
python pipeline.py ./images ./results

# With viewer
python pipeline.py ./images ./results --view

# With statistics
python pipeline.py ./images ./results --stats

# Both
python pipeline.py ./images ./results --view --stats
```

## Files
- `pipeline.py` - Main detection pipeline + viewer + statics
- `config.json` - Optional parameters

## Config
```json
{
  # Internal system settings
  "cellpose_flow_threshold": 0.4,
  "cellpose_cellprob_threshold": 0.0,
  "kernel_size": 7,
  "tophat_threshold": 30,
  # Common configurable parameters
  "erosion_pixels": 5,
  "min_inclusion_area": 7,
  "max_inclusion_area": 50,
  "edge_buffer": 5,
  "min_droplet_diameter": 80,
  "max_droplet_diameter": 200,
  "px_to_um": 1.14
}
```

Please use `config.json`, for configuring the pipeline.

## Outputs
- `results.csv` - Droplet data with inclusion counts
- `*.png` - Statistical plots (with --stats)
- `statistical_report.txt` - Analysis report

## License

Distributed under MIT License, see `LICENSE`
