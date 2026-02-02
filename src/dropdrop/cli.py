"""Command-line interface for DropDrop pipeline."""

import argparse
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cache import CacheManager
from .config import load_config
from .pipeline import DropletInclusionPipeline
from .stats import DropletStatistics
from .ui import InclusionEditor, Viewer


def parse_settings(settings_str):
    """Parse compact settings string.

    Format: key=value,key=value
    Keys: d[ilution], p[oisson], c[ount], l[abel]

    Examples:
        "d=1000,p=on,c=6.5e5,l=experiment1"
        "dilution=500,poisson=off"
    """
    settings = {
        "dilution": 500,
        "poisson": True,
        "count": 6.5e5,
        "label": None,
    }

    if not settings_str:
        return settings

    key_map = {
        "d": "dilution",
        "dilution": "dilution",
        "p": "poisson",
        "poisson": "poisson",
        "c": "count",
        "count": "count",
        "l": "label",
        "label": "label",
    }

    for part in settings_str.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key_map.get(key.strip().lower(), key.strip().lower())

        if key == "dilution":
            settings["dilution"] = int(value)
        elif key == "poisson":
            settings["poisson"] = value.lower() in ("on", "yes", "true", "1")
        elif key == "count":
            settings["count"] = float(value)
        elif key == "label":
            settings["label"] = value.strip()

    return settings


def prompt_settings():
    """Interactive prompts for settings when --settings not provided."""
    settings = {"dilution": 500, "poisson": True, "count": 6.5e5, "label": None}

    print("\n--- Project Settings ---")

    # Poisson analysis
    use_poisson = input("Use Poisson analysis? [yes/no] (yes): ").strip().lower()
    settings["poisson"] = use_poisson != "no"

    if settings["poisson"]:
        # Bead count
        count_input = input("Stock count/uL [6.5e5]: ").strip()
        if count_input:
            try:
                settings["count"] = float(count_input)
            except ValueError:
                print(f"  Invalid value, using default: {settings['count']}")

        # Dilution
        dilution_input = input("Dilution factor [500]: ").strip()
        if dilution_input:
            try:
                settings["dilution"] = int(dilution_input)
            except ValueError:
                print(f"  Invalid value, using default: {settings['dilution']}")

    # Label
    label_input = input("Project label (optional, press Enter to skip): ").strip()
    settings["label"] = label_input if label_input else None

    print("------------------------\n")
    return settings


def generate_project_name(settings):
    """Generate project directory name from date and label."""
    date_str = datetime.now().strftime("%Y%m%d")
    if settings.get("label"):
        return f"{date_str}_{settings['label']}"
    return date_str


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Droplet and inclusion detection pipeline using Cellpose"
    )

    parser.add_argument(
        "input_dir", type=str, help="Input directory containing z-stack images"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Output directory (default: ./results/<date>_<label>)",
    )

    parser.add_argument(
        "-s",
        "--settings",
        type=str,
        default=None,
        help='Compact settings: "d=1000,p=on,c=6.5e5,l=label" (d=dilution, p=poisson, c=count, l=label)',
    )

    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--view", action="store_true", help="Enable interactive viewer after processing"
    )
    viewer_group.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive inclusion correction mode",
    )

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        help="Process only the first N frames (for testing)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for this run",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before processing",
    )
    parser.add_argument(
        "-z",
        "--gzip",
        action="store_true",
        help="Archive project directory as .tar.gz after completion",
    )
    args = parser.parse_args()

    # Check input directory exists
    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    # Get settings (from --settings or interactive prompts)
    if args.settings:
        settings = parse_settings(args.settings)
    else:
        settings = prompt_settings()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_name = generate_project_name(settings)
        output_dir = Path("results") / project_name

    # Store settings for later use
    settings["input_dir"] = str(Path(args.input_dir).resolve())

    # Initialize and run pipeline
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    if settings["poisson"]:
        print(f"Poisson: ON (count={settings['count']:.2e}, dilution={settings['dilution']})")
    else:
        print("Poisson: OFF")
    if args.number:
        print(f"Frame limit: {args.number}")

    # Create pipeline with visualization storage if viewer is requested
    store_viz = args.view or args.interactive
    use_cache = not args.no_cache
    pipeline = DropletInclusionPipeline(store_visualizations=store_viz, use_cache=use_cache)

    # Handle cache clear request
    if args.clear_cache and pipeline.cache:
        pipeline.cache.clear()

    results = pipeline.run(args.input_dir, str(output_dir), frame_limit=args.number)

    if results:
        print("\nPipeline completed successfully!")

        # Interactive editing mode
        if args.interactive and pipeline.visualization_data:
            print("\nLaunching interactive inclusion editor...")
            editor = InclusionEditor(pipeline.visualization_data, results)
            results = editor.run()  # Update results with manual corrections

            # Save updated results
            df = pd.DataFrame(results)
            csv_path = output_dir / "data.csv"
            df.to_csv(csv_path, index=False)
            print(f"Updated results saved to: {csv_path}")

        # Always generate statistics (after any interactive corrections)
        print("\nGenerating statistical analysis...")
        csv_path = output_dir / "data.csv"
        stats_module = DropletStatistics(csv_path, settings)
        stats_module.run_analysis(str(output_dir))

        # Launch viewer if requested (no editing, just viewing)
        if args.view and pipeline.visualization_data:
            print("\nLaunching interactive viewer...")
            df = pd.DataFrame(results)
            viewer = Viewer(pipeline.visualization_data, df)
            viewer.run()

        # Archive project if requested
        if args.gzip:
            archive_name = f"{output_dir}.tar.gz"
            print(f"\nArchiving project to: {archive_name}")
            with tarfile.open(archive_name, "w:gz") as tar:
                tar.add(output_dir, arcname=output_dir.name)
            print(f"Archive created: {archive_name}")


if __name__ == "__main__":
    main()
