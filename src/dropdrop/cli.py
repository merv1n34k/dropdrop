"""Command-line interface for DropDrop pipeline."""

import argparse
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import load_config
from .pipeline import DropletInclusionPipeline
from .stats import DropletStatistics
from .ui import Editor


def prompt_settings(config=None):
    """Interactive prompts for project settings.

    Defaults are loaded from config.json 'settings' section.
    """
    if config is None:
        config = load_config()
    defaults = config.get("settings", {})

    settings = {
        "dilution": defaults.get("dilution", 500),
        "poisson": defaults.get("poisson", True),
        "count": defaults.get("count", 6.5e5),
        "label": None,
        "inclusions": defaults.get("inclusions", True),
    }

    inc_default = "yes" if settings["inclusions"] else "no"
    poi_default = "yes" if settings["poisson"] else "no"

    print("\n--- Project Settings ---")

    # Inclusion detection
    use_inclusions = input(f"Detect inclusions? [yes/no] ({inc_default}): ").strip().lower()
    if use_inclusions:
        settings["inclusions"] = use_inclusions != "no"

    # Poisson analysis (only if inclusions enabled)
    if settings["inclusions"]:
        use_poisson = input(f"Use Poisson analysis? [yes/no] ({poi_default}): ").strip().lower()
        if use_poisson:
            settings["poisson"] = use_poisson != "no"
    else:
        settings["poisson"] = False

    if settings["poisson"]:
        # Bead count
        count_input = input(f"Stock count/uL [{settings['count']:.2g}]: ").strip()
        if count_input:
            try:
                settings["count"] = float(count_input)
            except ValueError:
                print(f"  Invalid value, using default: {settings['count']}")

        # Dilution
        dilution_input = input(f"Dilution factor [{settings['dilution']}]: ").strip()
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
        "-e",
        "--edit",
        action="store_true",
        help="Open interactive editor/viewer",
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

    # Get settings via interactive prompts
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
    print(f"Inclusions: {'ON' if settings['inclusions'] else 'OFF'}")
    if settings["inclusions"] and settings["poisson"]:
        print(f"Poisson: ON (count={settings['count']:.2e}, dilution={settings['dilution']})")
    elif settings["inclusions"]:
        print("Poisson: OFF")
    if args.number:
        print(f"Frame limit: {args.number}")

    # Create pipeline with visualization storage if editor is requested
    store_viz = args.edit
    use_cache = not args.no_cache
    detect_inclusions = settings["inclusions"]
    pipeline = DropletInclusionPipeline(
        store_visualizations=store_viz, use_cache=use_cache,
        detect_inclusions=detect_inclusions,
    )

    # Handle cache clear request
    if args.clear_cache and pipeline.cache:
        pipeline.cache.clear()

    results = pipeline.run(args.input_dir, str(output_dir), frame_limit=args.number)

    if results:
        print("\nPipeline completed successfully!")

        # Interactive editor
        if args.edit and pipeline.visualization_data:
            print("\nLaunching editor...")
            editor = Editor(pipeline.visualization_data, results, detect_inclusions=detect_inclusions)
            results = editor.run()

            # Save updated results
            df = pd.DataFrame(results)
            csv_path = output_dir / "data.csv"
            df.to_csv(csv_path, index=False)
            print(f"Updated results saved to: {csv_path}")

        # Always generate statistics (after any interactive corrections)
        print("\nGenerating statistical analysis...")
        csv_path = output_dir / "data.csv"

        # Extract sample frames for report (always available from pipeline)
        sample_frames = None
        if pipeline.sample_frames:
            sample_frames = []
            for idx in sorted(pipeline.sample_frames.keys()):
                viz = pipeline.sample_frames[idx]
                sample_frames.append({
                    "frame_idx": idx,
                    "image": viz["min_projection"],
                    "droplet_masks": viz.get("droplet_masks", []),
                    "inclusion_masks": viz.get("inclusion_masks", []),
                })

        stats_module = DropletStatistics(csv_path, settings)
        stats_module.run_analysis(str(output_dir), sample_frames)

        # Archive project if requested
        if args.gzip:
            archive_name = f"{output_dir}.tar.gz"
            print(f"\nArchiving project to: {archive_name}")
            with tarfile.open(archive_name, "w:gz") as tar:
                tar.add(output_dir, arcname=output_dir.name)
            print(f"Archive created: {archive_name}")


if __name__ == "__main__":
    main()
