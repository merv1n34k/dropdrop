"""Command-line interface for DropDrop pipeline."""

import argparse
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from .analysis import Analysis
from .config import load_config
from .detection import Detection
from .ui import Editor

QUEUE_FILE = "queue.tmp"


# --- Settings ---

def prompt_settings(config=None, skip_label=False):
    """Interactive prompts for project settings."""
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

    use_inclusions = input(f"Detect inclusions? [yes/no] ({inc_default}): ").strip().lower()
    if use_inclusions:
        settings["inclusions"] = use_inclusions != "no"

    if settings["inclusions"]:
        use_poisson = input(f"Use Poisson analysis? [yes/no] ({poi_default}): ").strip().lower()
        if use_poisson:
            settings["poisson"] = use_poisson != "no"
    else:
        settings["poisson"] = False

    if settings["poisson"]:
        count_input = input(f"Stock count/uL [{settings['count']:.2g}]: ").strip()
        if count_input:
            try:
                settings["count"] = float(count_input)
            except ValueError:
                print(f"  Invalid value, using default: {settings['count']}")

        dilution_input = input(f"Dilution factor [{settings['dilution']}]: ").strip()
        if dilution_input:
            try:
                settings["dilution"] = int(dilution_input)
            except ValueError:
                print(f"  Invalid value, using default: {settings['dilution']}")

    if not skip_label:
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


# --- Queue management ---

def discover_subdirectories(parent_dir):
    """List subdirectories and prompt for labels. Skip if Enter is empty."""
    parent = Path(parent_dir)
    subdirs = sorted([d for d in parent.iterdir() if d.is_dir()])

    if not subdirs:
        print(f"ERROR: No subdirectories found in '{parent_dir}'")
        return []

    print(f"\nFound {len(subdirs)} directories in {parent_dir}:")
    queue = []
    for d in subdirs:
        label = input(f"  {d.name} -> label (Enter to skip): ").strip()
        if label:
            queue.append({"path": str(d.resolve()), "label": label, "status": "pending"})

    if not queue:
        print("No directories selected.")
    return queue


def write_queue(queue):
    """Write queue to queue.tmp file."""
    with open(QUEUE_FILE, "w") as f:
        for entry in queue:
            f.write(f"{entry['path']}\t{entry['label']}\t{entry['status']}\n")


def read_queue():
    """Read queue from queue.tmp file."""
    queue_path = Path(QUEUE_FILE)
    if not queue_path.exists():
        return None

    queue = []
    for line in queue_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            queue.append({"path": parts[0], "label": parts[1], "status": parts[2]})
    return queue


def update_queue_status(queue, index):
    """Mark queue entry as done and rewrite file."""
    queue[index]["status"] = "done"
    write_queue(queue)


def cleanup_queue():
    """Remove queue.tmp file."""
    queue_path = Path(QUEUE_FILE)
    if queue_path.exists():
        queue_path.unlink()
        print(f"Queue file removed: {QUEUE_FILE}")


# --- Pipeline ---

def run_detection(input_dir, tmp_dir, settings, args, cellpose_model=None):
    """Run detection on a single input directory into a .tmp dir.

    Returns the cellpose model for reuse across runs.
    """
    pipeline = Detection(
        store_visualizations=args.edit, use_cache=not args.no_cache,
        detect_inclusions=settings["inclusions"],
    )

    if cellpose_model is not None:
        pipeline._cellpose_model = cellpose_model
    if args.clear_cache and pipeline.cache:
        pipeline.cache.clear()

    tmp_dir = Path(tmp_dir)
    results = pipeline.run(input_dir, str(tmp_dir), frame_limit=args.number)

    if results:
        print("\nPipeline completed successfully!")

        if args.edit and pipeline.visualization_data:
            print("\nLaunching editor...")
            editor = Editor(pipeline.visualization_data, results,
                            detect_inclusions=settings["inclusions"])
            results = editor.run()
            df = pd.DataFrame(results)
            df.to_csv(tmp_dir / "data.csv", index=False)

    return pipeline._cellpose_model


def finalize(output_dir, settings, gzip=False):
    """Run analysis, clean up .tmp dirs, and optionally archive."""
    output_dir = Path(output_dir)

    print("\nGenerating statistical analysis...")
    Analysis(str(output_dir), settings).run()

    for tmp_dir in output_dir.glob(".tmp_*"):
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)

    if gzip and output_dir.exists():
        archive_name = f"{output_dir}.tar.gz"
        print(f"\nArchiving to: {archive_name}")
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)
        print(f"Archive created: {archive_name}")


def process_queue(queue, output_dir, settings, args):
    """Process all pending queue entries, then finalize."""
    pending = [(i, e) for i, e in enumerate(queue) if e["status"] == "pending"]

    if not pending:
        print("No pending directories to process.")
        cleanup_queue()
        return

    print(f"\nProcessing {len(pending)} directories...")
    cellpose_model = None

    for i, entry in pending:
        label = entry["label"]
        run_settings = settings.copy()
        run_settings["label"] = label
        run_settings["input_dir"] = str(Path(entry["path"]).resolve())

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(queue)}] {Path(entry['path']).name} -> {label}")
        print(f"{'='*60}")

        cellpose_model = run_detection(
            entry["path"], output_dir / f".tmp_{label}",
            run_settings, args, cellpose_model,
        )
        update_queue_status(queue, i)

    cleanup_queue()
    print(f"\nAll {len(pending)} directories processed.")
    finalize(output_dir, settings, gzip=args.gzip)


# --- Entry point ---

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Droplet and inclusion detection pipeline using Cellpose"
    )
    parser.add_argument("input_dir", type=str, nargs="?", default=None,
                        help="Input directory containing z-stack images")
    parser.add_argument("output_dir", type=str, nargs="?", default=None,
                        help="Output directory (default: ./results/<date>_<label>)")
    parser.add_argument("-m", "--multiplex", type=str, default=None,
                        help="Parent directory for batch processing")
    parser.add_argument("-r", "--resurrect", action="store_true",
                        help=f"Resume from existing {QUEUE_FILE}")
    parser.add_argument("-e", "--edit", action="store_true",
                        help="Open interactive editor/viewer")
    parser.add_argument("-n", "--number", type=int, default=None,
                        help="Process only first N frames")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching for this run")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cache before processing")
    parser.add_argument("-z", "--gzip", action="store_true",
                        help="Archive project directory as .tar.gz")
    args = parser.parse_args()

    # Validate
    if args.resurrect and (args.input_dir or args.multiplex):
        print("ERROR: --resurrect cannot be used with input_dir or --multiplex")
        sys.exit(1)
    if args.multiplex and args.input_dir:
        print("ERROR: Cannot use both input_dir and --multiplex")
        sys.exit(1)
    if not args.resurrect and not args.multiplex and not args.input_dir:
        parser.print_help()
        sys.exit(1)

    # --- Multiplex / Resurrect ---
    if args.resurrect or args.multiplex:
        if args.resurrect:
            queue = read_queue()
            if queue is None:
                print(f"ERROR: No {QUEUE_FILE} found. Nothing to resume.")
                sys.exit(1)
            pending_count = sum(1 for e in queue if e["status"] == "pending")
            print(f"Resuming from {QUEUE_FILE}: {pending_count}/{len(queue)} pending")
        else:
            if not Path(args.multiplex).exists():
                print(f"ERROR: Directory '{args.multiplex}' does not exist")
                sys.exit(1)
            queue = discover_subdirectories(args.multiplex)
            if not queue:
                return
            write_queue(queue)
            print(f"\nQueue saved to {QUEUE_FILE} ({len(queue)} entries)")

        settings = prompt_settings(skip_label=True)
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = Path("results") / f"{date_str}_multiplex"
        process_queue(queue, output_dir, settings, args)
        return

    # --- Single directory ---
    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    settings = prompt_settings()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / generate_project_name(settings)

    settings["input_dir"] = str(Path(args.input_dir).resolve())
    label = settings.get("label") or "data"

    print(f"Input: {args.input_dir}")
    print(f"Output: {output_dir}")
    print(f"Inclusions: {'ON' if settings['inclusions'] else 'OFF'}")
    if settings["inclusions"] and settings["poisson"]:
        print(f"Poisson: ON (count={settings['count']:.2e}, dilution={settings['dilution']})")
    elif settings["inclusions"]:
        print("Poisson: OFF")
    if args.number:
        print(f"Frame limit: {args.number}")

    run_detection(args.input_dir, output_dir / f".tmp_{label}", settings, args)
    finalize(output_dir, settings, gzip=args.gzip)


if __name__ == "__main__":
    main()
