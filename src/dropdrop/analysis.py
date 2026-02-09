"""Statistical analysis for droplet detection results."""

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300


class Analysis:
    """Statistical analysis — auto-discovers .tmp_* dirs in output directory.

    Single sample  → individual report with sample frames
    Multiple samples → multiplex report with overlaid plots and collage
    """

    def __init__(self, output_dir, settings):
        """Initialize by scanning output_dir for .tmp_* sample directories.

        Args:
            output_dir: Path to output directory containing .tmp_<label>/ dirs.
            settings: Dict with analysis settings (count, dilution, poisson, inclusions).
        """
        self.output_dir = Path(output_dir)
        self.settings = settings
        self.use_inclusions = settings.get("inclusions", True)
        self.use_poisson = settings.get("poisson", True) and self.use_inclusions
        self.bead_count = settings.get("count", 6.5e5)
        self.dilution = settings.get("dilution", 1000)

        # Auto-discover .tmp_* directories
        self.samples = []
        for tmp_dir in sorted(self.output_dir.glob(".tmp_*")):
            label = tmp_dir.name[5:]  # strip ".tmp_"
            csv_path = tmp_dir / "data.csv"
            if csv_path.exists():
                self.samples.append({
                    "label": label,
                    "df": pd.read_csv(csv_path),
                    "sample_pngs": sorted(tmp_dir.glob("sample_*.png")),
                })

    def run(self):
        """Run analysis — auto-detects single vs multiplex."""
        if not self.samples:
            print("No samples found.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_stats = [self._compute_sample_stats(s) for s in self.samples]

        if len(self.samples) == 1:
            self._run_single(all_stats[0])
        else:
            self._run_multiplex(all_stats)

    # --- Shared methods ---

    def _compute_sample_stats(self, sample):
        """Compute statistics for a single sample."""
        df = sample["df"]
        mean_d = df["diameter_um"].mean()
        median_d = df["diameter_um"].median()
        std_d = df["diameter_um"].std()
        cv = (std_d / mean_d * 100) if mean_d > 0 else 0
        total_droplets = len(df)
        total_inclusions = int(df["inclusions"].sum()) if self.use_inclusions else 0
        with_inclusions = int((df["inclusions"] > 0).sum()) if self.use_inclusions else 0

        sample_stats = {
            "label": sample["label"],
            "mean_d": mean_d,
            "median_d": median_d,
            "std_d": std_d,
            "cv": cv,
            "total_droplets": total_droplets,
            "total_inclusions": total_inclusions,
            "with_inclusions": with_inclusions,
            "lambda_val": None,
            "chi2": None,
            "p_value": None,
        }

        if self.use_poisson:
            x_range, theoretical, lambda_val = self._calculate_poisson(df, median_d)
            actual = df["inclusions"].value_counts().sort_index()
            chi2, p_value = self._chi_squared(actual, theoretical, total_droplets)
            sample_stats["lambda_val"] = lambda_val
            sample_stats["chi2"] = chi2
            sample_stats["p_value"] = p_value

        return sample_stats

    def _calculate_poisson(self, df, median_diameter_um):
        """Calculate theoretical Poisson distribution."""
        radius_um = median_diameter_um / 2
        volume_ml = (4 / 3) * np.pi * (radius_um**3) * 1e-9
        lambda_val = (self.bead_count / (self.dilution * 2)) * volume_ml
        max_inc = int(df["inclusions"].max()) + 3
        x_range = np.arange(0, max_inc + 1)
        theoretical = stats.poisson.pmf(x_range, lambda_val)
        return x_range, theoretical, lambda_val

    def _chi_squared(self, observed_counts, theoretical_probs, n_total):
        """Perform chi-squared goodness-of-fit test."""
        observed, expected = [], []
        for i in observed_counts.index:
            if i < len(theoretical_probs):
                observed.append(observed_counts[i])
                expected.append(theoretical_probs[i] * n_total)

        observed = np.array(observed)
        expected = np.array(expected)

        mask = expected >= 5
        if mask.sum() < 2:
            return None, None

        obs_f = observed[mask]
        exp_f = expected[mask]
        exp_f = exp_f * (obs_f.sum() / exp_f.sum())
        chi2, p_value = stats.chisquare(obs_f, exp_f)
        return chi2, p_value

    def _load_sample_pngs(self, sample):
        """Load sample frame PNGs as RGB arrays."""
        images = []
        for png_path in sample["sample_pngs"]:
            img = cv2.imread(str(png_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_idx = png_path.stem.split("_")[1]
                images.append({"image": img, "frame_idx": frame_idx})
        return images

    # --- Single sample path ---

    def _run_single(self, sample_stats):
        """Generate single-sample report."""
        sample = self.samples[0]
        df = sample["df"]

        # Copy data.csv to output dir
        shutil.copy2(
            self.output_dir / f".tmp_{sample['label']}" / "data.csv",
            self.output_dir / "data.csv",
        )

        # Standalone plots
        self._plot_size_distribution(df, sample_stats)
        if self.use_poisson:
            self._plot_poisson_comparison(df, sample_stats)

        # Combined report with sample frames
        sample_pngs = self._load_sample_pngs(sample)
        self._create_single_report(sample_stats, sample_pngs)

        # Summary text
        self._write_single_summary(sample_stats)

        # Console output
        self._print_single_summary(sample_stats)

    def _plot_size_distribution(self, df, sample_stats):
        """Plot droplet diameter distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        diameters = df["diameter_um"].values
        ax.hist(diameters, bins=25, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(sample_stats["mean_d"], color="red", linestyle="--",
                    label=f"Mean: {sample_stats['mean_d']:.1f}")
        ax.axvline(sample_stats["median_d"], color="green", linestyle="--",
                    label=f"Median: {sample_stats['median_d']:.1f}")
        ax.set_xlabel("Diameter (um)")
        ax.set_ylabel("Count")
        ax.set_title("Droplet Size Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "size_distribution.png", dpi=200)
        plt.close()

    def _plot_poisson_comparison(self, df, sample_stats):
        """Plot detected vs theoretical Poisson."""
        fig, ax = plt.subplots(figsize=(10, 6))
        median_d = df["diameter_um"].median()
        x_range, theoretical, lambda_val = self._calculate_poisson(df, median_d)
        actual = df["inclusions"].value_counts().sort_index()
        n_droplets = len(df)

        detected_pct = [actual.get(i, 0) / n_droplets * 100 for i in x_range]
        theoretical_pct = theoretical * 100

        x = np.arange(len(x_range))
        width = 0.35
        ax.bar(x - width / 2, detected_pct, width,
               label="Detected", color="royalblue", alpha=0.8)
        ax.bar(x + width / 2, theoretical_pct[:len(x)], width,
               label=f"Poisson (l={lambda_val:.3f})", color="coral", alpha=0.8)

        if sample_stats.get("p_value") is not None:
            result_text = f"X2 = {sample_stats['chi2']:.2f}, p = {sample_stats['p_value']:.4f}"
            result_text += "\nFollows Poisson" if sample_stats["p_value"] > 0.05 else "\nDeviates from Poisson"
            ax.text(0.98, 0.85, result_text, transform=ax.transAxes,
                    ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("Inclusions per Droplet")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Inclusion Distribution: Detected vs Theoretical")
        ax.set_xticks(x)
        ax.set_xticklabels(x_range)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(self.output_dir / "poisson_comparison.png", dpi=200)
        plt.close()

    def _create_single_report(self, sample_stats, sample_pngs):
        """Create combined report.png with plots, stats, and sample frames."""
        df = self.samples[0]["df"]
        n_frames = len(sample_pngs)

        if self.use_poisson:
            n_cols = max(3, n_frames)
            fig = plt.figure(figsize=(5 * n_cols, 10))
            gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 1])
            ax_size = fig.add_subplot(gs[0, 0])
            ax_poisson = fig.add_subplot(gs[0, 1])
            ax_stats = fig.add_subplot(gs[0, 2])
        else:
            n_cols = max(2, n_frames)
            fig = plt.figure(figsize=(5 * n_cols, 10))
            gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 1])
            ax_size = fig.add_subplot(gs[0, 0])
            ax_stats = fig.add_subplot(gs[0, 1])
            ax_poisson = None

        # Size distribution
        diameters = df["diameter_um"].values
        ax_size.hist(diameters, bins=25, color="steelblue", edgecolor="black", alpha=0.7)
        ax_size.axvline(sample_stats["mean_d"], color="red", linestyle="--",
                        label=f"Mean: {sample_stats['mean_d']:.1f}")
        ax_size.axvline(sample_stats["median_d"], color="green", linestyle="--",
                        label=f"Median: {sample_stats['median_d']:.1f}")
        ax_size.set_xlabel("Diameter (um)")
        ax_size.set_ylabel("Count")
        ax_size.set_title("Droplet Size Distribution")
        ax_size.legend()
        ax_size.grid(True, alpha=0.3)

        # Poisson comparison
        if ax_poisson is not None and sample_stats.get("lambda_val") is not None:
            median_d = df["diameter_um"].median()
            x_range, theoretical, lambda_val = self._calculate_poisson(df, median_d)
            actual = df["inclusions"].value_counts().sort_index()
            n_droplets = len(df)
            detected_pct = [actual.get(i, 0) / n_droplets * 100 for i in x_range]
            theoretical_pct = theoretical * 100

            x = np.arange(len(x_range))
            width = 0.35
            ax_poisson.bar(x - width / 2, detected_pct, width,
                           label="Detected", color="royalblue", alpha=0.8)
            ax_poisson.bar(x + width / 2, theoretical_pct[:len(x)], width,
                           label=f"Poisson (l={lambda_val:.3f})", color="coral", alpha=0.8)
            if sample_stats.get("p_value") is not None:
                result_text = f"X2 = {sample_stats['chi2']:.2f}, p = {sample_stats['p_value']:.4f}"
                result_text += "\nFollows Poisson" if sample_stats["p_value"] > 0.05 else "\nDeviates"
                ax_poisson.text(0.98, 0.85, result_text, transform=ax_poisson.transAxes,
                                ha="right", va="top", fontsize=10,
                                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
            ax_poisson.set_xlabel("Inclusions per Droplet")
            ax_poisson.set_ylabel("Percentage (%)")
            ax_poisson.set_title("Inclusion Distribution")
            ax_poisson.set_xticks(x)
            ax_poisson.set_xticklabels(x_range)
            ax_poisson.legend()
            ax_poisson.grid(True, alpha=0.3, axis="y")

        # Stats text box
        project_name = self.output_dir.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        total_frames = df["frame"].nunique()
        total_droplets = len(df)

        stats_lines = [
            f"Project: {project_name}",
            f"Date: {timestamp}",
            f"Frames: {total_frames}",
            "",
            f"Droplets: {total_droplets:,}",
        ]

        if self.use_inclusions:
            stats_lines.extend([
                f"Inclusions: {sample_stats['total_inclusions']:,}",
                f"Mean/droplet: {sample_stats['total_inclusions'] / total_droplets:.2f}",
                f"With incl: {sample_stats['with_inclusions'] / total_droplets * 100:.1f}%",
            ])
        else:
            stats_lines.append("Inclusions: OFF")

        stats_lines.extend([
            "",
            f"Diameter: {sample_stats['mean_d']:.1f} +/- {sample_stats['std_d']:.1f} um",
            f"CV (diameter): {sample_stats['cv']:.1f}%",
        ])

        if self.use_poisson and sample_stats.get("lambda_val") is not None:
            stats_lines.extend([
                "",
                f"Dilution: {self.dilution}x",
                f"l theoretical: {sample_stats['lambda_val']:.4f}",
            ])
            if sample_stats.get("p_value") is not None:
                result = "FOLLOWS" if sample_stats["p_value"] > 0.05 else "DEVIATES"
                stats_lines.append(f"Result: {result} Poisson")

        ax_stats.axis("off")
        ax_stats.text(0.1, 0.95, "\n".join(stats_lines), transform=ax_stats.transAxes,
                      fontsize=11, verticalalignment="top", fontfamily="monospace",
                      bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3))
        ax_stats.set_title("Summary")

        # Sample frames (bottom row)
        for i, frame in enumerate(sample_pngs[:n_cols]):
            ax_sample = fig.add_subplot(gs[1, i])
            ax_sample.imshow(frame["image"])
            ax_sample.set_title(f"Frame {frame['frame_idx']}")
            ax_sample.axis("off")

        plt.suptitle("DropDrop Analysis Report", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "report.png", dpi=200, bbox_inches="tight")
        plt.close()

    def _write_single_summary(self, s):
        """Write summary.txt for single sample."""
        project_name = self.output_dir.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        input_dir = self.settings.get("input_dir", "N/A")
        total_frames = self.samples[0]["df"]["frame"].nunique()

        lines = [
            "=" * 80,
            "DROPDROP ANALYSIS SUMMARY".center(80),
            "=" * 80,
            "",
            f"Project: {project_name}",
            f"Date: {timestamp}",
            f"Input: {input_dir} ({total_frames} frames)",
            "",
            "SETTINGS",
            "-" * 40,
            f"Inclusion Detection: {'ON' if self.use_inclusions else 'OFF'}",
            f"Poisson Analysis: {'ON' if self.use_poisson else 'OFF'}",
        ]

        if self.use_poisson:
            lines.extend([
                f"Stock Concentration: {self.bead_count:.2e} beads/uL",
                f"Dilution Factor: {self.dilution}x",
            ])

        lines.extend([
            "",
            "RESULTS",
            "-" * 40,
            f"Total Frames Processed: {total_frames}",
            f"Total Droplets Detected: {s['total_droplets']:,}",
            "",
            "Droplet Statistics:",
            f"  Mean Diameter: {s['mean_d']:.1f} um",
            f"  Median Diameter: {s['median_d']:.1f} um",
            f"  Std Deviation: {s['std_d']:.1f} um",
            f"  CV: {s['cv']:.1f}%",
        ])

        if self.use_inclusions:
            lines.extend([
                "",
                "Bead Statistics:",
                f"  Total Beads Detected: {s['total_inclusions']:,}",
                f"  Mean per Droplet: {s['total_inclusions'] / s['total_droplets']:.2f}",
                f"  Droplets with Beads: {s['with_inclusions']} ({s['with_inclusions'] / s['total_droplets'] * 100:.1f}%)",
            ])

        if self.use_poisson and s.get("lambda_val") is not None:
            lines.extend([
                "",
                "POISSON ANALYSIS",
                "-" * 40,
                f"Theoretical Lambda: {s['lambda_val']:.3f}",
            ])
            if s.get("p_value") is not None:
                result = "FOLLOWS" if s["p_value"] > 0.05 else "DEVIATES FROM"
                lines.extend([
                    f"Chi-squared: {s['chi2']:.2f}",
                    f"P-value: {s['p_value']:.4f}",
                    f"Result: Distribution {result} Poisson (p {'>' if s['p_value'] > 0.05 else '<'} 0.05)",
                ])

        lines.extend(["", "=" * 80, "Generated by DropDrop", "=" * 80])

        with open(self.output_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines))

    def _print_single_summary(self, s):
        """Print single-sample summary to console."""
        print("\nSTATISTICAL SUMMARY")
        print("-" * 40)
        print(f"Droplets: {s['total_droplets']}")
        print(f"Mean diameter: {s['mean_d']:.1f} um (CV: {s['cv']:.1f}%)")

        if self.use_inclusions:
            print(f"Inclusions: {s['total_inclusions']} total, "
                  f"{s['total_inclusions'] / s['total_droplets']:.2f} per droplet")
            print(f"With inclusions: {s['with_inclusions']} "
                  f"({s['with_inclusions'] / s['total_droplets'] * 100:.1f}%)")
        else:
            print("Inclusions: OFF")

        if self.use_poisson and s.get("lambda_val") is not None:
            print(f"Theoretical l: {s['lambda_val']:.3f}")
            if s.get("p_value") is not None:
                print(f"\nChi-squared test:")
                print(f"  X2 = {s['chi2']:.2f}, p = {s['p_value']:.4f}")
                if s["p_value"] > 0.05:
                    print("  -> Distribution follows Poisson (p > 0.05)")
                else:
                    print("  -> Distribution deviates from Poisson (p < 0.05)")

        print(f"\nOutput saved to: {self.output_dir}")

    # --- Multiplex path ---

    def _run_multiplex(self, all_stats):
        """Generate multiplex report."""
        axes_limits = self._compute_global_axes()

        # Merged CSV
        merged_df = self._merge_dataframes()
        merged_df.to_csv(self.output_dir / "data.csv", index=False)

        # Overlaid plots
        self._plot_overlaid_size_distribution(axes_limits)
        if self.use_poisson:
            self._plot_overlaid_poisson(axes_limits)

        # Summary report with sample frame collage
        self._create_summary_report(all_stats, axes_limits)
        self._write_multiplex_summary(all_stats)
        self._print_multiplex_summary(all_stats)

    def _compute_global_axes(self):
        """Compute shared axis limits across all samples."""
        all_diameters = pd.concat([s["df"]["diameter_um"] for s in self.samples])
        d_min = all_diameters.min()
        d_max = all_diameters.max()
        margin = (d_max - d_min) * 0.05

        bins = np.linspace(d_min - margin, d_max + margin, 26)

        y_max = 0
        for s in self.samples:
            counts, _ = np.histogram(s["df"]["diameter_um"], bins=bins)
            y_max = max(y_max, counts.max())

        poisson_x_max = 0
        poisson_y_max = 0
        if self.use_poisson:
            for s in self.samples:
                max_inc = int(s["df"]["inclusions"].max())
                poisson_x_max = max(poisson_x_max, max_inc + 3)
                actual = s["df"]["inclusions"].value_counts().sort_index()
                for val in actual.values:
                    pct = val / len(s["df"]) * 100
                    poisson_y_max = max(poisson_y_max, pct)

        return {
            "bins": bins,
            "diameter_y_max": int(y_max * 1.15),
            "poisson_x_max": poisson_x_max,
            "poisson_y_max": poisson_y_max * 1.15,
        }

    def _merge_dataframes(self):
        """Merge all sample DataFrames with 'sample' column."""
        dfs = []
        for s in self.samples:
            df_copy = s["df"].copy()
            df_copy.insert(0, "sample", s["label"])
            dfs.append(df_copy)
        return pd.concat(dfs, ignore_index=True)

    def _plot_overlaid_size_distribution(self, axes_limits):
        """Plot overlaid size distributions from all samples."""
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = axes_limits["bins"]
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.samples)))

        for i, sample in enumerate(self.samples):
            ax.hist(sample["df"]["diameter_um"].values, bins=bins,
                    color=colors[i], edgecolor="black", alpha=0.5, label=sample["label"])

        ax.set_xlabel("Diameter (um)")
        ax.set_ylabel("Count")
        ax.set_title("Droplet Size Distribution (All Samples)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "size_distribution.png", dpi=200)
        plt.close()

    def _plot_overlaid_poisson(self, axes_limits):
        """Plot overlaid Poisson comparisons from all samples."""
        if not self.use_poisson:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.samples)))
        x_range = np.arange(0, axes_limits["poisson_x_max"] + 1)
        n_s = len(self.samples)
        bar_width = 0.8 / n_s

        for i, sample in enumerate(self.samples):
            actual = sample["df"]["inclusions"].value_counts().sort_index()
            n_drop = len(sample["df"])
            detected_pct = [actual.get(k, 0) / n_drop * 100 for k in x_range]
            offset = (i - n_s / 2 + 0.5) * bar_width
            ax.bar(x_range + offset, detected_pct, bar_width,
                   label=sample["label"], color=colors[i], alpha=0.8)

        ax.set_xlabel("Inclusions per Droplet")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Inclusion Distribution (All Samples)")
        ax.set_xticks(x_range)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(self.output_dir / "poisson_comparison.png", dpi=200)
        plt.close()

    def _create_summary_report(self, all_stats, axes_limits):
        """Create summary_report.png with table, plots, CV barplot, and sample collage."""
        n_samples = len(self.samples)
        colors = plt.cm.Set2(np.linspace(0, 1, n_samples))

        # Layout: 3 rows — [table, size_dist] [CV, poisson] [sample collage]
        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])

        # [0,0]: Comparison table
        ax_table = fig.add_subplot(gs[0, 0])
        ax_table.axis("off")
        columns = [s["label"] for s in all_stats]
        rows = ["Droplets", "Mean (um)", "Median (um)", "Std (um)", "CV (%)"]
        if self.use_inclusions:
            rows.extend(["Inclusions", "Mean/droplet"])

        cell_data = []
        for row_name in rows:
            row_vals = []
            for s in all_stats:
                if row_name == "Droplets":
                    row_vals.append(f"{s['total_droplets']:,}")
                elif row_name == "Mean (um)":
                    row_vals.append(f"{s['mean_d']:.1f}")
                elif row_name == "Median (um)":
                    row_vals.append(f"{s['median_d']:.1f}")
                elif row_name == "Std (um)":
                    row_vals.append(f"{s['std_d']:.1f}")
                elif row_name == "CV (%)":
                    row_vals.append(f"{s['cv']:.1f}")
                elif row_name == "Inclusions":
                    row_vals.append(f"{s['total_inclusions']:,}")
                elif row_name == "Mean/droplet":
                    mean_inc = s["total_inclusions"] / s["total_droplets"] if s["total_droplets"] > 0 else 0
                    row_vals.append(f"{mean_inc:.2f}")
            cell_data.append(row_vals)

        table = ax_table.table(cellText=cell_data, rowLabels=rows, colLabels=columns,
                               loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax_table.set_title("Sample Comparison", fontweight="bold")

        # [0,1]: Overlaid size distribution
        ax_size = fig.add_subplot(gs[0, 1])
        bins = axes_limits["bins"]
        for i, sample in enumerate(self.samples):
            ax_size.hist(sample["df"]["diameter_um"].values, bins=bins,
                         color=colors[i], edgecolor="black", alpha=0.5, label=sample["label"])
        ax_size.set_xlabel("Diameter (um)")
        ax_size.set_ylabel("Count")
        ax_size.set_title("Size Distribution")
        ax_size.legend()
        ax_size.grid(True, alpha=0.3)

        # [1,0]: CV barplot
        ax_cv = fig.add_subplot(gs[1, 0])
        labels = [s["label"] for s in all_stats]
        cvs = [s["cv"] for s in all_stats]
        ax_cv.bar(labels, cvs, color=colors[:len(labels)], edgecolor="black", alpha=0.8)
        ax_cv.set_ylabel("CV (%)")
        ax_cv.set_title("Coefficient of Variation")
        ax_cv.grid(True, alpha=0.3, axis="y")
        for j, v in enumerate(cvs):
            ax_cv.text(j, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

        # [1,1]: Overlaid Poisson or placeholder
        ax_poisson = fig.add_subplot(gs[1, 1])
        if self.use_poisson:
            x_range = np.arange(0, axes_limits["poisson_x_max"] + 1)
            n_s = len(self.samples)
            bw = 0.8 / n_s
            for i, sample in enumerate(self.samples):
                actual = sample["df"]["inclusions"].value_counts().sort_index()
                n_drop = len(sample["df"])
                detected_pct = [actual.get(k, 0) / n_drop * 100 for k in x_range]
                offset = (i - n_s / 2 + 0.5) * bw
                ax_poisson.bar(x_range + offset, detected_pct, bw,
                               label=sample["label"], color=colors[i], alpha=0.8)
            ax_poisson.set_xlabel("Inclusions per Droplet")
            ax_poisson.set_ylabel("Percentage (%)")
            ax_poisson.set_title("Inclusion Distribution")
            ax_poisson.set_xticks(x_range)
            ax_poisson.legend()
            ax_poisson.grid(True, alpha=0.3, axis="y")
        else:
            ax_poisson.axis("off")
            ax_poisson.text(0.5, 0.5, "Poisson: OFF", ha="center", va="center",
                            fontsize=14, transform=ax_poisson.transAxes)

        # [2, :]: Sample frame collage — one representative per sample
        collage_gs = gs[2, :].subgridspec(1, n_samples)
        for i, sample in enumerate(self.samples):
            ax_frame = fig.add_subplot(collage_gs[0, i])
            pngs = self._load_sample_pngs(sample)
            if pngs:
                ax_frame.imshow(pngs[0]["image"])
                ax_frame.set_title(f"{sample['label']} (frame {pngs[0]['frame_idx']})")
            else:
                ax_frame.text(0.5, 0.5, "No samples", ha="center", va="center",
                              transform=ax_frame.transAxes)
                ax_frame.set_title(sample["label"])
            ax_frame.axis("off")

        plt.suptitle("DropDrop Multiplex Report", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_report.png", dpi=200, bbox_inches="tight")
        plt.close()

    def _write_multiplex_summary(self, all_stats):
        """Write merged summary.txt."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 80,
            "DROPDROP MULTIPLEX ANALYSIS SUMMARY".center(80),
            "=" * 80,
            "",
            f"Date: {timestamp}",
            f"Samples: {len(self.samples)}",
            f"Inclusions: {'ON' if self.use_inclusions else 'OFF'}",
            f"Poisson: {'ON' if self.use_poisson else 'OFF'}",
            "",
        ]

        for s in all_stats:
            lines.extend([
                f"--- {s['label']} ---",
                f"  Droplets: {s['total_droplets']:,}",
                f"  Mean Diameter: {s['mean_d']:.1f} um",
                f"  Median Diameter: {s['median_d']:.1f} um",
                f"  Std Deviation: {s['std_d']:.1f} um",
                f"  CV: {s['cv']:.1f}%",
            ])
            if self.use_inclusions:
                mean_inc = s["total_inclusions"] / s["total_droplets"] if s["total_droplets"] > 0 else 0
                lines.extend([
                    f"  Inclusions: {s['total_inclusions']:,}",
                    f"  Mean/Droplet: {mean_inc:.2f}",
                ])
            if self.use_poisson and s.get("lambda_val") is not None:
                lines.append(f"  Lambda: {s['lambda_val']:.4f}")
                if s.get("p_value") is not None:
                    result = "FOLLOWS" if s["p_value"] > 0.05 else "DEVIATES FROM"
                    lines.append(f"  Chi-squared: {s['chi2']:.2f}, p={s['p_value']:.4f} -> {result} Poisson")
            lines.append("")

        lines.extend(["=" * 80, "Generated by DropDrop (Multiplex Mode)", "=" * 80])

        with open(self.output_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines))

    def _print_multiplex_summary(self, all_stats):
        """Print multiplex summary to console."""
        print("\nMULTIPLEX SUMMARY")
        print("=" * 50)
        for s in all_stats:
            line = f"  {s['label']}: {s['total_droplets']} droplets, mean={s['mean_d']:.1f}um, CV={s['cv']:.1f}%"
            if self.use_inclusions:
                line += f", inclusions={s['total_inclusions']}"
            print(line)
        print(f"\nOutput: {self.output_dir}")
