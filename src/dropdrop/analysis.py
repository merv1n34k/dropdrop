"""Statistical analysis for droplet detection results."""

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


class DropletStatistics:
    """Statistical analysis for droplet detection."""

    def __init__(self, results_csv, settings=None):
        """Initialize with results data and optional settings.

        Args:
            results_csv: Path to CSV file with detection results.
            settings: Dict with 'count', 'dilution', 'poisson' keys.
        """
        self.df = pd.read_csv(results_csv)
        self.settings = settings or {}

        self.bead_count = self.settings.get("count", 6.5e5)
        self.dilution = self.settings.get("dilution", 1000)
        self.use_inclusions = self.settings.get("inclusions", True)
        self.use_poisson = self.settings.get("poisson", True) and self.use_inclusions

    def calculate_poisson(self, median_diameter_um):
        """Calculate theoretical Poisson distribution."""
        radius_um = median_diameter_um / 2
        volume_ml = (4 / 3) * np.pi * (radius_um**3) * 1e-9

        lambda_val = (self.bead_count / (self.dilution * 2)) * volume_ml

        max_inc = int(self.df["inclusions"].max()) + 3
        x_range = np.arange(0, max_inc + 1)
        theoretical = stats.poisson.pmf(x_range, lambda_val)

        return x_range, theoretical, lambda_val

    def plot_size_distribution(self, output_path):
        """Plot droplet diameter distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))

        diameters = self.df["diameter_um"].values
        ax.hist(diameters, bins=25, color="steelblue", edgecolor="black", alpha=0.7)

        mean_d = np.mean(diameters)
        median_d = np.median(diameters)

        ax.axvline(mean_d, color="red", linestyle="--", label=f"Mean: {mean_d:.1f}")
        ax.axvline(
            median_d, color="green", linestyle="--", label=f"Median: {median_d:.1f}"
        )

        ax.set_xlabel("Diameter (µm)")
        ax.set_ylabel("Count")
        ax.set_title("Droplet Size Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "size_distribution.png", dpi=200)
        plt.close()

        return mean_d, median_d

    def plot_poisson_comparison(self, output_path):
        """Plot detected vs theoretical Poisson with chi-squared test."""
        fig, ax = plt.subplots(figsize=(10, 6))

        median_d = self.df["diameter_um"].median()
        x_range, theoretical, lambda_val = self.calculate_poisson(median_d)

        actual = self.df["inclusions"].value_counts().sort_index()
        n_droplets = len(self.df)

        chi2, p_value = self.perform_chi_squared(actual, theoretical, n_droplets)

        detected_pct = []
        theoretical_pct = theoretical * 100

        for i in x_range:
            if i in actual.index:
                detected_pct.append(actual[i] / n_droplets * 100)
            else:
                detected_pct.append(0)

        x = np.arange(len(x_range))
        width = 0.35

        ax.bar(
            x - width / 2,
            detected_pct,
            width,
            label="Detected",
            color="royalblue",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            theoretical_pct[: len(x)],
            width,
            label=f"Poisson (λ={lambda_val:.3f})",
            color="coral",
            alpha=0.8,
        )

        if p_value is not None:
            result_text = f"X2 = {chi2:.2f}, p = {p_value:.4f}"
            if p_value > 0.05:
                result_text += "\nFollows Poisson"
            else:
                result_text += "\nDeviates from Poisson"
            ax.text(
                0.98,
                0.85,
                result_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        ax.set_xlabel("Inclusions per Droplet")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Inclusion Distribution: Detected vs Theoretical")
        ax.set_xticks(x)
        ax.set_xticklabels(x_range)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_path / "poisson_comparison.png", dpi=200)
        plt.close()

        return lambda_val, chi2, p_value

    def perform_chi_squared(self, observed_counts, theoretical_probs, n_total):
        """Perform chi-squared goodness-of-fit test."""
        observed = []
        expected = []

        for i in observed_counts.index:
            if i < len(theoretical_probs):
                obs = observed_counts[i]
                exp = theoretical_probs[i] * n_total
                observed.append(obs)
                expected.append(exp)

        observed = np.array(observed)
        expected = np.array(expected)

        mask = expected >= 5
        if mask.sum() < 2:
            return None, None

        observed_filtered = observed[mask]
        expected_filtered = expected[mask]

        expected_filtered = expected_filtered * (
            observed_filtered.sum() / expected_filtered.sum()
        )

        chi2, p_value = stats.chisquare(observed_filtered, expected_filtered)
        return chi2, p_value

    def create_report(self, output_path, stats_data, sample_frames=None):
        """Create combined report image with plots, stats, and sample frames.

        Args:
            output_path: Path object for output directory.
            stats_data: Dict with mean_d, median_d, std_d, lambda_val, chi2, p_value.
            sample_frames: Optional list of dicts with 'frame_idx', 'image', 'droplet_masks'.
        """
        n_samples = len(sample_frames) if sample_frames else 0

        if self.use_poisson:
            # 2 rows: [size_dist, poisson, stats] + [sample frames]
            n_cols = max(3, n_samples)
            fig = plt.figure(figsize=(5 * n_cols, 10))
            gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 1])
            ax_size = fig.add_subplot(gs[0, 0])
            ax_poisson = fig.add_subplot(gs[0, 1])
            ax_stats = fig.add_subplot(gs[0, 2])
        else:
            # 2 rows: [size_dist, stats] + [sample frames]
            n_cols = max(2, n_samples)
            fig = plt.figure(figsize=(5 * n_cols, 10))
            gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 1])
            ax_size = fig.add_subplot(gs[0, 0])
            ax_stats = fig.add_subplot(gs[0, 1])
            ax_poisson = None

        # Plot 1: Size distribution
        diameters = self.df["diameter_um"].values
        ax_size.hist(diameters, bins=25, color="steelblue", edgecolor="black", alpha=0.7)
        ax_size.axvline(
            stats_data["mean_d"], color="red", linestyle="--",
            label=f"Mean: {stats_data['mean_d']:.1f}"
        )
        ax_size.axvline(
            stats_data["median_d"], color="green", linestyle="--",
            label=f"Median: {stats_data['median_d']:.1f}"
        )
        ax_size.set_xlabel("Diameter (µm)")
        ax_size.set_ylabel("Count")
        ax_size.set_title("Droplet Size Distribution")
        ax_size.legend()
        ax_size.grid(True, alpha=0.3)

        # Plot 2: Poisson comparison (if enabled)
        if ax_poisson is not None and stats_data.get("lambda_val") is not None:
            median_d = self.df["diameter_um"].median()
            x_range, theoretical, lambda_val = self.calculate_poisson(median_d)
            actual = self.df["inclusions"].value_counts().sort_index()
            n_droplets = len(self.df)

            detected_pct = []
            theoretical_pct = theoretical * 100
            for i in x_range:
                detected_pct.append(actual.get(i, 0) / n_droplets * 100)

            x = np.arange(len(x_range))
            width = 0.35
            ax_poisson.bar(
                x - width / 2, detected_pct, width,
                label="Detected", color="royalblue", alpha=0.8
            )
            ax_poisson.bar(
                x + width / 2, theoretical_pct[:len(x)], width,
                label=f"Poisson (λ={lambda_val:.3f})", color="coral", alpha=0.8
            )

            if stats_data.get("p_value") is not None:
                result_text = f"X2 = {stats_data['chi2']:.2f}, p = {stats_data['p_value']:.4f}"
                result_text += "\nFollows Poisson" if stats_data["p_value"] > 0.05 else "\nDeviates"
                ax_poisson.text(
                    0.98, 0.85, result_text, transform=ax_poisson.transAxes,
                    ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
                )

            ax_poisson.set_xlabel("Inclusions per Droplet")
            ax_poisson.set_ylabel("Percentage (%)")
            ax_poisson.set_title("Inclusion Distribution")
            ax_poisson.set_xticks(x)
            ax_poisson.set_xticklabels(x_range)
            ax_poisson.legend()
            ax_poisson.grid(True, alpha=0.3, axis="y")

        # Stats text box
        total_droplets = len(self.df)
        total_inclusions = int(self.df["inclusions"].sum())
        with_inclusions = int((self.df["inclusions"] > 0).sum())

        project_name = output_path.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        total_frames = self.df["frame"].nunique()

        stats_lines = [
            f"Project: {project_name}",
            f"Date: {timestamp}",
            f"Frames: {total_frames}",
            "",
            f"Droplets: {total_droplets:,}",
        ]

        if self.use_inclusions:
            stats_lines.extend([
                f"Inclusions: {total_inclusions:,}",
                f"Mean/droplet: {total_inclusions / total_droplets:.2f}",
                f"With incl: {with_inclusions / total_droplets * 100:.1f}%",
            ])
        else:
            stats_lines.append("Inclusions: OFF")

        cv_diameter = (stats_data["std_d"] / stats_data["mean_d"] * 100) if stats_data["mean_d"] > 0 else 0

        stats_lines.extend([
            "",
            f"Diameter: {stats_data['mean_d']:.1f} +/- {stats_data['std_d']:.1f} um",
            f"CV (diameter): {cv_diameter:.1f}%",
        ])

        if self.use_poisson and stats_data.get("lambda_val") is not None:
            stats_lines.extend([
                "",
                f"Dilution: {self.dilution}x",
                f"λ theoretical: {stats_data['lambda_val']:.4f}",
            ])
            if stats_data.get("p_value") is not None:
                result = "FOLLOWS" if stats_data["p_value"] > 0.05 else "DEVIATES"
                stats_lines.append(f"Result: {result} Poisson")

        ax_stats.axis("off")
        ax_stats.text(
            0.1, 0.95, "\n".join(stats_lines), transform=ax_stats.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3)
        )
        ax_stats.set_title("Summary")

        # Sample frames (bottom row)
        if sample_frames:
            for i, sample in enumerate(sample_frames[:n_cols]):
                ax_sample = fig.add_subplot(gs[1, i])
                self._draw_sample_frame(ax_sample, sample)

        plt.suptitle("DropDrop Analysis Report", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "report.png", dpi=200, bbox_inches="tight")
        plt.close()

    def _draw_sample_frame(self, ax, sample):
        """Draw a sample frame with detection overlay."""
        frame_idx = sample["frame_idx"]
        image = sample["image"]
        droplet_masks = sample.get("droplet_masks", [])
        inclusion_masks = sample.get("inclusion_masks", [])

        # Convert grayscale to RGB for colored overlay
        if len(image.shape) == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display = image.copy()

        # Draw droplet contours in green
        for droplet in droplet_masks:
            mask = droplet.get("mask")
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
                # Draw inclusion count (only when inclusions enabled)
                if self.use_inclusions:
                    center = droplet.get("center")
                    count = droplet.get("inclusions", 0)
                    if center:
                        cv2.putText(
                            display, str(count), (int(center[0]) - 10, int(center[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
                        )

        # Draw inclusion masks in red
        for inc_mask in inclusion_masks:
            if inc_mask is not None:
                contours, _ = cv2.findContours(inc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (255, 0, 0), -1)

        ax.imshow(display)
        ax.set_title(f"Frame {frame_idx}")
        ax.axis("off")

    def run_analysis(self, output_dir, sample_frames=None):
        """Run analysis and print results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        mean_d, median_d = self.plot_size_distribution(output_path)

        lambda_val, chi2, p_value = None, None, None
        if self.use_poisson:
            lambda_val, chi2, p_value = self.plot_poisson_comparison(output_path)

        total_droplets = len(self.df)
        total_inclusions = int(self.df["inclusions"].sum()) if self.use_inclusions else 0
        with_inclusions = int((self.df["inclusions"] > 0).sum()) if self.use_inclusions else 0
        std_d = self.df["diameter_um"].std()

        self._write_summary(
            output_path,
            mean_d=mean_d,
            median_d=median_d,
            std_d=std_d,
            total_droplets=total_droplets,
            total_inclusions=total_inclusions,
            with_inclusions=with_inclusions,
            lambda_val=lambda_val,
            chi2=chi2,
            p_value=p_value,
        )

        # Create combined report
        stats_data = {
            "mean_d": mean_d,
            "median_d": median_d,
            "std_d": std_d,
            "lambda_val": lambda_val,
            "chi2": chi2,
            "p_value": p_value,
        }
        self.create_report(output_path, stats_data, sample_frames)

        cv_diameter = (std_d / mean_d * 100) if mean_d > 0 else 0

        print("\nSTATISTICAL SUMMARY")
        print("-" * 40)
        print(f"Droplets: {total_droplets}")
        print(f"Mean diameter: {mean_d:.1f} µm (CV: {cv_diameter:.1f}%)")

        if self.use_inclusions:
            print(
                f"Inclusions: {total_inclusions} total, {total_inclusions / total_droplets:.2f} per droplet"
            )
            print(
                f"With inclusions: {with_inclusions} ({with_inclusions / total_droplets * 100:.1f}%)"
            )
        else:
            print("Inclusions: OFF")

        if self.use_poisson and lambda_val is not None:
            print(f"Theoretical λ: {lambda_val:.3f}")

            if p_value is not None:
                print(f"\nChi-squared test:")
                print(f"  χ² = {chi2:.2f}, p = {p_value:.4f}")
                if p_value > 0.05:
                    print("  → Distribution follows Poisson (p > 0.05)")
                else:
                    print("  → Distribution deviates from Poisson (p < 0.05)")

        print(f"\nOutput saved to: {output_path}")

    def _write_summary(self, output_path, **stats):
        """Write summary.txt file with all settings and statistics."""
        project_name = output_path.name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        input_dir = self.settings.get("input_dir", "N/A")
        total_frames = self.df["frame"].nunique()

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

        cv_diameter = (stats["std_d"] / stats["mean_d"] * 100) if stats["mean_d"] > 0 else 0

        lines.extend([
            "",
            "RESULTS",
            "-" * 40,
            f"Total Frames Processed: {total_frames}",
            f"Total Droplets Detected: {stats['total_droplets']:,}",
            "",
            "Droplet Statistics:",
            f"  Mean Diameter: {stats['mean_d']:.1f} um",
            f"  Median Diameter: {stats['median_d']:.1f} um",
            f"  Std Deviation: {stats['std_d']:.1f} um",
            f"  CV: {cv_diameter:.1f}%",
        ])

        if self.use_inclusions:
            lines.extend([
                "",
                "Bead Statistics:",
                f"  Total Beads Detected: {stats['total_inclusions']:,}",
                f"  Mean per Droplet: {stats['total_inclusions'] / stats['total_droplets']:.2f}",
                f"  Droplets with Beads: {stats['with_inclusions']} ({stats['with_inclusions'] / stats['total_droplets'] * 100:.1f}%)",
            ])

        if self.use_poisson and stats.get("lambda_val") is not None:
            lines.extend([
                "",
                "POISSON ANALYSIS",
                "-" * 40,
                f"Theoretical Lambda: {stats['lambda_val']:.3f}",
            ])

            if stats.get("p_value") is not None:
                result = "FOLLOWS" if stats["p_value"] > 0.05 else "DEVIATES FROM"
                lines.extend([
                    f"Chi-squared: {stats['chi2']:.2f}",
                    f"P-value: {stats['p_value']:.4f}",
                    f"Result: Distribution {result} Poisson (p {'>' if stats['p_value'] > 0.05 else '<'} 0.05)",
                ])

        lines.extend([
            "",
            "=" * 80,
            "Generated by DropDrop",
            "=" * 80,
        ])

        summary_path = output_path / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))


class MultiplexStatistics:
    """Statistical analysis across multiple samples for multiplex mode.

    Reads CSVs directly — decoupled from pipeline processing.
    """

    def __init__(self, samples, settings):
        """Initialize with sample data and settings.

        Args:
            samples: List of (label, csv_path) tuples.
            settings: Dict with analysis settings.
        """
        self.settings = settings
        self.use_inclusions = settings.get("inclusions", True)
        self.use_poisson = settings.get("poisson", True) and self.use_inclusions

        self.samples = []
        for label, csv_path in samples:
            self.samples.append({
                "label": label,
                "df": pd.read_csv(csv_path),
            })

    def _calculate_poisson(self, df, median_diameter_um):
        """Calculate theoretical Poisson distribution for a sample."""
        bead_count = self.settings.get("count", 6.5e5)
        dilution = self.settings.get("dilution", 1000)
        radius_um = median_diameter_um / 2
        volume_ml = (4 / 3) * np.pi * (radius_um**3) * 1e-9
        lambda_val = (bead_count / (dilution * 2)) * volume_ml
        max_inc = int(df["inclusions"].max()) + 3
        x_range = np.arange(0, max_inc + 1)
        theoretical = stats.poisson.pmf(x_range, lambda_val)
        return x_range, theoretical, lambda_val

    def compute_per_sample_stats(self):
        """Compute statistics for each sample."""
        all_stats = []
        for sample in self.samples:
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
                # Chi-squared test
                observed, expected = [], []
                for i in actual.index:
                    if i < len(theoretical):
                        observed.append(actual[i])
                        expected.append(theoretical[i] * total_droplets)
                observed = np.array(observed)
                expected = np.array(expected)
                mask = expected >= 5
                chi2, p_value = None, None
                if mask.sum() >= 2:
                    obs_f = observed[mask]
                    exp_f = expected[mask]
                    exp_f = exp_f * (obs_f.sum() / exp_f.sum())
                    chi2, p_value = stats.chisquare(obs_f, exp_f)
                sample_stats["lambda_val"] = lambda_val
                sample_stats["chi2"] = chi2
                sample_stats["p_value"] = p_value

            all_stats.append(sample_stats)

        return all_stats

    def compute_global_axes(self):
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

    def merge_dataframes(self):
        """Merge all sample DataFrames with 'sample' column."""
        dfs = []
        for s in self.samples:
            df_copy = s["df"].copy()
            df_copy.insert(0, "sample", s["label"])
            dfs.append(df_copy)
        return pd.concat(dfs, ignore_index=True)

    def plot_overlaid_size_distribution(self, output_path, axes_limits):
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
        plt.savefig(output_path / "size_distribution.png", dpi=200)
        plt.close()

    def plot_overlaid_poisson(self, output_path, axes_limits):
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
        plt.savefig(output_path / "poisson_comparison.png", dpi=200)
        plt.close()

    def create_summary_report(self, output_path, all_stats, axes_limits):
        """Create combined summary_report.png with table, overlaid plots, CV barplot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.samples)))

        # [0,0]: Comparison table
        ax_table = axes[0, 0]
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
        ax_size = axes[0, 1]
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
        ax_cv = axes[1, 0]
        labels = [s["label"] for s in all_stats]
        cvs = [s["cv"] for s in all_stats]
        ax_cv.bar(labels, cvs, color=colors[:len(labels)], edgecolor="black", alpha=0.8)
        ax_cv.set_ylabel("CV (%)")
        ax_cv.set_title("Coefficient of Variation")
        ax_cv.grid(True, alpha=0.3, axis="y")
        for j, v in enumerate(cvs):
            ax_cv.text(j, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

        # [1,1]: Overlaid Poisson or placeholder
        ax_poisson = axes[1, 1]
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

        plt.suptitle("DropDrop Multiplex Report", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path / "summary_report.png", dpi=200, bbox_inches="tight")
        plt.close()

    def write_merged_summary(self, output_path, all_stats):
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

        lines.extend([
            "=" * 80,
            "Generated by DropDrop (Multiplex Mode)",
            "=" * 80,
        ])

        with open(output_path / "summary.txt", "w") as f:
            f.write("\n".join(lines))

    def run_analysis(self, output_path):
        """Run complete multiplex analysis."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        all_stats = self.compute_per_sample_stats()
        axes_limits = self.compute_global_axes()

        # Merged CSV
        merged_df = self.merge_dataframes()
        merged_df.to_csv(output_path / "data.csv", index=False)

        # Overlaid plots
        self.plot_overlaid_size_distribution(output_path, axes_limits)
        if self.use_poisson:
            self.plot_overlaid_poisson(output_path, axes_limits)

        # Summary report and text
        self.create_summary_report(output_path, all_stats, axes_limits)
        self.write_merged_summary(output_path, all_stats)

        # Console output
        print("\nMULTIPLEX SUMMARY")
        print("=" * 50)
        for s in all_stats:
            line = f"  {s['label']}: {s['total_droplets']} droplets, mean={s['mean_d']:.1f}um, CV={s['cv']:.1f}%"
            if self.use_inclusions:
                line += f", inclusions={s['total_inclusions']}"
            print(line)
        print(f"\nOutput: {output_path}")
