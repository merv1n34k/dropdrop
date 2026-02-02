"""Statistical analysis for droplet detection results."""

from datetime import datetime
from pathlib import Path

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
        self.use_poisson = self.settings.get("poisson", True)

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
            result_text = f"χ² = {chi2:.2f}, p = {p_value:.4f}"
            if p_value > 0.05:
                result_text += "\n✓ Follows Poisson"
            else:
                result_text += "\n✗ Deviates from Poisson"
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

    def run_analysis(self, output_dir):
        """Run analysis and print results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        mean_d, median_d = self.plot_size_distribution(output_path)

        lambda_val, chi2, p_value = None, None, None
        if self.use_poisson:
            lambda_val, chi2, p_value = self.plot_poisson_comparison(output_path)

        total_droplets = len(self.df)
        total_inclusions = int(self.df["inclusions"].sum())
        with_inclusions = int((self.df["inclusions"] > 0).sum())
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

        print("\nSTATISTICAL SUMMARY")
        print("-" * 40)
        print(f"Droplets: {total_droplets}")
        print(f"Mean diameter: {mean_d:.1f} µm")
        print(
            f"Inclusions: {total_inclusions} total, {total_inclusions / total_droplets:.2f} per droplet"
        )
        print(
            f"With inclusions: {with_inclusions} ({with_inclusions / total_droplets * 100:.1f}%)"
        )

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
            f"Total Droplets Detected: {stats['total_droplets']:,}",
            f"Total Beads Detected: {stats['total_inclusions']:,}",
            "",
            "Droplet Statistics:",
            f"  Mean Diameter: {stats['mean_d']:.1f} um",
            f"  Median Diameter: {stats['median_d']:.1f} um",
            f"  Std Deviation: {stats['std_d']:.1f} um",
            "",
            "Bead Statistics:",
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
