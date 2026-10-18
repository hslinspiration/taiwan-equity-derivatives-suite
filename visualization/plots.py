"""Institutional Visualization Engine for Taiwan Equity Derivatives.

Role:
    Generates publication-grade static visualizations (300 DPI PNGs) for multi-strategy
    terminal P&L comparisons, Covered Call interim mark-to-market Theta decay profiles,
    and 3D Gamma sensitivity surfaces illustrating expiration Pin Risk near the strike.
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def _apply_plot_style() -> None:
    """Apply clean institutional chart styling with fallback support."""
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "bmh"):
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue


def _ensure_parent_dir(file_path: str) -> None:
    """Ensure the parent directory for `file_path` exists."""
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


class DerivativesPlotter:
    """Master visualization engine for Taiwan equity derivative overlays and Greek surfaces."""

    def __init__(self, output_dir: str = "output") -> None:
        """Initialize DerivativesPlotter and ensure output directory exists.

        Args:
            output_dir: Default directory path for saving generated charts.
        """
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        _apply_plot_style()

    def plot_multi_strategy_pnl(
        self,
        spot_grid: np.ndarray,
        payoffs: Dict[str, np.ndarray],
        breakevens: Optional[Dict[str, float]] = None,
        save_path: str = "output/pnl_comparison.png",
    ) -> str:
        """Plot terminal net P&L profiles across Underlying Stock and 3 option overlays.

        Features:
            - High-contrast institutional color scheme (#7f8c8d, #1a5276, #a93226, #d35400).
            - Breakeven values integrated directly into the upper-left legend.
            - Clean vertical reference lines matching strategy colors without inline text collision.
            - Inset summary card in the lower-right quadrant displaying all Breakeven points.

        Args:
            spot_grid: 1D NumPy array of terminal underlying spot prices in TWD.
            payoffs: Dictionary mapping strategy names ('Underlying Stock', 'Covered Call',
                'Protective Put', 'Zero-Cost Collar') to 1D NumPy P&L arrays.
            breakevens: Optional dictionary mapping strategy names to breakeven spot prices.
            save_path: Destination file path for the rendered PNG image.

        Returns:
            str: Path to the saved figure.
        """
        _apply_plot_style()
        _ensure_parent_dir(save_path)

        fig, ax = plt.subplots(figsize=(11, 6.5))

        # High-contrast institutional palette
        palette = {
            "Underlying Stock": ("#7f8c8d", "--", 1.8),
            "Covered Call": ("#1a5276", "-", 2.5),
            "Protective Put": ("#a93226", "-", 2.5),
            "Zero-Cost Collar": ("#d35400", "-.", 2.6),
        }

        # 1. Plot strategy PnL curves with Breakeven integrated into legend labels
        for label, pnl_curve in payoffs.items():
            color, linestyle, linewidth = palette.get(label, ("#1a5276", "-", 2.2))

            if breakevens and label in breakevens:
                be_val = breakevens[label]
                legend_label = f"{label} (BE: NT${be_val:,.1f})"
            else:
                legend_label = label

            ax.plot(
                spot_grid,
                pnl_curve,
                label=legend_label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )

        # 2. Zero-profit horizontal reference line
        ax.axhline(0.0, color="#d62828", linestyle="--", linewidth=1.2, alpha=0.8, label="Breakeven (y = 0)")

        # 3. Clean vertical Breakeven lines without colliding inline text annotations
        if breakevens:
            be_colors = {
                "Underlying Stock": "#7f8c8d",
                "Covered Call": "#1a5276",
                "Protective Put": "#a93226",
                "Zero-Cost Collar": "#d35400",
            }
            for strat_name, be_price in breakevens.items():
                c = be_colors.get(strat_name, "#7f8c8d")
                ax.axvline(be_price, color=c, linestyle=":", linewidth=1.4, alpha=0.85)

            # 4. Inset Breakeven Summary Box in clean lower-right quadrant
            box_lines = ["BREAKEVEN (BE) POINTS"]
            for strat_name, be_price in breakevens.items():
                box_lines.append(f"• {strat_name}: NT${be_price:,.1f}")
            box_text = "\n".join(box_lines)

            ax.text(
                0.72,
                0.04,
                box_text,
                transform=ax.transAxes,
                fontsize=9.0,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="white",
                    edgecolor="#d5d8dc",
                    alpha=0.9,
                ),
            )

        ax.set_title(
            "Taiwan Equity Derivatives: Institutional Payoff Comparison at Expiration",
            fontsize=13.5,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlabel("Underlying Spot Price (TWD)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Net Profit / Loss (TWD)", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_theta_decay(
        self,
        spot_grid: np.ndarray,
        decay_curves: Dict[str, np.ndarray],
        save_path: str = "output/theta_decay.png",
    ) -> str:
        """Visualize Covered Call mark-to-market P&L evolution across 3 time horizons.

        Highlights how the at-the-money (ATM) and slightly out-of-the-money (OTM)
        regions lift upward over time as positive Theta is captured by the short call.

        Args:
            spot_grid: 1D NumPy array of underlying spot prices in TWD.
            decay_curves: Dictionary with keys such as 'T=30 Days (Inception)',
                'T=15 Days (Mid-term)', and 'T=0 Days (Expiration Payoff)'.
            save_path: Destination file path for the rendered PNG image.

        Returns:
            str: Path to the saved figure.
        """
        _apply_plot_style()
        _ensure_parent_dir(save_path)

        fig, ax = plt.subplots(figsize=(11, 6.5))

        curve_styles = {
            "T=30 Days (Inception)": ("#457b9d", "--", 2.0),
            "T=15 Days (Mid-term)": ("#2a9d8f", "-.", 2.3),
            "T=0 Days (Expiration Payoff)": ("#1d3557", "-", 2.7),
        }

        keys = list(decay_curves.keys())
        for label, curve in decay_curves.items():
            color, ls, lw = curve_styles.get(label, ("#264653", "-", 2.2))
            ax.plot(spot_grid, curve, label=label, color=color, linestyle=ls, linewidth=lw)

        # Shade positive Theta capture zone between Inception and Expiration curves
        if len(keys) >= 2:
            first_curve = decay_curves[keys[0]]
            last_curve = decay_curves[keys[-1]]
            ax.fill_between(
                spot_grid,
                first_curve,
                last_curve,
                where=(last_curve >= first_curve),
                color="#2a9d8f",
                alpha=0.16,
                label="Positive Theta Capture Region",
            )

        ax.axhline(0.0, color="#d62828", linestyle="--", linewidth=1.1, alpha=0.75)

        ax.set_title(
            "Covered Call Interim PnL & Theta Decay Dynamics",
            fontsize=13.5,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlabel("Spot Price (TWD)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mark-to-Market PnL (TWD)", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_greeks_surface(
        self,
        spot_range: np.ndarray,
        time_range: np.ndarray,
        gamma_surface: np.ndarray,
        strike: float,
        save_path: str = "output/greeks_surface.png",
    ) -> str:
        """Render a 3D Gamma sensitivity surface across Spot Price and Days to Expiration.

        Demonstrates expiration Pin Risk as Gamma spikes sharply around `strike` when
        time to maturity approaches zero.

        Args:
            spot_range: 1D or 2D meshgrid array of underlying spot prices in TWD.
            time_range: 1D or 2D meshgrid array of days to expiration.
            gamma_surface: 2D NumPy array of option Gamma values.
            strike: Option exercise strike price in TWD.
            save_path: Destination file path for the rendered PNG image.

        Returns:
            str: Path to the saved figure.
        """
        _apply_plot_style()
        _ensure_parent_dir(save_path)

        if spot_range.ndim == 1 and time_range.ndim == 1:
            s_mesh, t_mesh = np.meshgrid(spot_range, time_range)
        else:
            s_mesh, t_mesh = spot_range, time_range

        fig = plt.figure(figsize=(11.5, 7.5))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            s_mesh,
            t_mesh,
            gamma_surface,
            cmap="viridis",
            edgecolor="none",
            alpha=0.92,
            antialiased=True,
        )

        # Highlight strike plane line along the surface
        strike_line_t = np.linspace(float(np.min(t_mesh)), float(np.max(t_mesh)), 30)
        strike_line_s = np.full_like(strike_line_t, float(strike))
        strike_line_z = np.zeros_like(strike_line_t)
        ax.plot(
            strike_line_s,
            strike_line_t,
            strike_line_z,
            color="#d62828",
            linestyle="--",
            linewidth=2.0,
            label=f"Strike K = NT${strike:,.0f} (Pin Risk Axis)",
        )

        ax.set_title(
            "Gamma Sensitivity Surface (Spot vs. Time to Expiration)",
            fontsize=13.5,
            fontweight="bold",
            pad=16,
        )
        ax.set_xlabel("Spot Price (TWD)", fontsize=10.5, fontweight="bold", labelpad=10)
        ax.set_ylabel("Days to Expiration", fontsize=10.5, fontweight="bold", labelpad=10)
        ax.set_zlabel("Gamma", fontsize=10.5, fontweight="bold", labelpad=8)
        ax.view_init(elev=28, azim=-135)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.1)
        cbar.set_label("Gamma (dDelta / dSpot)", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=9)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path


# Backward-compatible alias
OptionVisualizer = DerivativesPlotter
