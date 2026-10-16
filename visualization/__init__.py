"""Visual Analytics and Reporting Package.

Role:
    Generates publication-quality charts and interactive visualizations for option
    payoff profiles, multidimensional Greek surfaces, and dynamic strategy comparisons.

Dependencies:
    - .plots: DerivativesPlotter plotting routines.

Core Classes Exposed:
    - DerivativesPlotter: Master plotting engine for option strategies and Greeks.
    - OptionVisualizer: Alias for DerivativesPlotter.
"""

from .plots import DerivativesPlotter, OptionVisualizer

__all__ = [
    "DerivativesPlotter",
    "OptionVisualizer",
]
