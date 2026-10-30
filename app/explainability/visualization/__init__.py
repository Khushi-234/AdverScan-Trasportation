"""
XAI Visualization package for AdverScan.
Exports per-sample and dataset-level figure builders.
"""

from app.explainability.visualization.sample_plots import plot_sample
from app.explainability.visualization.build_figures import build_all_figures

__all__ = [
    "plot_sample",
    "build_all_figures",
]
