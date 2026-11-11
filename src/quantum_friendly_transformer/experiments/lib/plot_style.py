"""Pinned plotting constants shared by experiment figure generators."""

from pathlib import Path


STYLE_VERSION = "qft-paper-v1"
STYLE_PATH = Path(__file__).resolve().parents[1] / "styles" / f"{STYLE_VERSION}.mplstyle"

FIGURE_SIZE = (7.0, 4.2)
SAVE_DPI = 220
LINE_WIDTH = 1.6
MARKER_SIZE = 4.5
GRID_ALPHA = 0.25
LEGEND_FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 7
LOG_X_TICK_ROTATION = 30


def apply_plot_style(plt) -> None:
    """Apply the repository's versioned Matplotlib style."""

    plt.style.use(str(STYLE_PATH))


def save_figure(fig, path) -> None:
    """Save one tightly laid-out figure with the pinned output DPI."""

    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI)
