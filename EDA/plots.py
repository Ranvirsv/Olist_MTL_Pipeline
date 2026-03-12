"""
Reusable, Pydantic-configured plotting utilities for EDA.

Usage:
    from plots import Plotter, PlotConfig

    # Use defaults
    plotter = Plotter()

    # Or customise via PlotConfig (dict, env vars, .model_validate(), etc.)
    plotter = Plotter(config=PlotConfig(figsize=(14, 7), save_dir="figures/", style="darkgrid"))

    plotter.histogram(df["delivery_days"], title="Delivery Days", bins=30)
    plotter.scatter(df["weight"], df["freight"], hue=df["region"], title="Weight vs Freight")

    # Get (fig, ax) back for further customization
    fig, ax = plotter.histogram(df["col"], title="Custom", show=False)
    ax.axvline(x=50, color="red", linestyle="--")
    plt.show()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray, list]


# ---------------------------------------------------------------------------
# Pydantic config model
# ---------------------------------------------------------------------------
class PlotConfig(BaseModel):
    """Validated, serialisable configuration for the Plotter."""

    model_config = {"arbitrary_types_allowed": True}

    style: str = Field("whitegrid", description="Seaborn theme style")
    context: str = Field("notebook", description="Seaborn scaling context (paper | notebook | talk | poster)")
    palette: str = Field("deep", description="Seaborn colour palette")
    figsize: tuple[int, int] = Field((10, 6), description="Default (width, height) in inches")
    dpi: int = Field(150, ge=50, le=600, description="Resolution for saved figures")
    save_dir: Optional[str] = Field(None, description="Directory to save figures to. Created automatically if set.")
    show_by_default: bool = Field(True, description="Call plt.show() after every plot by default")

    @model_validator(mode="after")
    def _create_save_dir(self) -> "PlotConfig":
        if self.save_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        return self


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------
class Plotter:
    """Stateless, reusable plotting helper driven by a PlotConfig.

    Each plot method is self-contained — pass data as arguments,
    not through the constructor.  The constructor only stores *defaults*.
    """

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()
        sns.set_theme(
            style=self.config.style,
            context=self.config.context,
            palette=self.config.palette,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_fig(self, figsize: Optional[tuple] = None) -> tuple[plt.Figure, plt.Axes]:
        return plt.subplots(figsize=figsize or self.config.figsize)

    def _finalise(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        fig.tight_layout()

        if save_as:
            dest = Path(self.config.save_dir) / save_as if self.config.save_dir else Path(save_as)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest, dpi=self.config.dpi, bbox_inches="tight")

        if show if show is not None else self.config.show_by_default:
            plt.show()

        return fig, ax

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------
    def histogram(
        self,
        data: ArrayLike,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = "Count",
        bins: Union[int, str] = "auto",
        kde: bool = True,
        color: Optional[str] = None,
        figsize: Optional[tuple] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Histogram with optional KDE overlay."""
        fig, ax = self._create_fig(figsize)
        sns.histplot(data, bins=bins, kde=kde, color=color, ax=ax, **kwargs)
        return self._finalise(fig, ax, title, xlabel, ylabel, save_as, show)

    def boxplot(
        self,
        data: ArrayLike,
        *,
        orient: str = "v",
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        figsize: Optional[tuple] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Box-and-whisker plot."""
        fig, ax = self._create_fig(figsize)
        sns.boxplot(data=data, orient=orient, color=color, ax=ax, **kwargs)
        return self._finalise(fig, ax, title, xlabel, ylabel, save_as, show)

    def scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        hue: Optional[ArrayLike] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        figsize: Optional[tuple] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Scatter plot of *x* vs *y*, with optional *hue* grouping."""
        fig, ax = self._create_fig(figsize)
        sns.scatterplot(x=x, y=y, hue=hue, color=color, ax=ax, **kwargs)
        return self._finalise(fig, ax, title, xlabel, ylabel, save_as, show)

    def countplot(
        self,
        data: ArrayLike,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = "Count",
        order: Optional[Sequence] = None,
        rotate_labels: int = 0,
        color: Optional[str] = None,
        figsize: Optional[tuple] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Count / bar plot for categorical data."""
        fig, ax = self._create_fig(figsize)
        sns.countplot(x=data, order=order, color=color, ax=ax, **kwargs)
        if rotate_labels:
            ax.tick_params(axis="x", rotation=rotate_labels)
        return self._finalise(fig, ax, title, xlabel, ylabel, save_as, show)

    def barplot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        hue: Optional[ArrayLike] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        rotate_labels: int = 0,
        figsize: Optional[tuple] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Bar plot showing mean (or other estimator) of *y* per *x*."""
        fig, ax = self._create_fig(figsize)
        sns.barplot(x=x, y=y, hue=hue, ax=ax, **kwargs)
        if rotate_labels:
            ax.tick_params(axis="x", rotation=rotate_labels)
        return self._finalise(fig, ax, title, xlabel, ylabel, save_as, show)

    def heatmap(
        self,
        data: pd.DataFrame,
        *,
        title: Optional[str] = None,
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = "coolwarm",
        figsize: Optional[tuple] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Correlation heatmap (computes `.corr()` automatically if raw df is passed)."""
        corr = data.corr(numeric_only=True) if not self._is_square(data) else data
        auto_size = (max(8, len(corr.columns)), max(6, len(corr.columns) - 1))
        fig, ax = self._create_fig(figsize or auto_size)
        sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, ax=ax, **kwargs)
        return self._finalise(fig, ax, title, save_as=save_as, show=show)

    def pairplot(
        self,
        data: pd.DataFrame,
        *,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> sns.PairGrid:
        """Seaborn pair plot (creates its own figure grid)."""
        g = sns.pairplot(data, hue=hue, **kwargs)
        if title:
            g.figure.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        if save_as:
            dest = Path(self.config.save_dir) / save_as if self.config.save_dir else Path(save_as)
            dest.parent.mkdir(parents=True, exist_ok=True)
            g.figure.savefig(dest, dpi=self.config.dpi, bbox_inches="tight")
        if show if show is not None else self.config.show_by_default:
            plt.show()
        return g

    def lineplot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        hue: Optional[ArrayLike] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Optional[tuple] = None,
        save_as: Optional[str] = None,
        show: Optional[bool] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Line plot, useful for time-series or trends."""
        fig, ax = self._create_fig(figsize)
        sns.lineplot(x=x, y=y, hue=hue, ax=ax, **kwargs)
        return self._finalise(fig, ax, title, xlabel, ylabel, save_as, show)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _is_square(df: pd.DataFrame) -> bool:
        return df.shape[0] == df.shape[1] and (df.columns == df.index).all()