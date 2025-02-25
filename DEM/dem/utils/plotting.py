from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import itertools

from fab.types_ import LogProbFunc, Distribution


# adjusted from fab.utils.plotting.plot_contours
def plot_imshow(
    log_prob_func: LogProbFunc,
    ax: Optional[plt.Axes] = None,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    grid_width_n_points: int = 20,
    n_contour_levels: Optional[int] = None,
    log_prob_min: float = -1000.0,
    plot_style: str = "imshow",
    plot_kwargs: dict = {},
):
    """Plot heatmap of log probability density.

    Args:
        samples: Samples to plot on top of heatmap
        name: Title for plot
        n_contour_levels: Not used for imshow
        bounds: Plot bounds as (min, max) tuple
        grid_width_n_points: Number of points in each dimension for grid
        log_prob_min: Minimum log probability to show
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.get_figure()

    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))

    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))

    x_points_dim1 = (
        x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    )
    x_points_dim2 = (
        x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    )

    # cmaps
    # ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    # 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    # 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    if plot_style == "imshow":
        # Create heatmap using imshow
        dflt = {
            "cmap": "viridis",
            "origin": "lower",
            "aspect": "equal",
        }
        for k, v in dflt.items():
            if k not in plot_kwargs:
                plot_kwargs[k] = v
        im = ax.imshow(
            log_p_x.numpy(),
            extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
            **plot_kwargs,
        )
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Log(P)', fraction=0.046, pad=0.04, shrink=0.8)

    elif plot_style == "scatter":
        # Use scatter
        dflt = {
            "cmap": "GnBu",
            "s": 1,
        }
        for k, v in dflt.items():
            if k not in plot_kwargs:
                plot_kwargs[k] = v
        im = ax.scatter(
            x_points_dim1,
            x_points_dim2,
            c=log_p_x.numpy(),
            **plot_kwargs,
        )
        fig.colorbar(im, ax=ax, label="log(P) = -E", fraction=0.046, pad=0.04, shrink=0.8)

    # Set equal aspect ratio
    ax.set_aspect("equal")


def plot_contours(
    log_prob_func: LogProbFunc,
    ax: Optional[plt.Axes] = None,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    grid_width_n_points: int = 20,
    n_contour_levels: Optional[int] = None,
    log_prob_min: float = -1000.0,
    plot_kwargs: dict = {},
):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)

    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))

    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))

    x_points_dim1 = (
        x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    )
    x_points_dim2 = (
        x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    )

    if n_contour_levels:
        ax.contour(
            x_points_dim1,
            x_points_dim2,
            log_p_x,
            levels=n_contour_levels,
            **plot_kwargs,
        )
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, **plot_kwargs)


def plot_marginal_pair(
    samples: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[float, float] = (-5.0, 5.0),
    alpha: float = 0.5,
    plot_kwargs: dict = {},
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = torch.clamp(samples, bounds[0], bounds[1])
    samples = samples.cpu().detach()
    dflt = {
        "alpha": alpha,
        "marker": "o",
        # "markersize": 1,
        "linestyle": "none",
    }
    for k, v in dflt.items():
        if k not in plot_kwargs:
            plot_kwargs[k] = v
    ax.plot(
        samples[:, marginal_dims[0]],
        samples[:, marginal_dims[1]],
        **plot_kwargs,
    )
