from typing import Callable, Optional, Tuple

import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import itertools
from tqdm import tqdm
from fab.types_ import LogProbFunc, Distribution


# adjusted from fab.utils.plotting.plot_contours
def plot_fn(
    log_prob_func: Callable,
    ax: Optional[plt.Axes] = None,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    grid_width: int = 20,
    n_contour_levels: Optional[int] = None,
    log_prob_min: float = -1000.0,
    plot_style: str = "imshow",
    plot_kwargs: dict = {},
    colorbar: bool = False,
    quantity: str = "log_prob",
    batch_size: int = -1,
    device: str = "cpu",
    load_path: str = None,
):
    """Plot heatmap of log probability density.

    Args:
        samples: Samples to plot on top of heatmap
        name: Title for plot
        n_contour_levels: Not used for imshow
        bounds: Plot bounds as (min, max) tuple
        grid_width: Number of points in each dimension for grid
        log_prob_min: Minimum log probability to show
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.get_figure()

    x_points_dim1 = torch.linspace(
        bounds[0], bounds[1], grid_width, device=device
    )
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(
        list(itertools.product(x_points_dim1, x_points_dim2)), device=device
    )

    recompute = True
    if load_path is not None:
        # Check if the file exists
        if os.path.exists(f"{load_path}.pkl"):
            print(f"plot_fn: Loading precomputed values from {load_path}.pkl")
            with open(f"{load_path}.pkl", "rb") as f:
                saved_data = pickle.load(f)

            # Verify the saved data matches our current parameters
            if saved_data["grid_width"] == grid_width and np.allclose(
                saved_data["bounds"], bounds
            ):
                log_p_x = saved_data["log_p_x"]
                recompute = False
            else:
                print(
                    f"Saved data parameters don't match current parameters. Recomputing."
                )
        else:
            pass
            # print(
            #     f"No precomputed values found at {load_path}.pkl. Computing from scratch."
            # )

    if recompute:
        # with torch.no_grad():
        if batch_size <= 0 or batch_size >= x_points.shape[0]:
            # Compute log_p_x for all points
            log_p_x = log_prob_func(x_points).detach().cpu()
        else:
            # Compute log_p_x in batches
            log_p_x = []
            so_far = 0
            nbatches = x_points.shape[0] // batch_size + 1
            # print(
            #     f"Dividing {x_points.shape[0]} points into {nbatches} batches of {batch_size} points"
            # )
            for i in tqdm(range(nbatches), desc="Plotting"):
                batch = x_points[so_far : so_far + batch_size]
                # Ensure the batch isn't empty
                if batch.shape[0] <= 0:
                    break
                # Compute log_p_x for the batch
                log_p_x.append(log_prob_func(batch).detach().cpu())
                so_far += batch_size
            log_p_x = torch.cat(log_p_x)
        if load_path is not None:
            os.makedirs(os.path.dirname(load_path), exist_ok=True)
            with open(f"{load_path}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "grid_width": grid_width,
                        "bounds": bounds,
                        "log_p_x": log_p_x,
                    },
                    f,
                )
            # print(f"plot_fn: Saved precomputed values to {load_path}.pkl")
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)

    if quantity in ["prob", "p"]:
        log_p_x = torch.exp(log_p_x)
        label = "P = exp(-E)"
    elif quantity in ["energy", "e"]:
        log_p_x = -log_p_x
        label = "E"
    elif quantity in ["loge", "log"]:
        log_p_x = -log_p_x
        if log_p_x.min() <= 0:
            print(f"Warning: plot_fn: shifting log_p_x by {log_p_x.min()} to avoid log(0)")
            log_p_x += torch.abs(log_p_x.min()) + 1e-1
        log_p_x = torch.log(log_p_x)
        label = "log(E)"
    else:
        label = "Log(P) = -E"
    log_p_x = log_p_x.reshape((grid_width, grid_width))

    x_points_dim1 = (
        x_points[:, 0].reshape((grid_width, grid_width)).cpu().numpy()
    )
    x_points_dim2 = (
        x_points[:, 1].reshape((grid_width, grid_width)).cpu().numpy()
    )

    # cmaps
    # ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    # 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    # 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    colorbar_kwargs = {
        "label": label,
        "fraction": 0.046,
        "pad": 0.04,
        "shrink": 0.8,
    }

    if plot_style == "imshow":
        # Create heatmap using imshow
        dflt = {
            "cmap": "viridis",
            # "origin": "lower",
            # "aspect": "equal",
        }
        for k, v in dflt.items():
            if k not in plot_kwargs:
                plot_kwargs[k] = v
        # imshow transposes the array just to screw with you
        im = ax.imshow(
            log_p_x.numpy().T,
            extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
            # extent=[x_points_dim1.min(), x_points_dim1.max(), x_points_dim2.min(), x_points_dim2.max()],
            origin="lower",
            **plot_kwargs,
        )
        if colorbar:
            # Add colorbar
            plt.colorbar(im, ax=ax, **colorbar_kwargs)

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
            # aspect="equal",
            **plot_kwargs,
        )
        if colorbar:
            fig.colorbar(im, ax=ax, **colorbar_kwargs)

        # Set equal aspect ratio
        ax.set_aspect("equal")

    elif plot_style == "contours":
        # By default, a linear scaling is used, mapping the lowest value to 0 and the highest to 1
        im = ax.contour(
            x_points_dim1,
            x_points_dim2,
            log_p_x,
            levels=n_contour_levels,
            # extend="both",
            **plot_kwargs,
        )
        if colorbar:
            fig.colorbar(im, ax=ax, **colorbar_kwargs)


def plot_contours(
    log_prob_func: LogProbFunc,
    ax: Optional[plt.Axes] = None,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    grid_width: int = 20,
    n_contour_levels: Optional[int] = None,
    log_prob_min: float = -1000.0,
    plot_kwargs: dict = {},
    colorbar: bool = False,
    device: str = "cpu",
):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)

    x_points_dim1 = torch.linspace(
        bounds[0], bounds[1], grid_width, device=device
    )
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(
        list(itertools.product(x_points_dim1, x_points_dim2)), device=device
    )

    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width, grid_width))

    x_points_dim1 = (
        x_points[:, 0].reshape((grid_width, grid_width)).cpu().numpy()
    )
    x_points_dim2 = (
        x_points[:, 1].reshape((grid_width, grid_width)).cpu().numpy()
    )
    #
    im = ax.contour(
        x_points_dim1,
        x_points_dim2,
        log_p_x,
        levels=n_contour_levels,
        **plot_kwargs,
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label="Log(P)", fraction=0.046, pad=0.04, shrink=0.8)


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
