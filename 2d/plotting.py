import torch
import numpy as np
import torch.nn as nn
import time
import os
import pathlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# plotting directory: current directory
plotting_dir = pathlib.Path(__file__).parent / "plots"
plotting_dir.mkdir(exist_ok=True)


def plot_potential_energy_surface_2d(
    X_np, Y_np, pes_np, grad_norm_np, p_xy, cutoff=None, percentile=95
):
    t1 = time.time()

    # Dynamically set cutoff values if not specified.
    if cutoff is None:
        cutoff_pes = np.percentile(pes_np, percentile)
        cutoff_grad = np.percentile(grad_norm_np, percentile)
        cutoff_pxy = np.percentile(p_xy, percentile)
    else:
        cutoff_pes = cutoff_grad = cutoff_pxy = cutoff

    mask = pes_np < cutoff_pes
    pes_np_masked = np.ma.masked_where(~mask, pes_np).filled(np.nan)
    # mask = grad_norm_np < cutoff_grad
    grad_norm_np_masked = np.ma.masked_where(~mask, grad_norm_np).filled(np.nan)

    # X_np_masked = np.ma.masked_where(~mask, X_np).filled(np.nan)
    # Y_np_masked = np.ma.masked_where(~mask, Y_np).filled(np.nan)
    # clip to the min and max of the masked values
    X_np_masked = np.clip(X_np, np.min(X_np[mask]), np.max(X_np[mask]))
    Y_np_masked = np.clip(Y_np, np.min(Y_np[mask]), np.max(Y_np[mask]))

    # Create a figure with 3 subplots using make_subplots.
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["PES", "Gradient Norm", "Boltzmann Distribution"],
        horizontal_spacing=0.05,
    )
    fig.update_layout(width=1800, height=500)

    # Plot the PES surface.
    fig.add_trace(
        go.Contour(
            x=X_np_masked[0, :],
            y=Y_np_masked[:, 0],
            z=pes_np_masked,
            colorscale="Viridis",
            colorbar=dict(title="f(x,y)", x=0.3),
            name="PES",
            zmin=np.min(pes_np_masked),
            zmax=np.max(pes_np_masked),
        ),
        row=1,
        col=1,
    )
    # Plot the gradient norm.
    fig.add_trace(
        go.Contour(
            x=X_np_masked[0, :],
            y=Y_np_masked[:, 0],
            z=grad_norm_np_masked,
            colorscale="Plasma",
            colorbar=dict(title="|∇f(x,y)|", x=0.64),
            name="Gradient Norm",
            zmin=np.min(grad_norm_np_masked),
            zmax=np.max(grad_norm_np_masked),
        ),
        row=1,
        col=2,
    )

    # Plot the Boltzmann distribution.
    fig.add_trace(
        go.Contour(
            x=X_np[0, :],
            y=Y_np[:, 0],
            z=p_xy,
            colorscale="Cividis",
            colorbar=dict(title="p(x,y)", x=0.98),
            name="Boltzmann Distribution",
            zmin=np.min(p_xy),
            zmax=np.max(p_xy),
        ),
        row=1,
        col=3,
    )

    # Update subplot titles and labels.
    fig.update_layout(
        title_text="Potential Energy Surface Analysis",
        showlegend=True,
        annotations=[
            dict(
                text="PES",
                xref="paper",
                yref="paper",
                x=0.16,
                y=1.1,
                showarrow=False,
            ),
            dict(
                text="Gradient Norm |∇PES(x,y)|",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.1,
                showarrow=False,
            ),
            dict(
                text="Boltzmann Distribution over |∇PES(x,y)|",
                xref="paper",
                yref="paper",
                x=0.84,
                y=1.1,
                showarrow=False,
            ),
        ],
    )

    # Update x and y axis labels for all subplots.
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")

    t2 = time.time()
    figname = plotting_dir / "potential_energy_surface.png"
    fig.write_image(figname)
    print(f"Figure saved as {figname} ({t2-t1:.2f} seconds)")


def plot_potential_energy_surface_3d(
    X_np, Y_np, pes_np, grad_norm_np, p_xy, cutoff=None, percentile=95
):
    """
    Plot the potential energy surface, gradient norm, and Boltzmann distribution in 3D.
    Uses dynamic cutoff values via the given percentile.
    """
    t1 = time.time()

    # Dynamically set cutoff for PES if not provided.
    if cutoff is None:
        cutoff_pes = np.percentile(pes_np, percentile)
    else:
        cutoff_pes = cutoff

    # Create a mask: only values below the cutoff are kept.
    mask = pes_np < cutoff_pes
    # Mask the PES along with the corresponding X and Y values.
    # The .filled(np.nan) converts masked values to NaN.
    pes_np_masked = np.ma.masked_where(~mask, pes_np).filled(np.nan)
    grad_norm_np_masked = np.ma.masked_where(~mask, grad_norm_np).filled(np.nan)
    x_masked = np.ma.masked_where(~mask, X_np).filled(np.nan)
    y_masked = np.ma.masked_where(~mask, Y_np).filled(np.nan)

    # Plot PES surface.
    fig1 = go.Figure()
    fig1.add_trace(
        go.Surface(
            x=x_masked,
            y=y_masked,
            z=pes_np_masked,
            colorscale="Viridis",
            colorbar=dict(title="PES(x,y)"),
            name="Potential Energy Surface",
            cmax=cutoff_pes,
        )
    )
    fig1.update_layout(
        title="Potential Energy Surface",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="PES Value",
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    figname1 = plotting_dir / "pes_surface_3d.png"
    fig1.write_image(figname1)
    print(f"3D figure saved as {figname1} ({time.time()-t1:.2f} seconds)")

    # Plot gradient norm surface.
    fig2 = go.Figure()
    fig2.add_trace(
        go.Surface(
            x=x_masked,
            y=y_masked,
            z=grad_norm_np_masked,
            colorscale="Plasma",
            colorbar=dict(title="|∇PES(x,y)|"),
            name="Gradient Norm",
        )
    )
    fig2.update_layout(
        title="Gradient Norm Surface",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="Gradient Norm",
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    figname2 = plotting_dir / "gradient_norm_3d.png"
    fig2.write_image(figname2)
    print(f"3D figure saved as {figname2} ({time.time()-t1:.2f} seconds)")

    # Plot Boltzmann distribution surface.
    fig3 = go.Figure()
    fig3.add_trace(
        go.Surface(
            x=X_np,
            y=Y_np,
            z=p_xy,
            colorscale="Cividis",
            colorbar=dict(title="Boltzmann p(x,y)"),
            name="Boltzmann Distribution",
        )
    )
    fig3.update_layout(
        title="Boltzmann Distribution Surface",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="Probability Density",
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    figname3 = plotting_dir / "boltzmann_dist_3d.png"
    fig3.write_image(figname3)
    print(f"3D figure saved as {figname3} ({time.time()-t1:.2f} seconds)")


def plot_potential_energy_surface(pes_fn, cutoff=None, percentile=50):
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a 2D grid over the interval [-3, 3] for both x and y with 400 points each.
    x = torch.linspace(-3, 3, 400).to(device)
    y = torch.linspace(-3, 3, 400).to(device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    # Enable gradient tracking for both grid variables.
    X.requires_grad_()
    Y.requires_grad_()

    pes, pes_grad_norm = pes_fn(torch.stack([X, Y], dim=1))

    # Convert the tensors to NumPy arrays for plotting.
    X_np = X.cpu().detach().numpy()
    Y_np = Y.cpu().detach().numpy()
    pes_np = pes.cpu().detach().numpy()
    grad_norm_np = pes_grad_norm.cpu().detach().numpy()

    # Compute the Boltzmann distribution using the gradient norm as the pseudo_energy.
    beta = 1.0
    E = grad_norm_np
    boltz_factor = np.exp(-beta * E)
    # Estimate the integral (normalization constant) via summing with the grid area element.
    dx = x[1].item() - x[0].item()
    dy = y[1].item() - y[0].item()
    Z = boltz_factor.sum() * dx * dy
    p_xy = boltz_factor / Z

    plot_potential_energy_surface_2d(
        X_np, Y_np, pes_np, grad_norm_np, p_xy, cutoff=cutoff, percentile=percentile
    )
    plot_potential_energy_surface_3d(
        X_np, Y_np, pes_np, grad_norm_np, p_xy, cutoff=cutoff, percentile=percentile
    )


###########################################################################################################


def plot_results_2d(final_samples, X_grid, Y_grid, p_true):
    t1 = time.time()

    # Create histogram data.
    hist, x_edges, y_edges = np.histogram2d(
        final_samples[:, 0],
        final_samples[:, 1],
        bins=50,
        density=True,
    )
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    print(f"final_samples.shape: {final_samples.shape}")
    print(f"hist.shape: {hist.shape}")
    print(f"x_centers.shape: {x_centers.shape}")
    print(f"y_centers.shape: {y_centers.shape}")
    print(f"p_true.shape: {p_true.shape}")
    print(f"X_grid.shape: {X_grid.shape}")
    print(f"Y_grid.shape: {Y_grid.shape}")

    # Create figure.
    fig = go.Figure()

    # Add histogram heatmap of generated samples.
    fig.add_trace(
        go.Contour(
            x=x_centers,
            y=y_centers,
            z=hist.T,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Sample Density"),
            name="Generated Samples",
        )
    )

    # Add contour lines of target Boltzmann distribution.
    fig.add_trace(
        go.Contour(
            x=X_grid[0, :],
            y=Y_grid[:, 0],
            z=p_true,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            contours=dict(coloring="lines", showlabels=True),
            name="Target Boltzmann",
        )
    )

    # Update layout.
    fig.update_layout(
        title="Iterative Diffusion pseudo_energy Matching for Sampling<br>from the Boltzmann Distribution",
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=600,
        showlegend=True,
    )

    figname = plotting_dir / "diffusion_datafree.png"
    fig.write_image(figname)
    print(f"Figure saved as {figname} ({time.time()-t1:.2f} seconds)")


def plot_results_3d(final_samples, X_grid, Y_grid, p_true):
    # Plot using Plotly
    t1 = time.time()
    fig_plotly = go.Figure()

    # Add Boltzmann distribution surface
    fig_plotly.add_trace(
        go.Surface(
            x=X_grid[0, :],
            y=Y_grid[:, 0],
            z=p_true,
            colorscale="Viridis",
            colorbar=dict(title="Boltzmann p(x,y)"),
            name="Boltzmann Distribution",
        )
    )

    # Add generated samples as scatter points on the surface
    fig_plotly.add_trace(
        go.Scatter3d(
            x=final_samples[:, 0],
            y=final_samples[:, 1],
            z=np.zeros(final_samples.shape[0]),  # z=0 for visualization
            mode="markers",
            marker=dict(size=2, color="blue"),
            name="Generated Samples",
        )
    )

    # Update layout for better visualization
    fig_plotly.update_layout(
        title="Generated Samples and Boltzmann Distribution",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="p(x,y)"),
        width=800,
        height=600,
    )

    # Save the Plotly figure
    plotly_figname_png = plotting_dir / "diffusion_datafree_3d.png"
    fig_plotly.write_image(plotly_figname_png)
    print(f"Plotly 3D figure saved as {plotly_figname_png}")


def plot_results(final_samples):
    # ---------------------------
    # 7. Plot the Results
    # ---------------------------
    # Compute the true Boltzmann density over a 2D grid.
    x_grid = np.linspace(-3, 3, 400)
    y_grid = np.linspace(-3, 3, 400)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing="ij")
    # Compute the analytical gradient norm on the grid (using numpy) for f(x,y) = (x^2-1)^3+(y^2-1)^3.
    grad_x = 6 * X_grid * (X_grid**2 - 1) ** 2
    grad_y = 6 * Y_grid * (Y_grid**2 - 1) ** 2
    U_grid = np.sqrt(grad_x**2 + grad_y**2)
    p_true = np.exp(-U_grid)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    Z = p_true.sum() * dx * dy
    p_true = p_true / Z

    plot_results_2d(final_samples, X_grid, Y_grid, p_true)
    plot_results_3d(final_samples, X_grid, Y_grid, p_true)
