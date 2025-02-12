import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

from alanine_dipeptide import compute_energy_and_forces

def create_ramachandran_plot(
    phi_range=(-180, 180), psi_range=(-180, 180), resolution=36, 
    datafile='alanine_dipeptide/ramachandran.npy',
    recompute=False,
    ):
    """
    Generate a Ramachandran plot for the alanine dipeptide energy landscape.
    
    This function samples the φ and ψ dihedral angles over the specified ranges (in degrees)
    and computes the corresponding potential energy using the compute_energy_and_forces function.
    A contour plot of the energy (in kJ/mol) is then displayed.
    
    Parameters:
        phi_range (tuple): (min_phi, max_phi) in degrees (default (-180, 180)).
        psi_range (tuple): (min_psi, max_psi) in degrees (default (-180, 180)).
        resolution (int): Number of grid points along each dihedral axis (default 36).
    
    Returns:
        None. Displays a contour plot.
    """
    # Create arrays for φ and ψ (in degrees)
    phi_values = np.linspace(phi_range[0], phi_range[1], resolution)
    psi_values = np.linspace(psi_range[0], psi_range[1], resolution)
    
    # Initialize an array to store energies.
    energies = np.zeros((resolution, resolution))
    forces_norm = np.zeros((resolution, resolution))
    forces_normmean = np.zeros((resolution, resolution))
    
    # if file exists, load it
    if os.path.exists(datafile) and not recompute:
        energies, forces_norm, forces_normmean = np.load(datafile)
    else:
        # Loop over grid points.
        for i, phi in tqdm(enumerate(phi_values), total=len(phi_values)):
            for j, psi in enumerate(psi_values):
                # Convert angles from degrees to radians
                dihedrals = torch.tensor([phi * np.pi/180.0, psi * np.pi/180.0], dtype=torch.float32)
                
                # Compute the energy and forces
                energy, force = compute_energy_and_forces(dihedrals)
                
                # Store the energy (in kJ/mol)
                energies[i, j] = energy.item()
                # force are shape [22,3] each row is a force vector for an atom
                forces_norm[i, j] = torch.linalg.norm(force).item()
                forces_normmean[i, j] = torch.linalg.norm(force, axis=1).mean().item()
                
                # Optionally, print the computed energy for each grid point.
                tqdm.write(f"phi={phi:6.1f}°, psi={psi:6.1f}° -> U={energy.item():8.2f} kJ/mol, F={forces_norm[i, j]:8.2f}")
                
        # Save the energies to a file
        np.save(datafile, (energies, forces_norm, forces_normmean))
        
    # clip negative values to smallest positive value since we are using a log scale
    energies = np.clip(energies, np.min(energies[energies > 0]), None)
    
    # Create a meshgrid for plotting (using 'ij' indexing so that phi_values index the first axis)
    phi_grid, psi_grid = np.meshgrid(phi_values, psi_values, indexing='ij')
    
    # Plot the contour (Ramachandran plot)
    
    # plt.figure(figsize=(10, 8))
    # contour = plt.contourf(
    #     phi_grid, psi_grid, energies, levels=50, cmap='viridis', norm=LogNorm()
    # )
    # plt.xlabel('Phi (degrees)')
    # plt.ylabel('Psi (degrees)')
    # plt.title('Ramachandran Plot for Alanine Dipeptide')
    # cbar = plt.colorbar(contour)
    # cbar.set_label('Potential Energy (kJ/mol)')
    # figname = 'alanine_dipeptide/plots/ramachandran.png'
    # plt.savefig(figname)
    # plt.close()
    # print(f"Saved {figname}")
    
    # same plot again but with plotly
    fig = go.Figure(data=go.Contour(
        x=phi_values, y=psi_values, 
        z=np.log10(energies).T,
        colorscale='Viridis',
        type='contour',
        colorbar=dict(
            title="[kJ/mol]",
        )
    ))
    fig.update_layout(
        title=r'\text{Ramachandran Plot for Alanine Dipeptide: } \log_{10}(U)',
        xaxis_title='Phi (degrees)', 
        yaxis_title='Psi (degrees)',
        margin=dict(l=0, r=0, t=50, b=0),
    )
    figname = 'alanine_dipeptide/plots/ramachandran_plotly.png'
    fig.write_image(figname)
    print(f"Saved {figname}")
    
    plot_tps = None # clusters, <int>, none
    _forces = forces_normmean
    
    if plot_tps == 'clusters':
        # find values where forces are approximately zero
        # Prepare points for clustering (only use points where force is close to zero)
        close_to_zero_mask = _forces < np.percentile(_forces, 10)  # Adjust percentile as needed
        points = np.column_stack((phi_grid[close_to_zero_mask], psi_grid[close_to_zero_mask]))
        
        # Apply KMeans clustering
        n_clusters = 1  # Adjust this number based on how many clusters you expect
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(points)
        
        # Get cluster centers
        zero_forces = kmeans.cluster_centers_
        print(f"Low force cluster centers: {zero_forces}")
    elif isinstance(plot_tps, int):
        # find the lowest force points
        zero_forces_idx = np.argsort(_forces.flatten())[:plot_tps]
        phi_flat = phi_grid.flatten()
        psi_flat = psi_grid.flatten()
        zero_forces = np.column_stack((phi_flat[zero_forces_idx], psi_flat[zero_forces_idx]))
        print(f"Smallest forces: {zero_forces}")
    else:
        zero_forces = None
    
    # plot force norm
    fig = go.Figure(data=go.Contour(
        x=phi_values, y=psi_values, 
        z=np.log10(forces_norm).T,
        colorscale='Viridis',
        type='contour',
        colorbar=dict(
            title="[kJ/mol]",
        )
    ))
    if zero_forces is not None:
        fig.add_trace(go.Scatter(
            x=zero_forces[:, 0], y=zero_forces[:, 1],
            mode='markers',
            marker=dict(color='red', size=10),
            # name='Zero Forces',
        ))
    fig.update_layout(
        title=r"$\text{Norm Force Plot for Alanine Dipeptide } \log_{10}(|F|)$",
        xaxis_title='Phi (degrees)', 
        yaxis_title='Psi (degrees)',
        margin=dict(l=0, r=5, t=50, b=0),
    )
    figname = 'alanine_dipeptide/plots/ramachandran_force_norm_plotly.png'
    fig.write_image(figname)
    print(f"Saved {figname}")
    
    # plot force norm mean
    fig = go.Figure(data=go.Contour(
        x=phi_values, y=psi_values, 
        z=np.log10(forces_normmean).T,
        colorscale='Viridis',
        type='contour',
        colorbar=dict(
            title="[kJ/mol]",
        )
    ))
    fig.update_layout(
        title=r"$\text{Mean Norm Force Plot for Alanine Dipeptide } \log_{10}(\frac{1}{22}\sum_{i=1}^{22}|F_i|)$",
        xaxis_title='Phi (degrees)', 
        yaxis_title='Psi (degrees)',
        margin=dict(l=0, r=5, t=50, b=0),
    )
    figname = 'alanine_dipeptide/plots/ramachandran_force_norm_mean_plotly.png'
    fig.write_image(figname)
    print(f"Saved {figname}")
    
    
# Example usage:
if __name__ == '__main__':
    # Ensure that compute_energy_and_forces is available in the current scope.
    # This will generate the Ramachandran plot over the full [-180, 180]° range for both φ and ψ.
    create_ramachandran_plot(
        phi_range=(-180, 180), 
        psi_range=(-180, 180), 
        resolution=100,
        recompute=False,
    )
