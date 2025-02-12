import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

from alanine_dipeptide import compute_energy_and_forces

def create_ramachandran_plot(
    phi_range=(-180, 180), psi_range=(-180, 180), resolution=36, 
    datafile='alanine_dipeptide/ramachandran.npy'
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
    
    # if file exists, load it
    if os.path.exists(datafile):
        energies = np.load(datafile)
    else:
        # Loop over grid points.
        for i, phi in enumerate(phi_values):
            for j, psi in enumerate(psi_values):
                # Convert angles from degrees to radians
                dihedrals = torch.tensor([phi * np.pi/180.0, psi * np.pi/180.0], dtype=torch.float32)
                
                # Compute the energy (and forces, which we ignore here)
                energy, _ = compute_energy_and_forces(dihedrals)
                
                # Store the energy (in kJ/mol)
                energies[i, j] = energy.item()
                
                # Optionally, print the computed energy for each grid point.
                print(f"phi={phi:6.1f}°, psi={psi:6.1f}° -> Energy = {energy.item():8.2f} kJ/mol")
                
        # Save the energies to a file
        np.save(datafile, energies)
    
    # Create a meshgrid for plotting (using 'ij' indexing so that phi_values index the first axis)
    phi_grid, psi_grid = np.meshgrid(phi_values, psi_values, indexing='ij')
    
    # Plot the contour (Ramachandran plot)
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(
        phi_grid, psi_grid, energies, levels=50, cmap='viridis', norm=LogNorm()
    )
    plt.xlabel('Phi (degrees)')
    plt.ylabel('Psi (degrees)')
    plt.title('Ramachandran Plot for Alanine Dipeptide')
    cbar = plt.colorbar(contour)
    cbar.set_label('Potential Energy (kJ/mol)')
    figname = 'alanine_dipeptide/ramachandran.png'
    plt.savefig(figname)
    plt.close()
    print(f"Saved {figname}")


# Example usage:
if __name__ == '__main__':
    # Ensure that compute_energy_and_forces is available in the current scope.
    # This will generate the Ramachandran plot over the full [-180, 180]° range for both φ and ψ.
    create_ramachandran_plot(phi_range=(-180, 180), psi_range=(-180, 180), resolution=36)
