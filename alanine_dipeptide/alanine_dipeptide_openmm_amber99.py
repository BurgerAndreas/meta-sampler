import numpy as np
import torch
import concurrent.futures

# Import OpenMM modules (using simtk.openmm in OpenMM 7.x; for OpenMM 8+ use openmm)
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity

"""Alanine dipeptide with OpenMM and Amber99 force field using dihedral angles as reaction coordinates.

Transforms dihedral angles to atom positions.
"""

from dihedral import set_dihedral, compute_dihedral

def build_alanine_dipeptide(phi, psi, pdbfile):
    """
    Build a full-atom 3D configuration of alanine dipeptide with the specified
    backbone dihedral angles phi and psi (in radians).
    
    The function loads a template PDB file and then
    rotates the appropriate groups of atoms so that the dihedrals match the desired values.
    
    Parameters:
      phi: Desired φ dihedral angle (radians)
      psi: Desired ψ dihedral angle (radians)
    
    Returns:
      positions_quantity: A simtk.unit.Quantity (shape: [N_atoms, 3]) in nanometers.
    
    > **Note:** Adjust the following dihedral atom indices to match your template.
    """
    # Load the template structure.
    pdb = PDBFile(pdbfile)
    # Extract positions as a NumPy array in nanometers.
    positions = np.array(pdb.positions.value_in_unit(nanometer))    
    positions = set_dihedral(positions, "phi", phi, "phi")
    positions = set_dihedral(positions, "psi", psi, "psi")
    
    # Convert the modified positions back into an OpenMM Quantity (with units of nanometer).
    # (doesn't do anything)
    positions_quantity = Quantity(positions, nanometer)
    return positions_quantity

def compute_energy_and_forces_openmm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor x of shape [2] (where x[0]=φ and x[1]=ψ, in radians), build an alanine dipeptide
    configuration, evaluate its energy and forces using a force field, and return these as torch tensors.
    
    Parameters:
      x: Torch tensor of shape [2] containing the dihedral angles (in radians)
    
    Returns:
      energy_tensor: Torch tensor (scalar) with the potential energy (in kJ/mol)
      forces_tensor: Torch tensor with shape (N_atoms, 3) containing the forces (in kJ/mol/nm)
    """
    # Extract φ and ψ from the input tensor.
    phi = x[0].item() if hasattr(x[0], 'item') else float(x[0])
    psi = x[1].item() if hasattr(x[1], 'item') else float(x[1])
    
    pdbfile = 'alanine_dipeptide/data/alanine_dipeptide_nowater.pdb'
    
    # Build the configuration (positions with units).
    positions = build_alanine_dipeptide(phi, psi, pdbfile)
    
    # Load the template again to obtain the topology.
    pdb = PDBFile(pdbfile)
    
    # Create the force field.
    forcefield = ForceField('alanine_dipeptide/amber99sb.xml')
    
    # Build the system. (Here we use NoCutoff for simplicity.)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
    
    # Set up a simulation (using the Reference platform for portability).
    integrator = VerletIntegrator(0.001 * picoseconds)
    platform = Platform.getPlatformByName('Reference')
    simulation = Simulation(pdb.topology, system, integrator, platform)
    
    # Set the positions.
    simulation.context.setPositions(positions)
    
    # Retrieve the state with energy and forces.
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole / nanometer)
    
    # Convert energy and forces to torch tensors.
    energy_tensor = torch.tensor(energy, dtype=torch.float32)
    forces_tensor = torch.tensor(forces, dtype=torch.float32)
    
    return energy_tensor, forces_tensor


# ------------------------------------------------------------------------------
# Batch version: compute multiple configurations in parallel.
# ------------------------------------------------------------------------------
def compute_energy_and_forces_batch_openmm(x: torch.Tensor, num_workers: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor x of shape [B, 2] (each row is a pair of dihedral angles in radians),
    compute the energy and forces for each configuration in parallel.
    
    Parameters:
      x           : Torch tensor of shape [B, 2] containing dihedral angles (radians)
      num_workers : Number of worker threads to use (default: 4)
      
    Returns:
      energy_tensor: Torch tensor of shape [B] containing energies (kJ/mol)
      forces_tensor: Torch tensor of shape [B, N_atoms, 3] containing forces (kJ/mol/nm)
    """
    B = x.shape[0]
    results = [None] * B

    def worker(i, dihedrals):
        # Each worker computes energy and forces for one configuration.
        return compute_energy_and_forces_openmm(dihedrals)

    # Use ThreadPoolExecutor to run computations in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit a separate job for each configuration.
        futures = {executor.submit(worker, i, x[i]): i for i in range(B)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"Configuration {idx} generated an exception: {exc}")
                results[idx] = (torch.tensor(float('nan')), torch.tensor([]))

    # Unpack results.
    energy_list, forces_list = zip(*results)
    energy_tensor = torch.stack(energy_list)  # Shape: [B]
    forces_tensor = torch.stack(forces_list)    # Shape: [B, N_atoms, 3]
    return energy_tensor, forces_tensor


# Example usage:
if __name__ == '__main__':
    # Suppose we want φ = -60° and ψ = -45° (in radians)
    dihedrals = torch.tensor([-60 * np.pi/180, -45 * np.pi/180], dtype=torch.float32)
    energy, forces = compute_energy_and_forces_openmm(dihedrals)
    print("Energy (kJ/mol):", energy.item())
    print("Forces shape:", forces.shape)
    
    #################################################################################
    
    # Create a batch of dihedral angle pairs (in radians). For example, B=3.
    # Each row: [phi, psi]
    dihedrals_batch = torch.tensor([
        [-60 * np.pi/180, -45 * np.pi/180],
        [ -80 * np.pi/180,  30 * np.pi/180],
        [  50 * np.pi/180,  60 * np.pi/180],
        [  0 * np.pi/180,  0 * np.pi/180],
        [  30 * np.pi/180,  0 * np.pi/180],
        [  30 * np.pi/180,  30 * np.pi/180],
        [  5 * np.pi/180,  180 * np.pi/180],
    ], dtype=torch.float32)
    
    # Compute energies and forces in parallel.
    energies, forces = compute_energy_and_forces_batch_openmm(dihedrals_batch, num_workers=3)
    
    # Display results.
    for i in range(dihedrals_batch.shape[0]):
        print(f"Configuration {i}: φ = {dihedrals_batch[i,0]*180/np.pi:.1f}°, ψ = {dihedrals_batch[i,1]*180/np.pi:.1f}°")
        print(f"  Energy (kJ/mol): {energies[i].item():.2f}")
        print(f"  Forces shape   : {forces[i].shape}")

    #################################################################################
