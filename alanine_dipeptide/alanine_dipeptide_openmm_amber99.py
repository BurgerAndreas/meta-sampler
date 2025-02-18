import numpy as np
import torch
import concurrent.futures

# Import OpenMM modules (using simtk.openmm in OpenMM 7.x; for OpenMM 8+ use openmm)
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff, HBonds
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity

from dihedral import set_dihedral, compute_dihedral

"""Alanine dipeptide with OpenMM and Amber99 force field using dihedral angles as reaction coordinates.

Transforms dihedral angles to atom positions.
"""

# https://github.com/noegroup/bgflow/blob/fbba56fac3eb88f6825d2bd4f745ee75ae9715e1/tests/conftest.py#L54
# pdbfile = 'alanine_dipeptide/data/alanine_dipeptide.pdb'
pdbfile = "alanine_dipeptide/data/alanine_dipeptide_nowater.pdb"  # Frank Noe Boltzmann Generator

# https://github.com/openmm/openmm/blob/cd8f19c5195cd583f5dec181388e681651515a1a/wrappers/python/openmm/app/data/amber99sbildn.xml
fffile = "alanine_dipeptide/data/amber99sbildn.xml"  # Frank Noe Boltzmann Generator
# fffile = 'alanine_dipeptide/data/amber99sb.xml'
# https://github.com/openmm/openmmforcefields/blob/main/openmmforcefields/ffxml/amber/ff99SBildn.xml
# fffile = 'alanine_dipeptide/data/ff99SBildn.xml'


def build_alanine_dipeptide_from_file(phi=None, psi=None, _pdbfile=pdbfile):
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
    pdb = PDBFile(_pdbfile)

    # Extract positions as a NumPy array in nanometers.
    positions = np.array(pdb.positions.value_in_unit(nanometer))
    if phi is not None:
        positions = set_dihedral(positions, "phi", phi, "phi")
    if psi is not None:
        positions = set_dihedral(positions, "psi", psi, "psi")

    # Convert the modified positions back into an OpenMM Quantity (with units of nanometer).
    # (doesn't do anything)
    positions_quantity = Quantity(positions, nanometer)
    return positions_quantity


def compute_energy_and_forces_openmm(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor x of shape [2] (where x[0]=φ and x[1]=ψ, in radians), build an alanine dipeptide
    configuration, evaluate its energy and forces using a force field, and return these as torch tensors.

    Parameters:
      x: Torch tensor of shape [2] containing the dihedral angles (in radians)

    Returns:
      energy_tensor: Torch tensor (scalar) with the potential energy (in kJ/mol)
      forces_tensor: Torch tensor with shape (N_atoms, 3) containing the forces (in kJ/mol/nm)
    """
    # Load the template to obtain the topology.
    pdb = PDBFile(pdbfile)

    # Extract positions as a NumPy array in nanometers.
    positions = np.array(pdb.positions.value_in_unit(nanometer))
    if x is not None:
        # Extract φ and ψ from the input tensor.
        phi = x[0].item() if hasattr(x[0], "item") else float(x[0])
        psi = x[1].item() if hasattr(x[1], "item") else float(x[1])
        positions = set_dihedral(positions, "phi", phi, "phi")
        positions = set_dihedral(positions, "psi", psi, "psi")

    # Create the force field.
    # https://github.com/noegroup/bgflow/blob/fbba56fac3eb88f6825d2bd4f745ee75ae9715e1/tests/conftest.py#L54
    forcefield = ForceField(fffile)

    system = forcefield.createSystem(
        pdb.topology,
        # removes center-of-mass motion during the simulation.
        # prevents the system from drifting through space
        # by subtracting out any net momentum of the entire system
        removeCMMotion=True,
        # applies constraints to all bonds involving hydrogen atoms. This is a common practice in molecular dynamics because:
        # H-bonds vibrate at very high frequencies
        # These vibrations require very small timesteps to simulate accurately
        # Constraining them allows for larger timesteps without losing important physics
        constraints=HBonds,
        # treats water molecules as rigid bodies, meaning the internal geometry (bond lengths and angles) of water molecules cannot change. This is another common simplification that:
        # Reduces computational cost
        # Allows larger timesteps
        # Is physically reasonable since water's internal geometry changes very little at normal temperatures
        rigidWater=True,
        # how to handle non-bonded interactions (like electrostatics and van der Waals forces):
        # NoCutoff means all pairs of particles interact directly
        # No distance-based cutoff is applied
        # This is computationally expensive for large systems but exact
        # It's reasonable here because alanine dipeptide is a small molecule
        nonbondedMethod=NoCutoff,
    )

    # Set up a simulation (using the Reference platform for portability).
    integrator = VerletIntegrator(0.001 * picoseconds)
    platform = Platform.getPlatformByName("Reference")
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # Convert the modified positions back into an OpenMM Quantity (with units of nanometer).
    # (doesn't do anything)
    positions_quantity = Quantity(positions, nanometer)

    # Set the positions.
    simulation.context.setPositions(positions_quantity)

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
def compute_energy_and_forces_batch_openmm(
    x: torch.Tensor, num_workers: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
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
                results[idx] = (torch.tensor(float("nan")), torch.tensor([]))

    # Unpack results.
    energy_list, forces_list = zip(*results)
    energy_tensor = torch.stack(energy_list)  # Shape: [B]
    forces_tensor = torch.stack(forces_list)  # Shape: [B, N_atoms, 3]
    return energy_tensor, forces_tensor


# Example usage:
if __name__ == "__main__":
    # energy of the relaxed configuration
    energy, forces = compute_energy_and_forces_openmm(None)
    print(
        f"Energy relaxed configuration (kJ/mol): {energy.item():.2f} = {energy.item()*0.239006:.2f} kcal/mol"
    )
    print(f"Forces shape: {forces.shape}")

    print("-" * 60)
    # Suppose we want φ = -60° and ψ = -45° (in radians)
    dihedrals = torch.tensor(
        [-60 * np.pi / 180, -45 * np.pi / 180], dtype=torch.float32
    )
    energy, forces = compute_energy_and_forces_openmm(dihedrals)
    print(
        f"Energy φ = -60° and ψ = -45° (kJ/mol): {energy.item():.2f} = {energy.item()*0.239006:.2f} kcal/mol"
    )

    #################################################################################
    print("-" * 60)
    # Create a batch of dihedral angle pairs (in radians). For example, B=3.
    # Each row: [phi, psi]
    dihedrals_batch = torch.tensor(
        [
            [-60 * np.pi / 180, -45 * np.pi / 180],
            [-80 * np.pi / 180, 30 * np.pi / 180],
            [50 * np.pi / 180, 60 * np.pi / 180],
            [0 * np.pi / 180, 0 * np.pi / 180],
            [30 * np.pi / 180, 0 * np.pi / 180],
            [30 * np.pi / 180, 30 * np.pi / 180],
            [5 * np.pi / 180, 180 * np.pi / 180],
        ],
        dtype=torch.float32,
    )

    # Compute energies and forces in parallel.
    energies, forces = compute_energy_and_forces_batch_openmm(
        dihedrals_batch, num_workers=3
    )

    min_energy = float("inf")
    max_energy = float("-inf")

    # Display results.
    for i in range(dihedrals_batch.shape[0]):
        print(
            f"Configuration {i}: φ = {dihedrals_batch[i,0]*180/np.pi:.1f}°, ψ = {dihedrals_batch[i,1]*180/np.pi:.1f}°"
        )
        if energies[i].item() < min_energy:
            min_energy = energies[i].item()
        if energies[i].item() > max_energy:
            max_energy = energies[i].item()
        print(
            f"  Energy (kJ/mol): {energies[i].item():.2f} = {energies[i].item()*0.239006:.2f} kcal/mol"
        )

    print(
        f"Energy range: {(max_energy - min_energy):.2f} kJ/mol = {(max_energy - min_energy)*0.239006:.2f} kcal/mol"
    )

    #################################################################################
