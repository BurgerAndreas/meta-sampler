import numpy as np
import torch

# Import OpenMM modules (using simtk.openmm in OpenMM 7.x; for OpenMM 8+ use openmm)
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix for a counterclockwise rotation about
    'axis' by angle 'theta' (in radians) using Rodrigues’ rotation formula.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta/2.0)
    b, c, d = -axis * np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
                     [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]])

def compute_dihedral(p0, p1, p2, p3):
    """
    Compute the dihedral angle (in radians) defined by four points.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so that it does not influence magnitude of vector
    b1 = b1 / np.linalg.norm(b1)
    
    # Compute the vectors normal to the planes defined by (p0,p1,p2) and (p1,p2,p3)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def set_dihedral(positions, indices, target_angle):
    """
    Adjust the dihedral angle defined by the four atoms specified in 'indices'
    to 'target_angle' (in radians) by rotating all atoms “downstream” of atom indices[3].
    
    Parameters:
      positions   : NumPy array of shape (N_atoms, 3)
      indices     : A list or tuple of 4 atom indices [i, j, k, l] defining the dihedral
      target_angle: The desired dihedral angle (in radians)
      
    Returns:
      positions   : Modified NumPy array of positions (in nanometers)
    
    > **Warning:** This is a simplified routine. In a production setting, one would determine
    > the proper set of atoms to rotate based on the molecular connectivity.
    """
    i, j, k, l = indices
    p0, p1, p2, p3 = positions[i].copy(), positions[j].copy(), positions[k].copy(), positions[l].copy()
    
    current_angle = compute_dihedral(p0, p1, p2, p3)
    delta = target_angle - current_angle

    # Define the rotation axis (passing through atoms j and k)
    axis = positions[k] - positions[j]
    axis /= np.linalg.norm(axis)
    
    # For simplicity, assume that all atoms with index >= l are to be rotated.
    rotating_indices = list(range(l, positions.shape[0]))
    origin = positions[k].copy()
    
    R = rotation_matrix(axis, delta)
    
    for idx in rotating_indices:
        vec = positions[idx] - origin
        positions[idx] = origin + np.dot(R, vec)
    
    return positions

def build_alanine_dipeptide(phi, psi):
    """
    Build a full-atom 3D configuration of alanine dipeptide with the specified
    backbone dihedral angles phi and psi (in radians).
    
    The function loads a template PDB file (alanine_dipeptide_template.pdb) and then
    rotates the appropriate groups of atoms so that the dihedrals match the desired values.
    
    Parameters:
      phi: Desired φ dihedral angle (radians)
      psi: Desired ψ dihedral angle (radians)
    
    Returns:
      positions_quantity: A simtk.unit.Quantity (shape: [N_atoms, 3]) in nanometers.
    
    > **Note:** Adjust the following dihedral atom indices to match your template.
    """
    # Load the template structure.
    pdb = PDBFile('alanine_dipeptide/alanine_dipeptide.pdb')
    # Extract positions as a NumPy array in nanometers.
    positions = np.array(pdb.positions.value_in_unit(nanometer))
    
    # Set the dihedral angles.
    # (The following indices are examples. For instance, if the template has the backbone atoms in order,
    #  you might define φ = dihedral(atoms[1,2,3,4]) and ψ = dihedral(atoms[2,3,4,5]).)
    phi_indices = [1, 2, 3, 4]  # <-- Adjust these indices to your template.
    psi_indices = [2, 3, 4, 5]  # <-- Adjust these indices to your template.
    
    positions = set_dihedral(positions, phi_indices, phi)
    positions = set_dihedral(positions, psi_indices, psi)
    
    # Convert the modified positions back into an OpenMM Quantity (with units of nanometer).
    positions_quantity = Quantity(positions, nanometer)
    return positions_quantity

def compute_energy_and_forces(x):
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
    
    # Build the configuration (positions with units).
    positions = build_alanine_dipeptide(phi, psi)
    
    # Load the template again to obtain the topology.
    pdb = PDBFile('alanine_dipeptide/alanine_dipeptide.pdb')
    
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

# Example usage:
if __name__ == '__main__':
    # Suppose we want φ = -60° and ψ = -45° (in radians)
    dihedrals = torch.tensor([-60 * np.pi/180, -45 * np.pi/180], dtype=torch.float32)
    energy, forces = compute_energy_and_forces(dihedrals)
    print("Energy (kJ/mol):", energy.item())
    print("Forces shape:", forces.shape)
