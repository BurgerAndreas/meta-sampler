
import copy
import io
import numpy as np
import py3Dmol
from Bio.PDB import PDBParser, PDBIO
import numpy as np
import time
import plotly.graph_objects as go
import os
from tqdm import tqdm

import torch
import nglview as nv
import py3Dmol
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity

import numpy as np
from Bio.PDB import PDBParser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import imageio


from dihedral import set_dihedral, compute_dihedral


def pdb_to_xyz(pdbfile):
    """Convert PDB file to xyz coordinates, atom types, and atom info"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdbfile)
    
    # Get coordinates, atom types and atom info
    coords = []
    atom_types = []
    atom_info = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
                    atom_types.append(atom.element)
                    atom_info.append(f"{atom.get_name()} from {residue.get_resname()} (residue {residue.get_id()[1]})")
                    
    coords = np.array(coords)
    
    return coords, atom_types, atom_info

def find_bonds(coords, atom_types, max_dist=2.0):
    """Find bonds between atoms based on distance criteria"""
    bonds = []
    n_atoms = len(coords)
    
    # Typical bond lengths (in Angstroms)
    bond_lengths = {
        ('C', 'C'): 1.54,
        ('C', 'N'): 1.47,
        ('C', 'O'): 1.43,
        ('C', 'H'): 1.09,
        ('N', 'H'): 1.01,
        ('O', 'H'): 0.96
    }
    
    # Make bond lengths symmetric
    for (a1, a2), dist in list(bond_lengths.items()):
        bond_lengths[(a2, a1)] = dist
    
    # Calculate distances between all pairs of atoms
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # Skip if both atoms are hydrogen
            if atom_types[i] == 'H' and atom_types[j] == 'H':
                continue
                
            dist = np.linalg.norm(coords[i] - coords[j])
            
            # Get expected bond length
            expected_length = bond_lengths.get((atom_types[i], atom_types[j]), 1.5)
            
            # Add bond if distance is within tolerance of expected length
            if dist < expected_length * 1.3:  # 30% tolerance
                bonds.append((i, j))
                
    return bonds


def ramachandran_plot(pdbfile, phi_indices, psi_indices, recompute=True):
    # Create a Ramachandran plot
    pdb = PDBFile(pdbfile)
    # Extract positions as a NumPy array in nanometers.
    positions_default = np.array(pdb.positions.value_in_unit(nanometer))


    # Create the force field.
    forcefield = ForceField('amber99sb.xml')

    # Build the system. (Here we use NoCutoff for simplicity.)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)

    # Set up a simulation (using the Reference platform for portability).
    integrator = VerletIntegrator(0.001 * picoseconds)
    platform = Platform.getPlatformByName('Reference')
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # plot resolution
    resolution = 36
    phi_values = np.linspace(-180, 180, resolution)
    psi_values = np.linspace(-180, 180, resolution)

    datafile = 'alanine_dipeptide/outputs/ramachandran_openmm_amber.npy'

    # if file exists, load it
    if os.path.exists(datafile) and not recompute:
        energies, forces_norm, forces_normmean = np.load(datafile)
    else:
        # Initialize an array to store energies.
        energies = np.zeros((len(phi_values), len(psi_values)))
        forces_norm = np.zeros((len(phi_values), len(psi_values)))
        forces_normmean = np.zeros((len(phi_values), len(psi_values)))

        # Loop over grid points.
        for i, phi in tqdm(enumerate(phi_values), total=len(phi_values)):
            for j, psi in enumerate(psi_values):
                # Convert angles from degrees to radians
                dihedrals = torch.tensor([phi * np.pi/180.0, psi * np.pi/180.0], dtype=torch.float32)
                
                # Compute the energy and forces
                # Set the positions.
                positions = positions_default.copy()
                positions = set_dihedral(positions, "phi", phi, "phi")
                positions = set_dihedral(positions, "psi", psi, "psi")
                
                # build system from scratch?
                # Build the system. (Here we use NoCutoff for simplicity.)
                system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
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
                energy = torch.tensor(energy, dtype=torch.float32)
                force = torch.tensor(forces, dtype=torch.float32)   
                
                # Store the energy (in kJ/mol)
                energies[i, j] = energy.item()
                # force are shape [22,3] each row is a force vector for an atom
                forces_norm[i, j] = torch.linalg.norm(force).item()
                forces_normmean[i, j] = torch.linalg.norm(force, axis=1).mean().item()
                
                # Optionally, print the computed energy for each grid point.
                tqdm.write(f"phi={phi:6.1f}°({dihedrals[0]:.1f}°), psi={psi:6.1f}°({dihedrals[1]:.1f}°) -> U={energy.item():8.2f} kJ/mol, F={forces_norm[i, j]:8.2f}")
                
        # Save the energies to a file
        np.save(datafile, (energies, forces_norm, forces_normmean))

    energies = np.clip(energies, 1e-3, None)

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
        title=r'$\text{Ramachandran Plot for Alanine Dipeptide: } \log_{10}(U)$',
        xaxis_title='Phi (degrees)', 
        yaxis_title='Psi (degrees)',
        margin=dict(l=0, r=0, t=50, b=0),
    )
    figname = 'alanine_dipeptide/plots/ramachandran_plotly.png'
    fig.write_image(figname)
    print(f"Saved {figname}")
    # fig.show()

def plot_default_structure(coords, atom_types, atom_info, bonds, colors, phi_indices=None, psi_indices=None):
    
    # Create scatter plot for atoms
    atoms_trace = go.Scatter3d(
        x=coords[:,0],
        y=coords[:,1], 
        z=coords[:,2],
        mode='markers',
        marker=dict(
            size=10,
            color=[colors.get(atom, 'gray') for atom in atom_types],
            opacity=0.8
        ),
        text=[f"Atom {i}: {info}" for i, info in enumerate(atom_info)],
        hoverinfo='text'
    )

    # Create lines for bonds
    bond_traces = []
    for bond in bonds:
        i, j = bond
        bond_traces.append(
            go.Scatter3d(
                x=[coords[i,0], coords[j,0]],
                y=[coords[i,1], coords[j,1]],
                z=[coords[i,2], coords[j,2]],
                mode='lines',
                line=dict(color='gray', width=2),
                hoverinfo='none'
            )
        )
    
    if phi_indices is not None:
        # Add markers for phi indices
        phi_trace = go.Scatter3d(
            x=coords[phi_indices,0],
            y=coords[phi_indices,1],
            z=coords[phi_indices,2],
            mode='markers+text',
            marker=dict(
                size=7,
                color='green',
                symbol='diamond',
                opacity=1
            ),
            text=[f'φ{_i}' for _i in range(len(phi_indices))],
            textposition='top center',
            name='Phi indices'
        )

    if psi_indices is not None:
        # Add markers for psi indices
        psi_trace = go.Scatter3d(
            x=coords[psi_indices,0],
            y=coords[psi_indices,1],
            z=coords[psi_indices,2],
            mode='markers+text',
            marker=dict(
                size=7,
                color='orange',
                symbol='cross', # options: diamond, circle, square, cross, x
                opacity=1
                ),
                text=[f'ψ{_i}' for _i in range(len(psi_indices))],
                textposition='top center',
                name='Psi indices',
            )

    # Create figure
    traces = [atoms_trace] + bond_traces
    if phi_trace is not None:
        traces.append(phi_trace)
    if psi_trace is not None:
        traces.append(psi_trace)
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title='Alanine Dipeptide Structure',
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)'
        ),
        showlegend=False
    )

    fig.show()
    figname = 'alanine_dipeptide/plots/alanine_dipeptide_structure.png'
    fig.write_image(figname)
    print(f"Saved {figname}")

def plot_rotated_structure(coords, atom_types, atom_info, bonds, colors, phi_indices, psi_indices):
    # Make a copy of original coordinates for comparison
    coords_original = coords.copy()
    positions = coords.copy()
    p0_phi, p1_phi, p2_phi, p3_phi = positions[phi_indices]
    p0_psi, p1_psi, p2_psi, p3_psi = positions[psi_indices]
    phi_default = compute_dihedral(p0_phi, p1_phi, p2_phi, p3_phi)
    psi_default = compute_dihedral(p0_psi, p1_psi, p2_psi, p3_psi)
    print("default angles:")
    print(f"Phi φ = {phi_default*180/np.pi:.1f}°, Psi ψ = {psi_default*180/np.pi:.1f}°")

    # Rotate phi by ... degrees (convert to radians)
    phi_offset = np.pi/180 * 0
    phi_offset = -phi_default
    phi_rotation = phi_default + phi_offset
    
    # Rotate psi
    psi_offset = np.pi/180 * 0
    # psi_offset = -psi_default
    psi_rotation = psi_default + psi_offset
    
    coords_rotated = coords.copy()
    coords_rotated = set_dihedral(coords_rotated, phi_indices, phi_rotation, "phi")
    coords_rotated = set_dihedral(coords_rotated, psi_indices, psi_rotation, "psi")

    # Create scatter plot for original structure (transparent)
    atoms_trace_original = go.Scatter3d(
        x=coords_original[:,0],
        y=coords_original[:,1], 
        z=coords_original[:,2],
        mode='markers',
        marker=dict(
            size=10,
            color=[colors.get(atom, 'gray') for atom in atom_types],
            opacity=0.3
        ),
        text=[f"Original {i}: {info}" for i, info in enumerate(atom_info)],
        hoverinfo='text'
    )

    # Create scatter plot for rotated structure
    atoms_trace_rotated = go.Scatter3d(
        x=coords_rotated[:,0],
        y=coords_rotated[:,1], 
        z=coords_rotated[:,2],
        mode='markers',
        marker=dict(
            size=10,
            color=[colors.get(atom, 'gray') for atom in atom_types],
            opacity=0.8
        ),
        text=[f"Rotated {i}: {info}" for i, info in enumerate(atom_info)],
        hoverinfo='text'
    )

    # Create bond traces for original structure (transparent)
    bond_traces_original = []
    for bond in bonds:
        i, j = bond
        bond_traces_original.append(
            go.Scatter3d(
                x=[coords_original[i,0], coords_original[j,0]],
                y=[coords_original[i,1], coords_original[j,1]],
                z=[coords_original[i,2], coords_original[j,2]],
                mode='lines',
                line=dict(color='gray', width=5),
                hoverinfo='none', opacity=0.3
            )
        )

    # Create bond traces for rotated structure
    bond_traces_rotated = []
    for bond in bonds:
        i, j = bond
        bond_traces_rotated.append(
            go.Scatter3d(
                x=[coords_rotated[i,0], coords_rotated[j,0]],
                y=[coords_rotated[i,1], coords_rotated[j,1]],
                z=[coords_rotated[i,2], coords_rotated[j,2]],
                mode='lines',
                line=dict(color='gray', width=5),
                hoverinfo='none'
            )
        )
    
    # # Add markers for phi indices
    # phi_trace = go.Scatter3d(
    #     x=coords_original[phi_indices,0],
    #     y=coords_original[phi_indices,1],
    #     z=coords_original[phi_indices,2],
    #     mode='markers+text',
    #     marker=dict(
    #         size=15,
    #         color='green',
    #         symbol='diamond',
    #         opacity=1
    #     ),
    #     text=[f'φ{i}' for i in range(4)],
    #     textposition='top center',
    #     name='Phi indices'
    # )

    # # Add markers for psi indices
    # psi_trace = go.Scatter3d(
    #     x=coords_original[psi_indices,0],
    #     y=coords_original[psi_indices,1],
    #     z=coords_original[psi_indices,2],
    #     mode='markers+text',
    #     marker=dict(
    #         size=15,
    #         color='orange',
    #         symbol='diamond',
    #         opacity=1
    #     ),
    #     text=[f'ψ{i}' for i in range(4)],
    #     textposition='top center',
    #     name='Psi indices',
    # )

    # Create figure with both structures
    fig = go.Figure(
        data=[atoms_trace_original, atoms_trace_rotated] + 
                        bond_traces_original + bond_traces_rotated #+ [phi_trace, psi_trace]
        )

    # Update layout
    fig.update_layout(
        title=f'Alanine Dipeptide - Original (transparent) vs Rotated (Phi φ={phi_rotation*180/np.pi:.1f}°, Psi ψ={psi_rotation*180/np.pi:.1f}°)',
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)', 
            zaxis_title='Z (Å)'
        ),
        showlegend=False, margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.show()
    figname = 'alanine_dipeptide/plots/alanine_dipeptide_phi_rotated.png'
    fig.write_image(figname)
    print(f"Saved {figname}")

# Ugly
def animate_dihedral_rotation(phi_indices, pdbfile):
    pdbfile = 'alanine_dipeptide/data/alanine_dipeptide_nowater.pdb'
    pdb = PDBFile(pdbfile)
    
    # Extract positions as a NumPy array in nanometers.
    positions = np.array(pdb.positions.value_in_unit(nanometer))
    
    # Set the dihedral angles.
    # (The following indices are examples. For instance, if the template has the backbone atoms in order,
    #  you might define φ = dihedral(atoms[1,2,3,4]) and ψ = dihedral(atoms[2,3,4,5]).)
    # phi_indices = [1, 2, 3, 4] 
    # xrange = [-0.4, 1.2]
    # yrange = [-2., 3.]
    # zrange = [-0.3, 1.]
    
    phi_indices = (5, 7, 9, 15)
    xrange = [-0.4, 1.2]
    yrange = [-2., 3.]
    zrange = [-0.3, 0.4]
    
    psi_indices = [2, 3, 4, 5] 
    
    current_angles = compute_dihedral(positions[phi_indices[0]], positions[phi_indices[1]], positions[phi_indices[2]], positions[phi_indices[3]])
    print("default angles:")
    print(f"φ = {current_angles*180/np.pi:.1f}°, ψ = {current_angles*180/np.pi:.1f}°")
    phi_default = current_angles
    
    # get atom types
    atom_types = [atom.name for atom in pdb.topology.atoms()]
    print(atom_types)
    
    # get atom indices
    atom_indices = [atom.index for atom in pdb.topology.atoms()]
    print(atom_indices)
    
    # get indices of heavy atoms
    heavy_atom_indices = [i for i, atom in enumerate(atom_types) if atom[0] != 'H']
    print(heavy_atom_indices)
    
    # plot the atoms using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=positions[:,0], y=positions[:,1], z=positions[:,2], mode='markers'))
    fig.write_image("alanine_dipeptide/plots/default.png")
    
    # Create directory for temporary PNGs if it doesn't exist
    os.makedirs("alanine_dipeptide/plots/temp", exist_ok=True)

    # Create frames for animation
    frames = []
    png_files = []  # Store filenames for PNG frames

    # Define colors for each atom type
    color_dict = {
        'H': '#FFFFFF',   # White
        'C': '#808080',   # Gray
        'N': '#0000FF',   # Blue
        'O': '#FF0000',   # Red
    }

    # Get atom colors based on their types
    colors = [color_dict[sym[0]] for sym in atom_types]  # sym[0] takes first letter of atom name

    # 36 frames for 360 degrees
    num_frames = 36
    for i, phi in tqdm(enumerate(np.linspace(0, 2*np.pi, num_frames)), total=num_frames):  
        pos = positions.copy()
        pos = set_dihedral(pos, "phi", phi_default+phi, "phi")
            
        # Create figure for this frame
        fig = go.Figure(data=[go.Scatter3d(
            x=pos[:,0], 
            y=pos[:,1], 
            z=pos[:,2], 
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.8
            ),
            text=atom_types,  # hover text will show atom names
            hoverinfo='text'
        )])
        
        # draw lines connecting the heavy atoms
        fig.add_trace(go.Scatter3d(
            x=pos[heavy_atom_indices,0], 
            y=pos[heavy_atom_indices,1], 
            z=pos[heavy_atom_indices,2], 
            mode='lines',
            line=dict(color='#000000', width=2),
            hoverinfo='none'
        ))
        
        # Set axis limits and other layout properties
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=xrange, autorange=False),
                yaxis=dict(range=yrange, autorange=False), 
                zaxis=dict(range=zrange, autorange=False),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1, y=-2, z=0),  # Position camera along y-axis looking at x-z plane
                    up=dict(x=0, y=0, z=1)     # Set up vector to align z-axis vertically
                )
            ),
            # width=1000,
            # height=1000,
            title=f"φ = {phi*180/np.pi:.1f}°",
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        
        # Save as PNG
        png_file = f"alanine_dipeptide/plots/temp/frame_{i:03d}.png"
        fig.write_image(png_file)
        png_files.append(png_file)
    
    # Create GIF from PNGs
    with imageio.get_writer(
        f"alanine_dipeptide/plots/phi_rotation_{'_'.join([str(i) for i in phi_indices])}.gif", 
        mode="I", 
        duration=300 
        ) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)

    # Clean up temporary PNG files
    for png_file in png_files:
        os.remove(png_file)

    # # Save interactive HTML version
    # fig = go.Figure()
    # fig.add_trace(go.Scatter3d(
    #     x=positions[:,0], 
    #     y=positions[:,1], 
    #     z=positions[:,2], 
    #     mode='markers'
    # ))
    # fig.write_html(f"alanine_dipeptide/plots/phi_rotation_{'_'.join([str(i) for i in phi_indices])}.html")
    

if __name__ == "__main__":
    pdbfile = "alanine_dipeptide/data/alanine_dipeptide_nowater.pdb"


    # Convert PDB to xyz coordinates
    coords, atom_types, atom_info = pdb_to_xyz(pdbfile)

    # Find bonds
    bonds = find_bonds(coords, atom_types)

    # Save coordinates to xyz file
    with open('alanine_dipeptide.xyz', 'w') as f:
        f.write(f"{len(coords)}\n")
        f.write("Alanine dipeptide structure\n")
        for atom_type, coord in zip(atom_types, coords):
            f.write(f"{atom_type:2s} {coord[0]:10.6f} {coord[1]:10.6f} {coord[2]:10.6f}\n")

    # Color scheme for atoms
    colors = {'H':'lightgray', 'C':'black', 'N':'blue', 'O':'red'}


    #######################################################################
    # Rotate dihedral angles
    #######################################################################

    # Set the dihedral angles.
    # A dihedral angle is defined by four atoms: p0, p1, p2, and p3.
    # When you change a dihedral angle, you rotate a subset of the molecule around the bond between atoms p1 and p2. Specifically:
    #     p0, p1, p2, and p3 define the plane and angle.
    #     The bond between p1 and p2 is the axis of rotation.
    #     Atoms p0 and p1 remain fixed (i.e., they do not move).
    #     Atom p2 is the pivot point, meaning it stays in place.
    #     Atom p3 and all atoms connected to p3 (and beyond) rotate as a rigid unit around the p1-p2 axis.
    # This means that when you set the dihedral angle, you are only rotating the part of the molecule that is "downstream" from p3, leaving the rest of the structure unchanged

    # Andreas invention
    # phi_indices = [CB, CA, NL, CLP] = [10, 8, 6, 4] 
    # psi_indices = [CB, CA, CRP, OR] = [10, 8, 14, 15] 
    # phi_indices = [10, 8, 6, 4]  
    # psi_indices = [10, 8, 14, 15]  
    from dihedral import phi_indices, psi_indices

    
    #######################################################################
    # Visualize default structure using plotly
    #######################################################################
    # plot_default_structure(coords, atom_types, atom_info, bonds, colors, phi_indices, psi_indices)

    
    #######################################################################
    # Visualize rotation using py3Dmol 
    #######################################################################
    # plot_rotated_structure(coords, atom_types, atom_info, bonds, colors, phi_indices, psi_indices)


    #######################################################################
    # Ramachandran plot
    #######################################################################

    ramachandran_plot(pdbfile, phi_indices, psi_indices)
    
    #######################################################################