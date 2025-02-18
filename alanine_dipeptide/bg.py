import os
import tempfile
import numpy as np

import openmm as mm
import openmm.unit as unit
import openmm.app 
from openmm import LangevinIntegrator

import mdtraj as md
from mdtraj.utils import ensure_type

from bgmol.systems.base import OpenMMToolsTestSystem, OpenMMSystem
from bgmol.tpl.download import download_url

from bgmol.datasets.base import DataSet
from bgmol.api import system_by_name
from bgmol.tpl.hdf5 import HDF5TrajectoryFile, load_hdf5

import bgflow as bg

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import binned_statistic_2d

import torch


#############################################################################################################
# https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/distance.py
#############################################################################################################

def _reduce_box_vectors(vectors):
    """Make sure box vectors are in reduced form."""
    (bv1, bv2, bv3) = vectors
    bv3 -= bv2 * round(bv3[1] / bv2[1])
    bv3 -= bv1 * round(bv3[0] / bv1[0])
    bv2 -= bv1 * round(bv2[0] / bv1[0])
    return (bv1, bv2, bv3)


def _distance_mic(xyz, pairs, box_vectors, orthogonal):
    """Distance between pairs of points in each frame under the minimum image
    convention for periodic boundary conditions.

    The computation follows scheme B.9 in Tukerman, M. "Statistical
    Mechanics: Theory and Molecular Simulation", 2010.

    This is a slow pure python implementation, mostly for testing.
    """
    out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
    for i in range(len(xyz)):
        bv1, bv2, bv3 = _reduce_box_vectors(box_vectors[i].T)

        for j, (a, b) in enumerate(pairs):
            r12 = xyz[i, b, :] - xyz[i, a, :]
            r12 -= bv3 * round(r12[2] / bv3[2])
            r12 -= bv2 * round(r12[1] / bv2[1])
            r12 -= bv1 * round(r12[0] / bv1[0])
            dist = np.linalg.norm(r12)
            if not orthogonal:
                for ii in range(-1, 2):
                    v1 = bv1 * ii
                    for jj in range(-1, 2):
                        v12 = bv2 * jj + v1
                        for kk in range(-1, 2):
                            new_r12 = r12 + v12 + bv3 * kk
                            dist = min(dist, np.linalg.norm(new_r12))
            out[i, j] = dist
    return out

def _displacement(xyz, pairs):
    "Displacement vector between pairs of points in each frame"
    value = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
    assert value.shape == (
        xyz.shape[0],
        pairs.shape[0],
        3,
    ), f"v.shape {str(value.shape)}, xyz.shape {str(xyz.shape)}, pairs.shape {str(pairs.shape)}"
    return value


def _displacement_mic(xyz, pairs, box_vectors, orthogonal):
    """Displacement vector between pairs of points in each frame under the
    minimum image convention for periodic boundary conditions.

    The computation follows scheme B.9 in Tukerman, M. "Statistical
    Mechanics: Theory and Molecular Simulation", 2010.

    This is a very slow pure python implementation, mostly for testing.
    """
    out = np.empty((xyz.shape[0], pairs.shape[0], 3), dtype=np.float32)
    for i in range(len(xyz)):
        bv1, bv2, bv3 = _reduce_box_vectors(box_vectors[i].T)
        # hinv, not used
        _ = np.linalg.inv(np.array([bv1, bv2, bv3]).T)

        for j, (a, b) in enumerate(pairs):
            r12 = xyz[i, b, :] - xyz[i, a, :]
            r12 -= bv3 * round(r12[2] / bv3[2])
            r12 -= bv2 * round(r12[1] / bv2[1])
            r12 -= bv1 * round(r12[0] / bv1[0])
            min_disp = r12
            dist2 = (r12 * r12).sum()
            if not orthogonal:
                for ii in range(-1, 2):
                    v1 = bv1 * ii
                    for jj in range(-1, 2):
                        v12 = bv2 * jj + v1
                        for kk in range(-1, 2):
                            tmp = r12 + v12 + bv3 * kk
                            new_dist2 = (tmp * tmp).sum()
                            if new_dist2 < dist2:
                                dist2 = new_dist2
                                min_disp = tmp
            out[i, j] = min_disp

    return out

def compute_displacements(traj, atom_pairs, periodic=True, opt=False):
    """Compute the displacement vector between pairs of atoms in each frame of a trajectory.

    Parameters
    ----------
    traj : Trajectory
        Trajectory to compute distances in
    atom_pairs : np.ndarray, shape[num_pairs, 2], dtype=int
        Each row gives the indices of two atoms.
    periodic : bool, default=True
        If `periodic` is True and the trajectory contains unitcell
        information, we will compute distances under the minimum image
        convention.
    opt : bool, default=True
        Use an optimized native library to calculate distances. Our
        optimized minimum image convention calculation implementation is
        over 1000x faster than the naive numpy implementation.

    Returns
    -------
    displacements : np.ndarray, shape=[n_frames, n_pairs, 3], dtype=float32
         The displacememt vector, in each frame, between each pair of atoms.
    """
    xyz = ensure_type(
        traj.xyz,
        dtype=np.float32,
        ndim=3,
        name="traj.xyz",
        shape=(None, None, 3),
        warn_on_cast=False,
    )
    pairs = ensure_type(
        np.asarray(atom_pairs),
        dtype=np.int32,
        ndim=2,
        name="atom_pairs",
        shape=(None, 2),
        warn_on_cast=False,
    )
    if not np.all(np.logical_and(pairs < traj.n_atoms, pairs >= 0)):
        raise ValueError("atom_pairs must be between 0 and %d" % traj.n_atoms)
    if len(pairs) == 0:  # If pairs is an empty slice of an array
        return np.zeros((len(xyz), 0, 3), dtype=np.float32)

    if periodic and traj._have_unitcell:
        box = ensure_type(
            traj.unitcell_vectors,
            dtype=np.float32,
            ndim=3,
            name="unitcell_vectors",
            shape=(len(xyz), 3, 3),
            warn_on_cast=False,
        )
        orthogonal = np.allclose(traj.unitcell_angles, 90)
        return _displacement_mic(xyz, pairs, box.transpose(0, 2, 1), orthogonal)

    # either there are no unitcell vectors or they dont want to use them

    return _displacement(xyz, pairs)

#############################################################################################################
# https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/dihedral.py
#############################################################################################################


def _dihedral(traj, indices, periodic, out=None):
    """Compute the dihedral angles of traj for the atom indices in indices.

    Parameters
    ----------
    xyz : np.ndarray, shape=(num_frames, num_atoms, 3), dtype=float
        The XYZ coordinates of a trajectory
    indices : np.ndarray, shape=(num_dihedrals, 4), dtype=int
        Atom indices to compute dihedrals.
    periodic : bool, default=True
        If `periodic` is True and the trajectory contains unitcell
        information, we will treat dihedrals that cross periodic images
        using the minimum image convention.

    Returns
    -------
    dih : np.ndarray, shape=(num_dihedrals), dtype=float
        dih[i,j] gives the dihedral angle at traj[i] correponding to indices[j].

    """
    ix10 = indices[:, [0, 1]]
    ix21 = indices[:, [1, 2]]
    ix32 = indices[:, [2, 3]]

    b1 = compute_displacements(traj, ix10, periodic=periodic, opt=False)
    b2 = compute_displacements(traj, ix21, periodic=periodic, opt=False)
    b3 = compute_displacements(traj, ix32, periodic=periodic, opt=False)

    c1 = np.cross(b2, b3)
    c2 = np.cross(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 *= (b2 * b2).sum(-1) ** 0.5
    p2 = (c1 * c2).sum(-1)

    return np.arctan2(p1, p2, out)


def compute_dihedrals(traj, indices, periodic=True, opt=False):
    """Compute the dihedral angles between the supplied quartets of atoms in each frame in a trajectory.

    Parameters
    ----------
    traj : Trajectory
        An mtraj trajectory.
    indices : np.ndarray, shape=(n_dihedrals, 4), dtype=int
        Each row gives the indices of four atoms which together make a
        dihedral angle. The angle is between the planes spanned by the first
        three atoms and the last three atoms, a torsion around the bond
        between the middle two atoms.
    periodic : bool, default=True
        If `periodic` is True and the trajectory contains unitcell
        information, we will treat dihedrals that cross periodic images
        using the minimum image convention.
    opt : bool, default=True
        Use an optimized native library to calculate angles.

    Returns
    -------
    dihedrals : np.ndarray, shape=(n_frames, n_dihedrals), dtype=float
        The output array gives, in each frame from the trajectory, each of the
        `n_dihedrals` torsion angles. The angles are measured in **radians**.

    """
    xyz = ensure_type(
        traj.xyz,
        dtype=np.float32,
        ndim=3,
        name="traj.xyz",
        shape=(None, None, 3),
        warn_on_cast=False,
    )
    quartets = ensure_type(
        indices,
        dtype=np.int32,
        ndim=2,
        name="indices",
        shape=(None, 4),
        warn_on_cast=False,
    )
    if not np.all(np.logical_and(quartets < traj.n_atoms, quartets >= 0)):
        raise ValueError("indices must be between 0 and %d" % traj.n_atoms)

    if len(quartets) == 0:
        return np.zeros((len(xyz), 0), dtype=np.float32)

    out = np.zeros((xyz.shape[0], quartets.shape[0]), dtype=np.float32)
    if periodic and traj._have_unitcell:
        box = ensure_type(
            traj.unitcell_vectors,
            dtype=np.float32,
            ndim=3,
            name="unitcell_vectors",
            shape=(len(xyz), 3, 3),
        )
        _dihedral(traj, quartets, periodic, out)
        return out

    _dihedral(traj, quartets, periodic, out)
    return out

#############################################################################################################
# https://github.com/noegroup/bgmol/blob/main/bgmol/systems/ala2.py
#############################################################################################################

def compute_phi_psi(traj):
    """Compute backbone dihedrals.

    Parameters
    ----------
    traj : mdtraj.Trajectory
    """
    phi_atoms = [4, 6, 8, 14]
    phi = compute_dihedrals(traj, indices=[phi_atoms])[:, 0]
    psi_atoms = [6, 8, 14, 16]
    psi = compute_dihedrals(traj, indices=[psi_atoms])[:, 0]
    return phi, psi


DEFAULT_RIGID_BLOCK = np.array([6, 8, 9, 10, 14])


DEFAULT_Z_MATRIX = np.array([
    [0, 1, 4, 6],
    [1, 4, 6, 8],
    [2, 1, 4, 0],
    [3, 1, 4, 0],
    [4, 6, 8, 14], # phi
    [5, 4, 6, 8],
    [7, 6, 8, 4],
    [11, 10, 8, 6],
    [12, 10, 8, 11],
    [13, 10, 8, 11],
    [15, 14, 8, 16],
    [16, 14, 8, 6], # psi # mdtraj uses [16, 14, 8, 6]
    [17, 16, 14, 15],
    [18, 16, 14, 8],
    [19, 18, 16, 14],
    [20, 18, 16, 19],
    [21, 18, 16, 19]
])


DEFAULT_GLOBAL_Z_MATRIX = np.row_stack([
    DEFAULT_Z_MATRIX,
    np.array([
        [9, 8, 6, 14],
        [10, 8, 14, 6],
        [6, 8, 14, -1],
        [8, 14, -1, -1],
        [14, -1, -1, -1]
    ])
])


class AlanineDipeptideTSF(OpenMMSystem):
    """Alanine Dipeptide from the Temperature-Steering Flows paper,
    Dibak, Klein, Noé (2020): https://arxiv.org/abs/2012.00429

    Notes
    -----
    Requires an internet connection to download the initial structure.
    """

    url = "http://ftp.mi.fu-berlin.de/pub/cmb-data/bgmol/systems/ala2/"

    def __init__(self,  root=tempfile.gettempdir(), download=True):
        super(AlanineDipeptideTSF, self).__init__()

        # download pdb file
        filename = "alanine-dipeptide-nowater.pdb"
        full_filename = os.path.join(root, filename)
        if download:
            download_url(self.url + filename, root, filename, md5="728635667ed4937cf4a0e5b7c801d9ea")
        assert os.path.isfile(full_filename)

        pdb = openmm.app.PDBFile(full_filename)
        ff = openmm.app.ForceField("amber99sbildn.xml", "amber96_obc.xml")
        self._system = ff.createSystem(
            pdb.getTopology(),
            removeCMMotion=True,
            nonbondedMethod=openmm.app.NoCutoff,
            constraints=openmm.app.HBonds,
            rigidWater=True
        )
        self._positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        self._topology = pdb.getTopology()
        self.z_matrix = DEFAULT_Z_MATRIX.copy()
        self.rigid_block = DEFAULT_RIGID_BLOCK.copy()

    @staticmethod
    def compute_phi_psi(traj):
        return compute_phi_psi(traj)


class AlanineDipeptideImplicit(OpenMMToolsTestSystem):
    def __init__(self, constraints=openmm.app.HBonds, hydrogenMass=None):
        super().__init__("AlanineDipeptideImplicit", constraints=constraints, hydrogenMass=hydrogenMass)
        self.z_matrix = DEFAULT_Z_MATRIX.copy()
        self.rigid_block = DEFAULT_RIGID_BLOCK.copy()

    @staticmethod
    def compute_phi_psi(traj):
        return compute_phi_psi(traj)

#############################################################################################################
# https://github.com/noegroup/bgmol/blob/main/bgmol/datasets/ala2.py
#############################################################################################################


class Ala2Implicit300(DataSet):
    """AlanineDipeptideImplicit at 300 K.
    1 ms Langevin dynamics with 1/ps friction coefficient and 2fs time step,
    output spaced in 10 ps intervals.
    """
    url = "http://ftp.mi.fu-berlin.de/pub/cmb-data/bgmol/datasets/ala2/Ala2Implicit300.tgz"
    md5 = "ce9d6f6aa214f3eb773d52255aeaeacb"
    num_frames = 99999
    size = 49692000 # in bytes
    selection = "all"
    openmm_version = "7.4.1"
    date = "2020/09/18"
    author = "Andreas Krämer"

    def __init__(self, root=os.getcwd(), download: bool = False, read: bool = False):
        super(Ala2Implicit300, self).__init__(root=root, download=download, read=read)
        self._system = system_by_name("AlanineDipeptideImplicit")
        self._temperature = 300

    @property
    def trajectory_file(self):
        return os.path.join(self.root, "Ala2Implicit300/traj0.h5")

    def read(self, n_frames=None, stride=None, atom_indices=None):
        self.trajectory = load_hdf5(self.trajectory_file)
        f = HDF5TrajectoryFile(self.trajectory_file)
        frames = f.read(n_frames=n_frames, stride=stride, atom_indices=atom_indices)
        self._energies = frames.potentialEnergy
        self._forces = frames.forces
        f.close()

#############################################################################################################
# Plotting
#############################################################################################################

def plot_phi_psi(trajectory, system):
    """Ramachandran Plot for the Backbone Angles.
    notebooks/alanine_dipeptide_basics.ipynb
    plot the Boltzmann distribution / probability P(x) over the dihedral angles, 
    estimated by the occurance in the dataset
    log scale on the color axis
    """
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.cpu().detach().numpy().reshape(-1, 22, 3), 
            topology=system.mdtraj_topology
        )
    phi, psi = system.compute_phi_psi(trajectory)
    
    fig, ax = plt.subplots(figsize=(3,3))
    ax.hist2d(phi, psi, 50, norm=LogNorm())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    _ = ax.set_ylabel("$\psi$")
    figname = "alanine_dipeptide/plots/bg_ramachandran.png"
    plt.savefig(figname)
    print(f"{figname} saved")


def scatter_plot_energy(dataset):
    """Energy (color) for the Backbone Angles.
    """
    phi, psi = dataset.system.compute_phi_psi(dataset.trajectory)
    fig, ax = plt.subplots(figsize=(3,3))
    z = dataset.energies
    x = phi
    y = psi
    ax.scatter(x, y, c=z, cmap="viridis")
    # plt.colorbar(ax, label="Energy (kJ/mol)")
    figname = "alanine_dipeptide/plots/bg_energy_scatter.png"
    plt.savefig(figname)
    print(f"{figname} saved")

def binning_plot_energy(dataset):
    phi, psi = dataset.system.compute_phi_psi(dataset.trajectory)
    z = dataset.energies

    # Define the number of bins along each dimension (adjust num_bins as needed)
    num_bins = 50

    # Bin the data: compute the average energy in each (phi, psi) bin
    stat, xedges, yedges, binnumber = binned_statistic_2d(
        phi, psi, values=z, statistic='mean', bins=num_bins
    )

    # Compute the centers of the bins
    x_centers = (xedges[:-1] + xedges[1:]) / 2.0
    y_centers = (yedges[:-1] + yedges[1:]) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)

    # Create a contour plot of the average energy
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, stat.T, levels=50, cmap="viridis")
    plt.xlabel("$\\phi$")
    plt.ylabel("$\\psi$")
    plt.title("Contour Plot of Average Energy over Dihedral Angles")
    plt.colorbar(contour, label="Average Energy (kJ/mol)")
    figname = "alanine_dipeptide/plots/bg_energy_contour.png"
    plt.savefig(figname)
    print(f"{figname} saved")

if __name__ == "__main__":

    # is_data_here = os.path.isfile("Ala2TSF300.tgz")
    is_data_here = os.path.isdir("Ala2TSF300")
    dataset = Ala2Implicit300(download=not is_data_here, read=True)

    # The dataset contains forces, energies and coordinates it also holds a reference to the system that defines the potential energy function.

    openmmsystem = dataset.system
    system = dataset.system

    # The system is an OpenMMSystem object, it provides access to the openmm.system instance, the topology, and a set of initial coordinates. For example, we can run an OpenMM simulation as follows

    integrator = LangevinIntegrator(dataset.temperature, 1, 0.001)
    simulation = openmm.app.Simulation(openmmsystem.topology, openmmsystem.system, integrator)
    simulation.context.setPositions(openmmsystem.positions)
    simulation.step(10)

    # The dataset contains coordinates (xyz), forces, and energies.

    # print("energies", dataset.energies.shape)
    # print("forces", dataset.forces.shape)
    # print("positions", dataset.trajectory.xyz.shape)
    # print("first frame positions:", dataset.trajectory.xyz[0].shape)
    # print("trajectory shape:", dataset.trajectory.xyz.shape)
    
    # we can also get the internal coordinates 
    phi, psi = dataset.system.compute_phi_psi(dataset.trajectory)
    # print("phi", phi)
    # print("psi", psi)
    
    ###################################################################################
    
    # Define the Internal Coordinate Transform
    # Rather than generating all-Cartesian coordinates, we use a mixed internal coordinate transform.
    # The five central alanine atoms will serve as a Cartesian "anchor", 
    # from which all other atoms are placed with respect to internal coordinates (IC) 
    # defined through a z-matrix. 
    # We have deposited a valid `z_matrix` and the corresponding `rigid_block` in the `dataset.system` 

    # throw away 6 degrees of freedom (rotation and translation)
    dim_cartesian = len(system.rigid_block) * 3 - 6
    dim_bonds = len(system.z_matrix)
    dim_angles = dim_bonds
    dim_torsions = dim_bonds
    
    # a context tensor to send data to the right device and dtype via '.to(ctx)'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    ctx = torch.zeros([], device=device, dtype=dtype)   
    
    training_data = torch.tensor(dataset.trajectory.xyz).to(ctx)
    
    coordinate_transform = bg.MixedCoordinateTransformation(
        data=training_data, 
        z_matrix=system.z_matrix,
        fixed_atoms=system.rigid_block,
        keepdims=dim_cartesian, 
        normalize_angles=True,
    ).to(ctx)
    
    # For demonstration, we transform the first 3 samples from the training data set into internal coordinates:
    bonds, angles, torsions, cartesian, dlogp = coordinate_transform.forward(training_data[:3])
    # print("bonds.shape", bonds.shape)
    # print("angles.shape", angles.shape)
    # print("torsions.shape", torsions.shape)
    # print("cartesian.shape", cartesian.shape)
    # print("dlogp.shape", dlogp.shape)
    
    ###################################################################################
    
    plot_phi_psi(dataset.trajectory, dataset.system)
    
    scatter_plot_energy(dataset)
    
    binning_plot_energy(dataset)
    