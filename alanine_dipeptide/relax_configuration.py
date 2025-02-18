from openmm.app import PDBFile, ForceField, Simulation, NoCutoff
from openmm import Platform, VerletIntegrator
from openmm.unit import picoseconds, nanometer, kilojoule_per_mole


def relax_configuration(
    pdb_input,
    forcefield_file,
    output_pdb,
    tolerance=1.0,
    max_iterations=500,
    platform_name="Reference",
):
    """
    Load a configuration from a PDB file, relax (minimize) the structure using the
    specified force field, and save the relaxed structure to a new PDB file.

    Parameters:
      pdb_input         : Path to the input PDB file.
      forcefield_file   : Path to the OpenMM force field XML file (e.g., amber99sb.xml).
      output_pdb        : Path to write the relaxed PDB file.
      tolerance         : Minimization tolerance (default 1.0 kJ/mol/nm).
      max_iterations    : Maximum number of minimization iterations (default 500).
      platform_name     : OpenMM platform to use (default "Reference").
    """
    # Load the original configuration
    pdb = PDBFile(pdb_input)

    # Create the force field and system
    forcefield = ForceField(forcefield_file)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)

    # Set up the integrator and simulation
    integrator = VerletIntegrator(0.001 * picoseconds)
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # Set initial positions
    simulation.context.setPositions(pdb.positions)

    # Get initial energy
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(f"Initial energy: {energy:.2f} kJ/mol")

    # Custom energy minimizer that reports progress
    def minimize_with_progress():
        # Get initial energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        pos = state.getPositions()
        energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        print(f"Initial energy: {energy:.2f} kJ/mol")

        # Initialize
        last_energy = energy
        step = 0

        while step < max_iterations:
            simulation.minimizeEnergy(maxIterations=1, tolerance=tolerance)
            state = simulation.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

            # Print progress every 10 steps
            if step % 10 == 0:
                print(
                    f"Step {step}: Energy = {energy:.2f} kJ/mol, ΔE = {energy-last_energy:.2f} kJ/mol"
                )

            # change energy to kcal/mol
            # energy = energy * 0.239006
            # last_energy = last_energy * 0.239006

            # Check for convergence
            if abs(energy - last_energy) < tolerance:
                print(f"Converged at step {step}")
                break

            last_energy = energy
            step += 1

        return state.getPositions()

    # Run energy minimization with progress reporting
    relaxed_positions = minimize_with_progress()

    # Get final energy
    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(f"Final energy: {final_energy:.2f} kJ/mol")

    # Save the relaxed configuration to a new PDB file
    with open(output_pdb, "w") as f:
        PDBFile.writeFile(pdb.topology, relaxed_positions, f)


if __name__ == "__main__":
    # Specify the input files and the output relaxed pdb
    input_pdb = "alanine_dipeptide/data/alanine_dipeptide_nowater.pdb"
    ff_file = "alanine_dipeptide/data/amber99sb.xml"
    output_pdb = "alanine_dipeptide/data/relaxed_alanine_dipeptide.pdb"

    relax_configuration(input_pdb, ff_file, output_pdb)
    print("Relaxation complete. Relaxed PDB written to:", output_pdb)
