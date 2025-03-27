import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks


# Define the double well potential and its force.
def double_well_potential(x, a=1.0, b=4.0):
    """Double-well potential: U(x) = a*x^4 - b*x^2"""
    return a * x**4 - b * x**2


def double_well_force(x, a=1.0, b=4.0):
    """Force from double-well potential: F(x) = -dU/dx"""
    return -(4 * a * x**3 - 2 * b * x)


# Define the bias potential from deposited Gaussian hills.
def bias_potential(x, hill_positions, hill_heights, hill_width):
    """
    Compute the total bias potential at position x due to deposited hills.
    Each hill is a Gaussian of the form:
       w * exp[-(x - x_i)**2 / (2 * sigma**2)]
    """
    bias = 0.0
    for pos, height in zip(hill_positions, hill_heights):
        bias += height * np.exp(-((x - pos) ** 2) / (2 * hill_width**2))
    return bias


def bias_force(x, hill_positions, hill_heights, hill_width):
    """
    Compute the force due to the bias potential.
    F_bias(x) = -dV_bias/dx.
    """
    force = 0.0
    for pos, height in zip(hill_positions, hill_heights):
        gaussian = np.exp(-((x - pos) ** 2) / (2 * hill_width**2))
        force += height * (x - pos) / (hill_width**2) * gaussian
    return -force


# To speed up escaping minima:
# Reduce deposit_interval (e.g., from 100 to 50 or 20) so that hills are added more frequently.
# increasing initial_hill_height to a larger value, such as 0.5 or 1.0, so that the bias builds up more quickly
# Increase temperature in the Langevin dynamics from 0.1

# Simulation parameters
dt = 0.005  # time step
n_steps = int(2e4)  # number of simulation steps
mass = 1.0  # mass of the particle
gamma = 0.5  # friction coefficient for Langevin dynamics
temperature = 1.0  # temperature for stochastic noise. default: 0.1

# Well-tempered metadynamics parameters
initial_hill_height = 1.0  # initial hill height (w0). default: 0.2
hill_width = 1.0  # width (sigma) of each Gaussian hill. default: 0.1
deposit_interval = 10  # deposit a hill every deposit_interval steps. default: 100
delta_T = 0.5  # effective bias "temperature" that controls tempering. default: 0.5

# Initialize arrays to store trajectory and times
trajectory = np.zeros(n_steps)
times = np.zeros(n_steps)

# Initial conditions: start at one of the minima
x = -1.0
v = 0.0

# Lists to store the hill positions and heights (bias potential)
hill_positions = []
hill_heights = []

# For reproducibility, you may set a random seed.
np.random.seed(42)

# Main simulation loop using Langevin dynamics
for step in tqdm(range(n_steps)):
    t = step * dt
    times[step] = t

    # Total force: from the double well and the bias
    F = double_well_force(x) + bias_force(x, hill_positions, hill_heights, hill_width)

    # Include Langevin thermostat: damping and stochastic noise
    noise = np.sqrt(2 * gamma * temperature / dt) * np.random.randn()

    # Update velocity and position (simple Euler integration)
    v = v + dt * (F - gamma * v + noise) / mass
    x = x + dt * v

    trajectory[step] = x

    # Deposit a hill every deposit_interval steps using well-tempered scaling.
    if step % deposit_interval == 0:
        # Evaluate current bias potential at x
        current_bias = bias_potential(x, hill_positions, hill_heights, hill_width)
        # Scale hill height using well-tempered protocol:
        effective_hill_height = initial_hill_height * np.exp(-current_bias / delta_T)
        hill_positions.append(x)
        hill_heights.append(effective_hill_height)

# Plot the trajectory over time.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(times, trajectory, lw=0.8)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Trajectory")

# Plot the free energy (biased potential plus original potential) landscape
x_vals = np.linspace(-2, 2, 400)
U_vals = double_well_potential(x_vals)
bias_vals = np.array(
    [
        bias_potential(x_val, hill_positions, hill_heights, hill_width)
        for x_val in x_vals
    ]
)
total_potential = U_vals + bias_vals

# Identify transition states as peaks between basins in the original potential
# For the double well potential, we know there's a transition state at x=0
# But we'll use a more general approach to find peaks in the potential
peaks, _ = find_peaks(U_vals)
transition_states = [x_vals[peak] for peak in peaks]
print(f"Identified transition states in original potential: {transition_states}")

# We can also look for transition states in the biased potential landscape
# This might reveal additional transition pathways discovered during metadynamics
biased_peaks, _ = find_peaks(total_potential)
biased_transition_states = [x_vals[peak] for peak in biased_peaks]
print(f"Identified transition states in biased potential: {biased_transition_states}")

plt.subplot(1, 2, 2)
plt.plot(x_vals, U_vals, "k--", label="Original Potential")
plt.plot(x_vals, bias_vals, "r:", label="Bias Potential")
plt.plot(x_vals, total_potential, "b-", label="Total Potential")
# Mark transition states on the plot
for ts in transition_states:
    plt.axvline(x=ts, color="g", linestyle="--", alpha=0.7)
    plt.plot(ts, double_well_potential(ts), "go", markersize=8)
plt.xlabel("x")
plt.ylabel("Potential")
plt.legend()
plt.title("Potential Landscape with Transition States")

plt.tight_layout()
plt.savefig("metadynamics_well_tempered.png")
plt.show()
