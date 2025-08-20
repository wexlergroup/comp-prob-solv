"""
Make plots for the lecture on ensembles and ergodicity.
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Hello, world!")

def microcanonical_ensemble():
    """
    Make a plot of the microcanonical ensemble.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define the properties of the microstates
    mlandms = [
        ("-3", "-1/2"),
        ("-3", "+1/2"),
        ("-2", "-1/2"),
        ("-2", "+1/2"),
        ("-1", "-1/2"),
        ("-1", "+1/2"),
        ("0", "-1/2"),
        ("0", "+1/2"),
        ("+1", "-1/2"),
        ("+1", "+1/2"),
        ("+2", "-1/2"),
        ("+2", "+1/2"),
        ("+3", "-1/2"),
        ("+3", "+1/2")
    ]
    num_orbitals = len(mlandms)
    orbital_positions = np.linspace(1, num_orbitals, num_orbitals)
    energies = [1] * num_orbitals

    # Plot the microstates
    for i, (x, y) in enumerate(zip(orbital_positions, energies)):
        ax.plot(x, y, '_', markersize=20, color='black')
    
    # Add one electron in an f orbital
    ax.text(orbital_positions[6], 1, "e", fontsize=16, ha='center')

    # Set the x-axis labels
    ax.set_xticks(orbital_positions)
    ax.set_xticklabels([f"{ml}\n{ms}" for ml, ms in mlandms])
    ax.set_yticks([])

    # Labels and annotations
    ax.set_xlabel("$m_l$\n$m_s$")
    ax.set_ylabel("Energy")
    ax.set_title("One Electron in an $f$ Orbital")
    
    # Display the plot
    plt.tight_layout()
    plt.savefig('microcanonical_ensemble.png')

import matplotlib.pyplot as plt
import numpy as np

def plot_adsorption_sites(grid_size=4, sorbent_positions=None):
    """
    Plots a grid of adsorption sites with sorbent particles.
    
    Parameters:
    - grid_size (int): The size of the grid (grid_size x grid_size).
    - sorbent_positions (list of tuples): List of (x, y) positions for sorbent particles.
    """
    if sorbent_positions is None:
        sorbent_positions = [(0.5, 0.5), (2.5, 0.5), (1.5, 1.5), (3.5, 1.5), (0.5, 2.5)]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw grid lines and label adsorption sites with 'A'
    for x in range(grid_size):
        for y in range(grid_size):
            # Draw grid lines
            ax.plot([x, x], [0, grid_size], color='gray', linewidth=0.5)
            ax.plot([0, grid_size], [y, y], color='gray', linewidth=0.5)
            # Label adsorption sites with 'A'
            ax.text(x + 0.5, y + 0.5, 'A', ha='center', va='center', fontsize=12)
    
    # Draw the outer boundary of the grid
    ax.plot([0, grid_size, grid_size, 0, 0], [0, 0, grid_size, grid_size, 0], color='black')
    
    # Plot sorbent particles
    for x, y in sorbent_positions:
        ax.plot(x, y, 'o', markersize=15, color='blue')
    
    # Labels for N_a and N_s
    ax.text(grid_size / 2, -0.5, f'$N_a = {grid_size} \\times {grid_size} = {grid_size * grid_size}$ adsorption sites', 
            ha='center', fontsize=12)
    ax.text(grid_size / 2, grid_size + 0.5, f'$N_s = {len(sorbent_positions)}$ sorbent particles', 
            ha='center', fontsize=12, color='blue')
    
    # Set limits and remove axis labels
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-1, grid_size + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Display the plot
    plt.title("Adsorption Sites and Sorbent Particles")
    plt.tight_layout()
    plt.savefig('adsorption_sites.png')

import matplotlib.pyplot as plt
import numpy as np

def plot_coverage_vs_pressure(beta_mu_0, P_0, K, pressure_range=(0, 10), num_points=100):
    """
    Plots the coverage (θ) as a function of pressure (P).
    
    Parameters:
    - beta_mu_0 (float): Value of β * μ^0.
    - P_0 (float): Reference pressure.
    - K (float): Equilibrium constant K.
    - pressure_range (tuple): Range of pressures to plot, as (min, max).
    - num_points (int): Number of pressure points to plot.
    """
    # Define pressure values
    P_values = np.linspace(pressure_range[0], pressure_range[1], num_points)
    
    # Calculate coverage (θ) for each pressure
    theta_values = (K * P_values) / (1 + K * P_values)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(P_values, theta_values, label=r'$\theta = \frac{KP}{1 + KP}$', color='blue')
    plt.text(9, 0.1, r'$K = 1.0$', fontsize=14, color='blue', ha='center', va='center')
    plt.xlabel('Pressure (P)', fontsize=14)
    plt.ylabel('Coverage (θ)', fontsize=14)
    plt.ylim(0, 1)
    plt.title('Coverage vs. Pressure', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('coverage_vs_pressure.png')


if __name__ == "__main__":
    main()
    microcanonical_ensemble()
    plot_adsorption_sites()
    plot_coverage_vs_pressure(beta_mu_0=1.0, P_0=1.0, K=1.0, pressure_range=(0, 10), num_points=200)
