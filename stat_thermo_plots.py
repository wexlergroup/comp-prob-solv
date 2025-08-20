"""
Make plots for statistical thermodynamics
"""

import matplotlib.pyplot as plt
import numpy as np

# Function to create and save a PNG file showing the difference between microstates and macrostates
def plot_microstate_macrostate(filename="microstate_macrostate.png"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Microstate plot
    axs[0].set_title("Microstate: Particle Positions & Velocities")
    axs[0].set_xlim(0, 10)
    axs[0].set_ylim(0, 10)
    axs[0].set_xlabel("X Position")
    axs[0].set_ylabel("Y Position")
    for _ in range(100):  # Simulate 100 particles
        x, y = np.random.uniform(0, 10, 2)
        vx, vy = np.random.uniform(-1, 1, 2)  # Random velocity vectors
        axs[0].quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, color='b', alpha=0.7)
    axs[0].grid(True)
    
    # Macrostate plot
    axs[1].set_title("Macrostate: Temperature, Pressure, Volume")
    axs[1].set_xlim(0, 10)
    axs[1].set_ylim(0, 10)
    axs[1].text(5, 7, "T = 300 K", fontsize=14, ha='center')
    axs[1].text(5, 5, "P = 1 bar", fontsize=14, ha='center')
    axs[1].text(5, 3, "V = 22.4 L", fontsize=14, ha='center')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].grid(False)

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # Call the function to create and save the image
    plot_microstate_macrostate()
