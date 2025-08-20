""" Nanoparticle plots for the book. """

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_corrected_minimum():
    # Define the potential energy surface (PES) as a function
    def objective_function(x):
        left_well = (x + 1.8)**2 - 1
        right_well = (x - 2)**2
        return left_well * right_well

    # Define the range of x values
    x = np.linspace(-4, 4, 400)
    y = objective_function(x)

    # Simulate a local optimization process
    # Suppose the optimization starts near the right-hand local minimum
    x_start = 2.5
    steps = np.linspace(x_start, 1.5, 15)  # Optimization steps going to the local minimum
    y_steps = objective_function(steps)

    # Plotting the potential energy surface
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "k:", label='Potential Energy Surface')

    # Highlight the global and local minima
    global_min_x = -2.0
    global_min_y = objective_function(global_min_x)
    local_min_x = 2.0
    local_min_y = objective_function(local_min_x)

    plt.scatter(global_min_x, global_min_y, color='blue', zorder=5, label='Global Minimum')
    plt.scatter(local_min_x, local_min_y, color='red', zorder=5, label='Local Minimum')

    # Plot the steps of the optimization getting stuck at the local minimum
    plt.plot(steps, y_steps, 'ko', label='Optimization Path', markersize=5, alpha=0.5)

    # Add labels and legend
    plt.xlabel('Geometry (Arbitrary Units)')
    plt.ylabel('Energy (Arbitrary Units)')
    plt.title('Geometry Optimization: Stuck in a Local Minimum')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.savefig('local_minimum.png')

def main():
    print("Nanoparticle plots for the book.")

    # Call the function to generate and display the plot
    plot_corrected_minimum()

if __name__ == "__main__":
    main()