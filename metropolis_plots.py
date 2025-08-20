import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def river_depth(x, y):
    """
    Models the depth of the Mississippi River.
    Depth is a Gaussian function of y (distance from river center).
    """
    # River's depth peaks at y=0 and falls off as a Gaussian
    # Center the depth profile along y=0, assume river width of 10 units
    depth = np.exp(-y**2 / (2 * 2.0**2))  # Gaussian depth profile
    return depth * (x >= 0)  # Only valid for x >= 0, river exists in this domain

def generate_contour_plot(x_range, y_range, resolution, metropolis_points=1000):
    """
    Generates a contour plot showing the river's depth, illustrating
    the difference between conventional quadrature sampling and 
    Metropolis-like sampling.
    
    Args:
        x_range (tuple): Range of x values (start, end)
        y_range (tuple): Range of y values (start, end)
        resolution (int): Resolution of the grid (for quadrature sampling)
        metropolis_points (int): Number of points to sample using Metropolis
    """
    # Create a grid for quadrature sampling (conventional approach)
    x_grid = np.linspace(x_range[0], x_range[1], resolution)
    y_grid = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = river_depth(X, Y)

    # Metropolis-like sampling (samples near the river center)
    x_samples = np.random.uniform(x_range[0], x_range[1], metropolis_points)
    y_samples = np.random.normal(loc=0, scale=2, size=metropolis_points)  # Sample near y=0

    # Plotting the results
    plt.figure(figsize=(10, 6))

    # Contour plot for quadrature sampling
    plt.contourf(X, Y, Z, levels=20, cmap=cm.Blues, alpha=0.7)
    plt.colorbar(label='Depth of the River')

    # Metropolis-like sampling points
    plt.scatter(x_samples, y_samples, color='red', s=10, label='Metropolis Sampling')

    # Conventional grid points
    x_grid_pts = np.linspace(x_range[0], x_range[1], 20)
    y_grid_pts = np.linspace(y_range[0], y_range[1], 20)
    X_grid_pts, Y_grid_pts = np.meshgrid(x_grid_pts, y_grid_pts)
    plt.scatter(X_grid_pts, Y_grid_pts, color='black', s=10, label='Conventional Grid Sampling', zorder=-10)  # , alpha=0.5)

    plt.title("Depth of Mississippi River (Contour) with Metropolis vs Grid Sampling")
    plt.xlabel('x (Length along the river)')
    plt.ylabel('y (Distance from river center)')
    plt.legend()

    # Save plot to a PNG file
    plt.savefig('mississippi_river_depth.png')

def morse_potential(x, D_e, alpha, x_e):
    """
    Computes the Morse potential for a given displacement x.
    
    Args:
        x (float or ndarray): Displacement from the equilibrium bond length.
        D_e (float): Dissociation energy.
        alpha (float): Width of the potential well.
        x_e (float): Equilibrium bond length.
        
    Returns:
        ndarray: Potential energy at displacement x.
    """
    return D_e * (1 - np.exp(-alpha * (x - x_e)))**2

def plot_morse_potential(D_e, alpha, x_e, x_range=(-2, 8)):
    """
    Plots the Morse potential and visually defines its key parameters.
    
    Args:
        D_e (float): Dissociation energy.
        alpha (float): Width of the potential well.
        x_e (float): Equilibrium bond length.
        x_range (tuple): Range of x values to plot over.
    """
    x_vals = np.linspace(x_range[0], x_range[1], 500)
    U_vals = morse_potential(x_vals, D_e, alpha, x_e)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the Morse potential
    plt.plot(x_vals, U_vals, label=r'$U(x) = D_e \left[ 1 - e^{-\alpha (x - x_e)} \right]^2$', color='blue')
    
    # Mark the equilibrium bond length (x_e)
    plt.axvline(x=x_e, color='green', linestyle='--', label=r'$x_e$ (Equilibrium bond length)')
    
    # Mark the dissociation energy (D_e)
    plt.hlines(y=D_e, xmin=x_range[0], xmax=x_e, color='red', linestyle='--', label=r'$D_e$ (Dissociation energy)')
    
    # Mark the potential well width (alpha)
    plt.annotate(r'$\alpha$ (Width of potential well)', 
                 xy=(x_e + 1.0/alpha, morse_potential(x_e + 1.0/alpha, D_e, alpha, x_e)),
                 xytext=(x_e + 2, D_e / 2),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)
    
    # Labels and titles
    plt.title('Morse Oscillator Potential')
    plt.xlabel('Displacement (x)')
    plt.ylabel('Potential Energy U(x)')
    
    # Add a legend
    plt.legend()
    
    # Show grid and display the plot
    plt.grid(True)
    plt.ylim(0, D_e * 1.2)
    plt.xlim(x_range)
    plt.savefig('morse_potential.png')

def main():
    # Define the river's depth profile
    x_range = (0, 100)  # x-axis: 0 to 100 units along the river
    y_range = (-10, 10)  # y-axis: -10 to 10 units, width of the river
    resolution = 100  # Grid resolution for quadrature sampling
    generate_contour_plot(x_range, y_range, resolution)

    # Define the Morse potential parameters
    D_e = 5  # Dissociation energy
    alpha = 1  # Width of the potential well
    x_e = 3   # Equilibrium bond length
    plot_morse_potential(D_e, alpha, x_e)

if __name__ == '__main__':
    main()