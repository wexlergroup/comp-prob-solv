import matplotlib.pyplot as plt
import numpy as np

def plot_periodic_boundary_conditions_colored():
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set the size of the simulation box
    box_size = 10
    ax.set_xlim(-box_size, 2 * box_size)
    ax.set_ylim(-box_size, 2 * box_size)

    # Define particle positions inside the central box
    particles = np.array([
        [2, 8], [8, 2], [5, 5], [1, 1], [9, 9]
    ])

    # Define colors for each particle
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']

    # Plot the central cell particles and periodic images in surrounding cells
    shifts = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]  # Shift coordinates for 8 neighboring cells
    
    for (dx, dy) in shifts:
        shifted_particles = particles + np.array([dx * box_size, dy * box_size])
        for idx, (x, y) in enumerate(shifted_particles):
            if dx == 0 and dy == 0:
                # Original particles (central cell)
                ax.scatter(x, y, color=colors[idx], s=100, edgecolor='black', label=f'Particle {idx + 1}')
            else:
                # Periodic images
                ax.scatter(x, y, color=colors[idx], s=100, edgecolor='black', alpha=0.7)

    # Labels and title
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('2D Periodic Boundary Conditions')

    # Draw the box boundaries for the central cell and surrounding cells
    for dx, dy in shifts:
        rect = plt.Rectangle((dx * box_size, dy * box_size), box_size, box_size, 
                             linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # Highlight the central box
    central_rect = plt.Rectangle((0, 0), box_size, box_size, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(central_rect)

    # Make the plot square
    ax.set_aspect('equal')

    # Save to PNG
    plt.savefig('pbc_2d_example_colored.png')

def plot_all_minimum_image_distances():
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set the size of the simulation box
    box_size = 10
    ax.set_xlim(-box_size, 2 * box_size)
    ax.set_ylim(-box_size, 2 * box_size)

    # Define particle positions inside the central box
    particles = np.array([
        [2, 8],  # Blue particle
        [8, 2],  # Green particle
        [5, 5],  # Purple particle
        [1, 1],  # Orange particle
        [9, 9]   # Cyan particle
    ])

    # Define colors for each particle
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']

    # Plot the central cell particles and periodic images in surrounding cells
    shifts = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]  # Shift coordinates for 8 neighboring cells
    
    all_shifted_positions = []
    for (dx, dy) in shifts:
        shifted_particles = particles + np.array([dx * box_size, dy * box_size])
        all_shifted_positions.append(shifted_particles)
        for idx, (x, y) in enumerate(shifted_particles):
            if dx == 0 and dy == 0:
                # Original particles (central cell)
                ax.scatter(x, y, color=colors[idx], s=100, edgecolor='black', label=f'Particle {idx + 1}')
            else:
                # Periodic images
                ax.scatter(x, y, color=colors[idx], s=100, edgecolor='black', alpha=0.7)

    # Draw the box boundaries for the central cell and surrounding cells
    for dx, dy in shifts:
        rect = plt.Rectangle((dx * box_size, dy * box_size), box_size, box_size, 
                             linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # Highlight the central box
    central_rect = plt.Rectangle((0, 0), box_size, box_size, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(central_rect)

    # Minimum image convention: calculate the shortest distance between the blue and cyan particles
    blue_particle = particles[0]  # [2, 8]
    cyan_particle = particles[4]  # [9, 9]

    # Find all distances between the blue particle and all cyan particle images
    min_dist = float('inf')
    nearest_cyan_image = None
    for shifted_positions in all_shifted_positions:
        current_cyan_image = shifted_positions[4]  # Cyan particle's position in each periodic image
        dist = np.linalg.norm(blue_particle - current_cyan_image)
        
        # Plot a line for every distance
        ax.plot([blue_particle[0], current_cyan_image[0]], 
                [blue_particle[1], current_cyan_image[1]], 
                color='gray', linestyle='--', linewidth=1, alpha=0.6)
        
        # Check if it's the minimum distance
        if dist < min_dist:
            min_dist = dist
            nearest_cyan_image = current_cyan_image

    # Highlight the minimum distance
    ax.plot([blue_particle[0], nearest_cyan_image[0]], [blue_particle[1], nearest_cyan_image[1]], 
            color='red', linestyle='-', linewidth=2, label='Minimum Image Distance')

    # Mark the blue and cyan particles clearly
    ax.scatter(blue_particle[0], blue_particle[1], color='blue', s=200, edgecolor='black', label='Blue Particle')
    ax.scatter(nearest_cyan_image[0], nearest_cyan_image[1], color='cyan', s=200, edgecolor='black', label='Nearest Cyan Image')

    # Annotate the minimum distance
    # ax.text((blue_particle[0] + nearest_cyan_image[0]) / 2, (blue_particle[1] + nearest_cyan_image[1]) / 2, 
    #         f'Dist = {min_dist:.2f}', fontsize=10, ha='center')

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Computing Minimum Image Distance')

    # Make the plot square
    ax.set_aspect('equal')

    # Save to PNG
    plt.savefig('pbc_minimum_image_convention_all_distances.png')

def lennard_jones_potential(r, epsilon, sigma):
    """
    Calculate the Lennard-Jones potential for a given distance r.
    
    Parameters:
    r : float or np.array
        Distance between two particles.
    epsilon : float
        Depth of the potential well.
    sigma : float
        Distance at which the potential is zero.
    
    Returns:
    V : float or np.array
        Lennard-Jones potential.
    """
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def plot_lennard_jones_with_cutoff_and_shift(epsilon=1.0, sigma=1.0, r_cutoff=2.5):
    """
    Generate and plot the Lennard-Jones potential with both a cutoff radius
    and the truncated and shifted potential. Includes an inset showing the
    behavior of the potentials near the cutoff radius.
    
    Parameters:
    epsilon : float
        Depth of the potential well.
    sigma : float
        Distance at which the potential is zero.
    r_cutoff : float
        Cutoff radius beyond which interactions are ignored.
    """
    # Generate an array of r values (distance)
    r = np.linspace(0.8, 3.0, 500)  # Avoid r=0 to prevent division by zero

    # Calculate the Lennard-Jones potential
    V = lennard_jones_potential(r, epsilon, sigma)
    
    # Apply cutoff by setting potential to zero beyond r_cutoff
    V_cutoff = np.where(r > r_cutoff, 0, V)

    # Calculate the truncated and shifted potential
    V_shifted = np.where(r > r_cutoff, 0, V - lennard_jones_potential(r_cutoff, epsilon, sigma))

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the full Lennard-Jones potential
    ax.plot(r, V, label='Lennard-Jones Potential', linestyle='--', color='blue')

    # Plot the potential with cutoff
    ax.plot(r, V_cutoff, label=f'With Cutoff at $r_c={r_cutoff}$', color='red')

    # Plot the truncated and shifted potential
    ax.plot(r, V_shifted, label='Truncated and Shifted Potential', color='green')

    # Highlight the cutoff point
    ax.axvline(r_cutoff, color='gray', linestyle=':', label=f'Cutoff Radius $r_c={r_cutoff}$')

    # Labels and title
    ax.set_xlabel('Distance ($r$)')
    ax.set_ylabel('Potential ($V$)')
    ax.set_title('Lennard-Jones Potential with Cutoff and Shifted Potential')
    
    # Set limits for clarity
    ax.set_xlim(0.8, 3.0)
    ax.set_ylim(-1.5, 1.0)
    
    # Add legend
    ax.legend()

    # Inset zooming near the cutoff
    inset_ax = fig.add_axes([0.5, 0.7, 0.35, 0.1])  # [x, y, width, height] for the inset

    # Plot the same potentials in the inset
    inset_ax.plot(r, V, linestyle='--', color='blue')
    inset_ax.plot(r, V_cutoff, color='red')
    inset_ax.plot(r, V_shifted, color='green')

    # Zoomed region around the cutoff radius
    inset_ax.set_xlim(r_cutoff - 0.2, r_cutoff + 0.2)
    inset_ax.set_ylim(-0.03, 0.01)
    inset_ax.axvline(r_cutoff, color='gray', linestyle=':')

    # Inset title and labels
    inset_ax.set_title('Near Cutoff')
    inset_ax.set_xlabel('$r$')
    inset_ax.set_ylabel('$V$')

    # Save the plot as PNG
    plt.savefig('lennard_jones_with_cutoff_and_shift_inset.png')

def main():
    # Call the function to generate and show the plot
    plot_periodic_boundary_conditions_colored()

    # Call the function to generate and show the plot
    plot_all_minimum_image_distances()

    # Call the function to generate and show the plot
    plot_lennard_jones_with_cutoff_and_shift()

if __name__ == '__main__':
    main()
