import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8, 6))

def plot_thermo_system():
    # System
    system = patches.Rectangle((0.3, 0.3), 0.4, 0.4, fill=True, facecolor='lightblue')
    ax.add_patch(system)
    plt.text(0.5, 0.5, 'System', fontsize=14, ha='center', va='center', weight='bold')

    # Surroundings
    surroundings = patches.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(surroundings)
    plt.text(0.1, 0.1, 'Surroundings', fontsize=12, ha='left', va='bottom')

    # Boundary
    boundary = patches.Rectangle((0.3, 0.3), 0.4, 0.4, fill=False, edgecolor='blue', linewidth=1)
    ax.add_patch(boundary)
    plt.text(0.5, 0.25, 'Boundary', fontsize=12, ha='center', va='center', color='blue')

    # Format
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('thermo_system.png')

def plot_types_of_thermo_systems():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Titles for each subplot
    titles = ['Isolated System', 'Closed System', 'Open System']
    boundary_linewidths = [10, 1, 1]
    boundary_linestyles = ['-', '-', '--']

    # Create rectangles for each system
    for i, ax in enumerate(axs):
        # System
        system = patches.Rectangle((0.3, 0.3), 0.4, 0.4, fill=True, facecolor='lightblue')
        ax.add_patch(system)
        ax.text(0.5, 0.5, 'System', fontsize=14, ha='center', va='center', weight='bold')

        # Surroundings
        surroundings = patches.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(surroundings)
        ax.text(0.1, 0.1, 'Surroundings', fontsize=12, ha='left', va='bottom')

        # Boundary
        boundary = patches.Rectangle((0.3, 0.3), 0.4, 0.4, fill=False, edgecolor='blue', linewidth=boundary_linewidths[i], linestyle=boundary_linestyles[i])
        ax.add_patch(boundary)
        ax.text(0.3, 0.25, 'Boundary', fontsize=12, ha='center', va='center', color='blue')

        # Common properties for all subplots
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(titles[i])

    # Annotate isolated system (No heat or mass flow)
    axs[0].annotate('No Heat Flow', xy=(0.5, 0.7), xytext=(0.5, 0.9),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=7))
    axs[0].annotate('No Mass Flow', xy=(0.5, 0.3), xytext=(0.5, 0.1),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=7))

    # Annotate closed system (Heat flow but no mass flow)
    axs[1].annotate('Heat Flow', xy=(0.5, 0.6), xytext=(0.5, 0.9),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=7))
    axs[1].annotate('No Mass Flow', xy=(0.5, 0.3), xytext=(0.5, 0.1),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=7))

    # Annotate open system (Heat and mass flow)
    axs[2].annotate('Heat Flow', xy=(0.5, 0.6), xytext=(0.5, 0.9),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=7))
    axs[2].annotate('Mass Flow', xy=(0.5, 0.4), xytext=(0.5, 0.1),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=7))

    # Show the plot
    plt.tight_layout()
    plt.savefig('types_of_thermo_systems.png')

def zeroth_law_diagram(filename='zeroth_law.png'):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Coordinates for the systems
    positions = {'A': (1, 2), 'C': (3, 2), 'B': (5, 2)}
    
    # Draw circles representing the systems
    for system, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.3, color='skyblue', ec='black', lw=2)
        ax.add_patch(circle)
        plt.text(x, y, system, fontsize=15, ha='center', va='center', fontweight='bold')
    
    # Draw arrows to indicate thermal equilibrium
    ax.annotate('', xy=(2.7, 2), xytext=(1.3, 2), arrowprops=dict(arrowstyle='<->', lw=2))
    ax.annotate('', xy=(3.3, 2), xytext=(4.7, 2), arrowprops=dict(arrowstyle='<->', lw=2))
    ax.annotate('', xy=(1, 1.6), xytext=(5, 1.6), arrowprops=dict(arrowstyle='<->', lw=2, linestyle='dashed', color='grey'))
    
    # Text labels for the thermal equilibrium
    plt.text(2, 2.1, 'Thermal \n Equilibrium', fontsize=10, ha='center', va='bottom')
    plt.text(4, 2.1, 'Thermal \n Equilibrium', fontsize=10, ha='center', va='bottom')
    plt.text(3, 1.5, 'Thermal \n Equilibrium', fontsize=10, ha='center', va='top', color='grey')
    
    # Set limits and aspect
    ax.set_xlim(0, 6)
    ax.set_ylim(1, 3)
    ax.set_aspect('equal')
    
    # Remove axis
    ax.axis('off')
    
    # Save the figure as a PNG file
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Diagram saved as {filename}")

def spring_diagram(filename='spring_diagram.png'):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))

    # Particle positions
    particle1_x, particle1_y = 1, 2
    particle2_x, particle2_y = 7, 2
    
    # Draw particles as circles
    particle_radius = 0.3
    circle1 = plt.Circle((particle1_x, particle1_y), particle_radius, color='skyblue', ec='black', lw=2)
    circle2 = plt.Circle((particle2_x, particle2_y), particle_radius, color='lightcoral', ec='black', lw=2)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Draw the spring
    def draw_spring(ax, x_start, x_end, y, num_coils=10, amplitude=0.3):
        x_spring = np.linspace(x_start, x_end, 100)
        y_spring = y + amplitude * np.sin(2 * np.pi * num_coils * (x_spring - x_start) / (x_end - x_start))
        ax.plot(x_spring, y_spring, color='black', lw=2)
    
    draw_spring(ax, particle1_x + particle_radius, particle2_x - particle_radius, particle1_y)
    
    # Annotate particles
    ax.text(particle1_x, particle1_y + 0.5, 'Particle 1', fontsize=12, ha='center', va='bottom')
    ax.text(particle2_x, particle2_y + 0.5, 'Particle 2', fontsize=12, ha='center', va='bottom')
    
    # Set limits and aspect
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    
    # Remove axis
    ax.axis('off')
    
    # Save the figure as a PNG file
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Schematic saved as {filename}")

import matplotlib.pyplot as plt
import numpy as np

def create_pt_phase_diagram(filename='pt_phase_diagram.png'):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Triple point
    T_t = 216.58  # Triple point temperature for CO2 in K
    P_t = 5.185  # Triple point pressure for CO2 in bar

    # Critical point
    T_c = 304.18  # Critical point temperature for CO2 in K

    # Latent heats
    L_s = 26.1  # Latent heat of sublimation for CO2 in kJ/mol
    L_v = 16.55  # Latent heat of vaporization for CO2 in kJ/mol

    # Solid-gas
    T_low = 200
    T_sg = np.linspace(T_t, T_low, 100)  # Temperature range for solid-gas boundary
    P_sg = P_t * np.exp(-L_s * 1000 / 8.314 * (1 / T_sg - 1 / T_t))  # Clausius-Clapeyron equation

    # Liquid-gas
    T_lg = np.linspace(T_t, T_c, 100)  # Temperature range for liquid-gas boundary
    P_lg = P_t * np.exp(-L_v * 1000 / 8.314 * (1 / T_lg - 1 / T_t))  # Clausius-Clapeyron equation

    # Solid-liquid
    dS_fus = 40  # Entropy of fusion for CO2 in J/(mol K)
    dV_fus = 1.17e-5  # Volume change of fusion for CO2 in m^3/mol
    T_sl = np.linspace(T_t, T_c, 100)  # Temperature range for solid-liquid boundary
    dP_over_dT = dS_fus / dV_fus  # Clapeyron equation
    P_sl = P_t + dP_over_dT * (T_sl - T_t)
    
    # Plot the phase boundaries
    ax.plot(T_sg, P_sg, label='Solid-Gas Boundary', color='orange')
    ax.plot(T_lg, P_lg, label='Liquid-Gas Boundary', color='blue')
    ax.plot(T_sl, P_sl, label='Solid-Liquid Boundary', color='green')

    # Add triple point
    ax.scatter(216.6, 5.185, color='red', zorder=5, label='Triple Point')
    
    # Annotate regions
    plt.text(208, 13, 'Solid', fontsize=12, ha='center', va='bottom')
    plt.text(230, 13, 'Liquid', fontsize=12, ha='center', va='bottom')
    plt.text(270, 13, 'Gas', fontsize=12, ha='center', va='bottom')
    
    # Labels and titles
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_yscale('log')  # Use logarithmic scale for pressure
    ax.set_xlim(200, 300)
    ax.set_ylim(1, 100)
    ax.grid(True, which="both", ls="--")
    
    # Add legend
    ax.legend()
    
    # Save to PNG file
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == '__main__':
    plot_thermo_system()
    plot_types_of_thermo_systems()

    # Create the zeroth law diagram
    zeroth_law_diagram()

    # Create the spring schematic
    spring_diagram()

    # Create and save the P-T phase diagram as a PNG file
    create_pt_phase_diagram()
