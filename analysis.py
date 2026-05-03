import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_total_energy(pos, vel, masses, G):
    """Calculates the total Newtonian mechanical energy of the system."""
    N = pos.shape[0]
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * vel**2)
    potential = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            r_mag = np.linalg.norm(pos[i] - pos[j])
            potential -= G * masses[i] * masses[j] / r_mag
            
    return kinetic + potential

def plot_trajectories(history_pos, names, filename="outputs/trajectories.png"):
    """Plots the 2D orbital trajectories."""
    plt.figure(figsize=(8, 8))
    N = history_pos.shape[1]
    
    for i in range(N):
        plt.plot(history_pos[:, i, 0], history_pos[:, i, 1], label=names[i])
        plt.scatter(history_pos[0, i, 0], history_pos[0, i, 1], marker='x') # Start
        
    plt.title("N-Body Orbital Trajectories")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close() # Good practice to close the plot to free memory

def plot_energy_error(history_energy, filename="outputs/energy_error.png"):
    """Plots the fractional energy error over time."""
    plt.figure(figsize=(8, 4))
    E0 = history_energy[0]
    
    # Avoid division by zero if initial energy is perfectly zero
    if E0 == 0:
        fractional_error = np.abs(history_energy)
    else:
        fractional_error = np.abs((history_energy - E0) / E0)
    
    plt.plot(fractional_error, color='red')
    plt.yscale('log')
    plt.title("Fractional Energy Error |ΔE / E0|")
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()