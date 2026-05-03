import os
import json
import argparse
import numpy as np
from integrators import rk4_step, leapfrog_step
from analysis import calculate_total_energy, plot_trajectories, plot_energy_error

def load_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="GR-NBody: N-Body Integrator with 1PN Corrections")
    parser.add_argument("config", help="Path to the configuration JSON file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    dt = config['dt']
    t_max = config['t_max']
    G = config['G']
    c = config['c']
    mode_is_1pn = (config['physics_mode'].lower() == '1pn')
    step_func = rk4_step if config['method'].lower() == 'rk4' else leapfrog_step

    particles = config['particles']
    N = len(particles)
    
    # Initialize arrays
    masses = np.array([p['mass'] for p in particles], dtype=np.float64)
    pos = np.array([p['position'] for p in particles], dtype=np.float64)
    vel = np.array([p['velocity'] for p in particles], dtype=np.float64)
    names = [p['name'] for p in particles]

    steps = int(t_max / dt)
    
    # Pre-allocate history arrays for data output
    history_pos = np.zeros((steps, N, 3))
    history_energy = np.zeros(steps)

    print(f"Starting integration for {steps} steps using {config['method']}...")

    # Main Integration Loop
    for step in range(steps):
        history_pos[step] = pos
        history_energy[step] = calculate_total_energy(pos, vel, masses, G)
        
        pos, vel = step_func(pos, vel, masses, dt, G, c, mode_is_1pn)
        
        if step % (steps // 10) == 0:
            print(f"Progress: {step / steps * 100:.0f}%")

    print("Simulation complete. Generating outputs...")
    
    # Ensure the outputs directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Outputs with updated paths
    plot_trajectories(history_pos, names, filename=os.path.join(output_dir, "trajectories.png"))
    plot_energy_error(history_energy, filename=os.path.join(output_dir, "energy_error.png"))
    
    # Save raw data to outputs folder
    flattened_data = history_pos.reshape(steps, -1)
    data_path = os.path.join(output_dir, "trajectory_data.csv")
    np.savetxt(data_path, flattened_data, delimiter=",")
    print(f"Saved raw data: {data_path}")

if __name__ == "__main__":
    main()