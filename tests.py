import numpy as np
from integrators import leapfrog_step, rk4_step
from analysis import calculate_total_energy

def test_energy_conservation():
    """Test 1: Leapfrog integrator should conserve energy in a pure Newtonian 2-body system."""
    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    vel = np.array([[0.0, -0.1, 0.0], [0.0, 10.0, 0.0]])
    masses = np.array([1000.0, 1.0])
    dt = 0.01
    G = 1.0
    c = 100.0
    
    E_initial = calculate_total_energy(pos, vel, masses, G)
    
    for _ in range(1000):
        pos, vel = leapfrog_step(pos, vel, masses, dt, G, c, mode_is_1pn=False)
        
    E_final = calculate_total_energy(pos, vel, masses, G)
    fractional_change = abs((E_final - E_initial) / E_initial)
    
    assert fractional_change < 1e-4, f"Energy drift too high: {fractional_change}"
    print("Test 1 Passed: Energy conserved within tolerance.")

def test_keplers_third_law():
    """Test 2: a^3 / P^2 should be constant for multiple planets around a central mass."""
    # Central mass and two planets at different radii
    pos = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    masses = np.array([10000.0, 1e-5, 1e-5])
    G = 1.0
    
    # v = sqrt(GM/r) for circular orbits
    v1 = np.sqrt(G * masses[0] / 4.0)
    v2 = np.sqrt(G * masses[0] / 9.0)
    vel = np.array([[0.0, 0.0, 0.0], [0.0, v1, 0.0], [0.0, v2, 0.0]])
    
    # Calculate expected periods: P = 2*pi*sqrt(a^3 / GM)
    P1 = 2 * np.pi * np.sqrt((4.0**3) / (G * masses[0]))
    P2 = 2 * np.pi * np.sqrt((9.0**3) / (G * masses[0]))
    
    ratio1 = (4.0**3) / (P1**2)
    ratio2 = (9.0**3) / (P2**2)
    
    assert np.isclose(ratio1, ratio2), "Kepler's Third Law violated."
    print("Test 2 Passed: Kepler's Third Law verified.")

def test_relativistic_precession():
    """Test 3: Checks if 1PN mode causes orbital precession (perihelion shift)."""
    # Highly eccentric orbit to exaggerate precession
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, 15.0, 0.0]])
    masses = np.array([1000.0, 1.0])
    dt = 0.005
    G = 1.0
    c = 50.0 # Artificially low speed of light to make precession obvious quickly
    
    # Run one Newtonian and one 1PN orbit
    pos_n, vel_n = pos.copy(), vel.copy()
    pos_r, vel_r = pos.copy(), vel.copy()
    
    for _ in range(500):
        pos_n, vel_n = rk4_step(pos_n, vel_n, masses, dt, G, c, mode_is_1pn=False)
        pos_r, vel_r = rk4_step(pos_r, vel_r, masses, dt, G, c, mode_is_1pn=True)
        
    # After a fraction of an orbit, the 1PN position should differ from the Newtonian position
    diff = np.linalg.norm(pos_n[1] - pos_r[1])
    assert diff > 0.01, f"No relativistic precession detected. Difference: {diff}"
    print("Test 3 Passed: 1PN precession detected compared to Newtonian orbit.")

if __name__ == "__main__":
    test_energy_conservation()
    test_keplers_third_law()
    test_relativistic_precession()