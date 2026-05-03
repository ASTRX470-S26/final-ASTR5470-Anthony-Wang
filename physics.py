import numpy as np
from numba import njit

@njit
def get_accelerations(pos, vel, masses, G, c, mode_is_1pn):
    """
    Calculates the acceleration for N bodies.
    Utilizes Numba JIT for high-performance array operations.
    """
    N = pos.shape[0]
    acc = np.zeros_like(pos)
    
    # 1. Calculate Standard Newtonian Pairwise Gravity
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r_mag = np.linalg.norm(r_vec)
                # Avoid division by zero in case of collisions
                if r_mag > 1e-10: 
                    acc[i] += G * masses[j] * r_vec / (r_mag**3)
                    
    # 2. Add 1PN Relativistic Corrections (if enabled)
    # Assumes index 0 is the dominant central mass for the 1PN approximation
    if mode_is_1pn and N > 1:
        M = masses[0]
        for i in range(1, N):
            r_vec = pos[i] - pos[0]
            r_mag = np.linalg.norm(r_vec)
            v_vec = vel[i]
            
            v_mag_sq = np.sum(v_vec**2)
            r_dot_v = np.dot(r_vec, v_vec)
            
            term1 = (4 * G * M / r_mag) - v_mag_sq
            term2 = 4 * r_dot_v
            
            # 1PN acceleration vector
            acc_1pn = (G * M / (c**2 * r_mag**3)) * (term1 * r_vec + term2 * v_vec)
            acc[i] += acc_1pn
            
    return acc