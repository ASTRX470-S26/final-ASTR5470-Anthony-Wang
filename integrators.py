import numpy as np
from physics import get_accelerations

def rk4_step(pos, vel, masses, dt, G, c, mode_is_1pn):
    """Performs a single 4th-Order Runge-Kutta integration step."""
    k1_v = get_accelerations(pos, vel, masses, G, c, mode_is_1pn)
    k1_r = vel

    k2_v = get_accelerations(pos + 0.5 * dt * k1_r, vel + 0.5 * dt * k1_v, masses, G, c, mode_is_1pn)
    k2_r = vel + 0.5 * dt * k1_v

    k3_v = get_accelerations(pos + 0.5 * dt * k2_r, vel + 0.5 * dt * k2_v, masses, G, c, mode_is_1pn)
    k3_r = vel + 0.5 * dt * k2_v

    k4_v = get_accelerations(pos + dt * k3_r, vel + dt * k3_v, masses, G, c, mode_is_1pn)
    k4_r = vel + dt * k3_v

    new_pos = pos + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    new_vel = vel + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return new_pos, new_vel

def leapfrog_step(pos, vel, masses, dt, G, c, mode_is_1pn):
    """
    Performs a single Kick-Drift-Kick Leapfrog integration step.
    Note: Standard leapfrog is strictly for position-dependent forces.
    For 1PN (velocity-dependent), RK4 is recommended, but this uses an explicit pseudo-leapfrog.
    """
    # Kick 1 (Half step velocity)
    acc_initial = get_accelerations(pos, vel, masses, G, c, mode_is_1pn)
    v_half = vel + 0.5 * dt * acc_initial
    
    # Drift (Full step position)
    new_pos = pos + dt * v_half
    
    # Kick 2 (Full step velocity using v_half as approximation for velocity-dependent forces)
    acc_final = get_accelerations(new_pos, v_half, masses, G, c, mode_is_1pn)
    new_vel = v_half + 0.5 * dt * acc_final
    
    return new_pos, new_vel