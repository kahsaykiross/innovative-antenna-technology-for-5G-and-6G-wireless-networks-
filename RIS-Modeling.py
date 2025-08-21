"""
ris_modeling.py

Python script to model a Reconfigurable Intelligent Surface (RIS) in a wireless communication system.
Simulates the effect of adjustable phase shifts on signal reflection and received power.

Author: Kahsay Kiross Meresa
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Simulation Parameters
# ---------------------------
num_elements = 64         # Number of RIS elements
theta_inc_deg = 30        # Incident signal angle (degrees)
theta_ref_deg = 0         # Desired reflection angle (degrees)
wavelength = 1            # Normalized wavelength
d = 0.5 * wavelength      # Inter-element spacing

# Convert degrees to radians
theta_inc = np.deg2rad(theta_inc_deg)
theta_ref = np.deg2rad(theta_ref_deg)

# ---------------------------
# Function: Compute RIS phase shifts
# ---------------------------
def ris_phase_shifts(num_elements, d, theta_inc, theta_ref):
    """
    Computes phase shifts for RIS elements to reflect the signal toward desired angle.
    
    Parameters:
    - num_elements: Number of RIS elements
    - d: Inter-element spacing
    - theta_inc: Incident angle (rad)
    - theta_ref: Reflection angle (rad)
    
    Returns:
    - phase_shifts: Array of phase shifts for each RIS element
    """
    n = np.arange(num_elements)
    phase_shifts = -2 * np.pi * d * n * (np.sin(theta_ref) - np.sin(theta_inc))
    return phase_shifts

# ---------------------------
# Function: Compute array factor
# ---------------------------
def ris_array_factor(phase_shifts, theta_obs):
    """
    Computes normalized RIS array factor for observation angles.
    
    Parameters:
    - phase_shifts: RIS phase shifts
    - theta_obs: Observation angles (rad)
    
    Returns:
    - AF: Normalized array factor magnitude
    """
    n = np.arange(len(phase_shifts))
    AF = np.sum(np.exp(1j * (2 * np.pi * d * n[:, np.newaxis] * np.sin(theta_obs) + phase_shifts[:, np.newaxis])), axis=0)
    AF = np.abs(AF) / len(phase_shifts)
    return AF

# ---------------------------
# Main Simulation
# ---------------------------
def main():
    # Observation angles
    theta = np.linspace(-90, 90, 1000)
    theta_rad = np.deg2rad(theta)

    # Compute RIS phase shifts
    phase_shifts = ris_phase_shifts(num_elements, d, theta_inc, theta_ref)

    # Compute array factor
    AF = ris_array_factor(phase_shifts, theta_rad)
    AF_dB = 20 * np.log10(AF + 1e-12)  # Convert to dB

    # Plot RIS beam pattern
    plt.figure(figsize=(8,5))
    plt.plot(theta, AF_dB)
    plt.title(f'RIS Reflection Pattern for {num_elements}-Element Surface')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Magnitude (dB)')
    plt.grid(True)
    plt.ylim([-40, 0])
    plt.show()

# ---------------------------
# Run simulation
# ---------------------------
if __name__ == "__main__":
    main()
