"""
beamforming_visualization.py

Python script to visualize beamforming patterns for a uniform linear antenna array (ULA).
Demonstrates main lobe steering and array factor calculation.

Author: Kahsay Kiross Meresa
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Simulation Parameters
# ---------------------------
num_antennas = 16          # Number of antennas in the array
d = 0.5                     # Inter-element spacing in wavelength units
theta_scan_deg = 30         # Desired beam steering angle in degrees
theta = np.linspace(-90, 90, 1000)  # Observation angles in degrees

# Convert degrees to radians
theta_rad = np.deg2rad(theta)
theta_scan_rad = np.deg2rad(theta_scan_deg)

# ---------------------------
# Function: Array Factor
# ---------------------------
def array_factor(num_antennas, d, theta, theta_scan):
    """
    Computes the normalized array factor for a uniform linear array.
    
    Parameters:
    - num_antennas: Number of array elements
    - d: Spacing between elements (in wavelength)
    - theta: Observation angles (rad)
    - theta_scan: Beam steering angle (rad)
    
    Returns:
    - AF: Normalized array factor magnitude
    """
    n = np.arange(num_antennas)
    AF = np.sum(np.exp(1j * 2 * np.pi * d * n[:, np.newaxis] * (np.sin(theta) - np.sin(theta_scan))), axis=0)
    AF = np.abs(AF) / num_antennas  # Normalize
    return AF

# ---------------------------
# Main Simulation
# ---------------------------
def main():
    # Compute array factor
    AF = array_factor(num_antennas, d, theta_rad, theta_scan_rad)

    # Convert to dB for plotting
    AF_dB = 20 * np.log10(AF + 1e-12)  # Add epsilon to avoid log(0)

    # Plot the beam pattern
    plt.figure(figsize=(8,5))
    plt.plot(theta, AF_dB)
    plt.title(f'Beamforming Pattern for {num_antennas}-Element ULA')
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
