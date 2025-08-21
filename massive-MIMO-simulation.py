"""
massive_mimo_simulation.py

A basic Python simulation of a Massive MIMO system.
This script calculates the spectral efficiency for a
multi-user MIMO system using Maximum Ratio Transmission (MRT).

Author: Kahsay Kiross Meresa
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Simulation Parameters
# ---------------------------
num_users = 10          # Number of users in the system
num_antennas = 64       # Number of base station antennas (Massive MIMO)
snr_db = np.arange(-10, 21, 2)  # SNR range in dB
snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale

# ---------------------------
# Function: Generate channel matrix
# ---------------------------
def generate_channel(num_users, num_antennas):
    """
    Generates a Rayleigh fading channel matrix H
    with shape (num_users, num_antennas)
    Each element is complex Gaussian with zero mean, unit variance.
    """
    H = (1/np.sqrt(2)) * (np.random.randn(num_users, num_antennas) +
                          1j * np.random.randn(num_users, num_antennas))
    return H

# ---------------------------
# Function: Compute spectral efficiency
# ---------------------------
def compute_spectral_efficiency(H, snr_linear):
    """
    Computes the spectral efficiency (bits/s/Hz) for each SNR.
    Uses Maximum Ratio Transmission (MRT) precoding.
    """
    num_users, num_antennas = H.shape
    SE = []

    for snr in snr_linear:
        # Compute MRT precoding: normalize each user's channel
        W = H.conj().T / np.linalg.norm(H, axis=1)
        
        # Effective channel gain for each user
        signal_power = np.abs(np.sum(H * W.T, axis=1))**2
        
        # Interference: sum power from other users
        interference = np.sum(np.abs(H @ W)**2, axis=1) - signal_power
        
        # Compute SINR
        sinr = signal_power / (interference + 1/snr)
        
        # Spectral efficiency: log2(1 + SINR)
        SE.append(np.sum(np.log2(1 + sinr)))
    
    return np.array(SE)

# ---------------------------
# Main Simulation
# ---------------------------
def main():
    np.random.seed(42)  # For reproducibility

    # Generate channel
    H = generate_channel(num_users, num_antennas)

    # Compute spectral efficiency
    SE = compute_spectral_efficiency(H, snr_linear)

    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(snr_db, SE, marker='o')
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sum Spectral Efficiency (bits/s/Hz)")
    plt.title("Massive MIMO Simulation: Sum Spectral Efficiency vs SNR")
    plt.show()

# ---------------------------
# Run simulation
# ---------------------------
if __name__ == "__main__":
    main()
