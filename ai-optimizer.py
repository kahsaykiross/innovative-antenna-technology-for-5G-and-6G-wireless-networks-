"""
ai_optimizer.py

Python script to perform AI-driven antenna optimization.
Uses a simple gradient descent approach to optimize antenna weights
for maximum received signal power in a multi-antenna system.

Author: Kahsay Kiross Meresa
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Simulation Parameters
# ---------------------------
num_antennas = 16           # Number of antennas
num_iterations = 100        # Optimization iterations
learning_rate = 0.1         # Gradient descent step size

# True channel (complex Gaussian)
np.random.seed(42)
h = (1/np.sqrt(2)) * (np.random.randn(num_antennas) + 1j*np.random.randn(num_antennas))

# Initialize antenna weights randomly
w = np.random.randn(num_antennas) + 1j*np.random.randn(num_antennas)
w /= np.linalg.norm(w)  # Normalize initial weights

# ---------------------------
# Function: Received power
# ---------------------------
def received_power(w, h):
    """
    Computes received signal power for given antenna weights and channel.
    """
    return np.abs(np.vdot(h, w))**2

# ---------------------------
# Gradient Descent Optimization
# ---------------------------
power_history = []

for i in range(num_iterations):
    # Compute gradient: derivative of |h^H w|^2 w.r.t w*
    gradient = 2 * h * np.vdot(h, w).conj()
    
    # Update weights
    w += learning_rate * gradient
    
    # Normalize weights
    w /= np.linalg.norm(w)
    
    # Record received power
    power_history.append(received_power(w, h))

# ---------------------------
# Plot Optimization Results
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(power_history, marker='o')
plt.title('AI-Driven Antenna Optimization (Gradient Descent)')
plt.xlabel('Iteration')
plt.ylabel('Received Power')
plt.grid(True)
plt.show()
