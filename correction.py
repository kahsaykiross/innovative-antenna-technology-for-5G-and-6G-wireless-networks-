# Extend all user channels to Nx*Ny size
H_users_ext = []
for h in H_users:
    h_ext = np.zeros(Nx*Ny, dtype=complex)
    h_ext[:len(h)] = h
    H_users_ext.append(h_ext)

# RIS-assisted weights
weights_ris = weights_bs.copy()
for h in H_users_ext:
    weights_ris += h*np.mean(ris_phases)

# Now compute metrics using extended channels
sum_rate, fairness, energy_eff = calculate_metrics(weights_ris, H_users_ext, SNR_dB)
