# Correct RIS-assisted weights by extending RIS to array size
weights_ris = weights_bs.copy()
for h in H_users:
    # Extend RIS vector to match antenna array size
    h_ext = np.zeros(Nx*Ny, dtype=complex)
    h_ext[:len(h)] = h
    weights_ris += h_ext * np.mean(ris_phases)
