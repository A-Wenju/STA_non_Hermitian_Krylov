"""
Physics for HN_OBC_CD_fig: exact-CD counterdiabatic driving of the OPEN
Hatano-Nelson model (Eq. 55/56 of the manuscript).

With open boundaries the imaginary gauge S(h) = exp(-a h X) gives an exact
adiabatic gauge potential A_h = -i a X (see core.py), so the CD drive is
transitionless at any ramp speed: the excess energy with CD is zero to
integrator precision at every tau. That is the point of the figure -- it
contrasts the bare ramp's decaying-but-nonzero excess energy against the
exact CD ramp's flat zero.

Computes E(tau) = <H(h0)>/<psi|psi> - E_0, complex, for the bare ramp and
for the exact-CD ramp on a tau grid, and caches to HN_OBC_CD_fig.npz.

Usage:  python3 data_obc_cd.py
"""

import numpy as np

from core import model_operators, hamiltonian, evolve_obc_exact, evolve_obc_stepper


def excess_energy(tau, ops, use_cd, J=1.0, U=1.0, a=1.0, h0=0.1):
    """E(tau), complex, from the exact closed-form OBC propagator."""
    psi = evolve_obc_exact(tau, ops, use_cd, J, U, a, h0)
    H_f = hamiltonian(h0, ops, J, U, a)
    E0 = np.linalg.eigvalsh(hamiltonian(0.0, ops, J, U, a).real)[0]
    return np.vdot(psi, H_f @ psi) / np.vdot(psi, psi) - E0


def verify_obc(N=8):
    """Cross-check the closed form against the general time stepper."""
    ops = model_operators(N, N // 2, pbc=False)
    print(f"verification, N = {N}, D = {ops[-1]}")
    print("   tau     stepper                    closed form                 |diff|")
    for tau in (1.0, 5.0, 20.0):
        for use_cd in (False, True):
            psi_s = evolve_obc_stepper(tau, ops, use_cd)
            psi_f = evolve_obc_exact(tau, ops, use_cd)
            H_f = hamiltonian(0.1, ops)
            E0 = np.linalg.eigvalsh(hamiltonian(0.0, ops).real)[0]
            s_ = np.vdot(psi_s, H_f @ psi_s) / np.vdot(psi_s, psi_s) - E0
            f_ = np.vdot(psi_f, H_f @ psi_f) / np.vdot(psi_f, psi_f) - E0
            tag = "CD " if use_cd else "bare"
            print(f"  {tau:5.1f} {tag}  {s_:+.9f}  {f_:+.9f}  {abs(s_ - f_):.2e}")


def run(N=10, J=1.0, U=1.0, a=1.0, h0=0.1, taus=None, outfile="HN_OBC_CD_fig.npz"):
    if taus is None:
        taus = np.linspace(0.5, 50.0, 60)

    ops = model_operators(N, N // 2, pbc=False)
    print(f"N = {N}, Hilbert-space dimension D = {ops[-1]}")

    E_bare = np.array([excess_energy(t, ops, False, J, U, a, h0) for t in taus])
    print("  bare ramp done")
    E_cd = np.array([excess_energy(t, ops, True, J, U, a, h0) for t in taus])
    print("  counterdiabatic ramp done")
    print(f"  max |E| with exact CD = {np.abs(E_cd).max():.3e}  (should be ~0)")

    np.savez(outfile, taus=taus, E_bare=E_bare, E_cd=E_cd,
             N=np.array([N]), h0=np.array([h0]))
    print(f"saved {outfile}")
    return taus, E_bare, E_cd


if __name__ == "__main__":
    verify_obc()
    run()
