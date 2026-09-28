"""
Physics for HN_PBC_Excess_Energy: condition-(17) admissible ramp window and
excess-energy ramps for the PERIODIC Hatano-Nelson model, restricted to (and
swept somewhat past) the window where it holds.

Condition (17) of the manuscript, adapted to the tracked ground state
(Eq. 58), reads

    W(tau) = exp[ int_0^tau max_n Im( E_n(t') - E_0(t') ) dt' ]  ~  1 .

For a linear ramp h(t) = h_f t/tau the time integral factorises (Eq. 59),

    Gamma(h_f, tau) = int_0^tau mu dt = (tau/h_f) int_0^{h_f} mu(h) dh = tau I(h_f) ,
    mu(h) = max_n Im( E_n(h) - E_0(h) ) ,

with I(h_f) in units of energy, and W = exp(Gamma). The condition W ~ 1
therefore fixes tau_max(h_f) = Gamma_target / I(h_f).

Two regimes:
  * h_f <= h_c : the whole spectrum is real, mu == 0, I = 0, tau_max = inf.
    Condition (17) holds identically at any ramp speed.
  * h_f >  h_c : I > 0 and (17) holds only for tau <= tau_max(h_f), a
    decreasing function of h_f.

Excess energy is computed at h_f = 0.10 (bare ramp, no CD; see
data_pbc_cd.py for the Krylov-CD comparison at the same h_f), plus an OBC
reference curve at the same h_f, both against E_ad(h_f) (the eigenvalue
adiabatically continued from the t=0 ground state).

Output: HN_PBC_Excess_Energy.npz, consumed by plot.py (pbc_excess).

Usage:  python3 data_pbc_excess.py
"""

import numpy as np

from core import model_operators, hamiltonian, condition17, tau_max, excess_series


def run(N=10, J=1.0, U=1.0, a=1.0, h_max=0.15, gamma_target=1.0,
        h_finals=(0.10,), tau_extend=10.0, outfile="HN_PBC_Excess_Energy.npz"):
    Np = N // 2
    ops_p = model_operators(N, Np, pbc=True)
    ops_o = model_operators(N, Np, pbc=False)
    print(f"N = {N}, D = {ops_p[-1]}, PBC, J={J}, U={U}, a={a}")

    hs, I, mu_max, E = condition17(ops_p, h_max, J, U, a)
    tm = tau_max(hs, I, gamma_target)

    # threshold h_c: first h where any branch acquires an imaginary part
    nz = np.where(np.abs(E.imag).max(axis=1) > 1e-9)[0]
    h_c = hs[nz[0]] if len(nz) else np.inf
    print(f"  h_c (spectrum becomes complex) = {h_c:.4f}")
    print(f"  below h_c: I = 0, condition (17) holds identically at any tau\n")

    print(f"  tau_max for Gamma = {gamma_target} (i.e. LHS of Eq. 17 = e^{gamma_target:.0f}):")
    print("     h_f     mu_max(h_f)      I(h_f)     tau_max   tau(G=0.5)  tau(G=0.1)")
    for hf in list(h_finals) + [0.04, 0.06, 0.10, 0.15]:
        k = np.argmin(np.abs(hs - hf))
        if I[k] <= 1e-12:
            print(f"   {hs[k]:6.4f}   {mu_max[k]:10.5f}   {I[k]:10.5f}       inf         inf         inf")
        else:
            print(f"   {hs[k]:6.4f}   {mu_max[k]:10.5f}   {I[k]:10.5f}  {1/I[k]:10.2f}  "
                  f"{0.5/I[k]:10.2f}  {0.1/I[k]:10.2f}")

    # ---- excess energy, swept somewhat past the admissible window ----
    # tmax is the Eq. (17) boundary Gamma_target / I(h_f), Eq. (59);
    # tau_sample_max is how far the ramp is actually run, tau_extend > 1
    # pushes the sweep beyond tau_max so the post-tau_max divergence is visible.
    series = {}
    for hf in h_finals:
        k = np.argmin(np.abs(hs - hf))
        tmax = gamma_target / I[k] if I[k] > 1e-12 else 200.0
        tau_sample_max = tau_extend * tmax
        # dense at short times (the PBC oscillation has period ~3), log beyond
        tsw = min(30.0, tau_sample_max)
        taus = np.unique(np.concatenate([
            np.linspace(0.25, tsw, 400),
            np.geomspace(tsw, tau_sample_max, 80) if tau_sample_max > tsw else [tsw]]))
        series[hf] = (taus, tmax, excess_series(ops_p, hf, taus, J, U, a, dt_target=0.04))
        print(f"  h_f = {hf}: ramps up to tau = {tau_sample_max:.1f} "
              f"(tau_max = {tmax:.1f}) done")

    # OBC reference at the largest h_f, same tau grid
    hf0 = h_finals[-1]
    t_ref = series[hf0][0]
    E_obc = excess_series(ops_o, hf0, t_ref, J, U, a, dt_target=0.04)
    print("  OBC reference done")

    np.savez(outfile, hs=hs, I=I, mu_max=mu_max, tm=tm, h_c=h_c,
             **{f"s{h}": series[h][2] for h in h_finals},
             **{f"t{h}": series[h][0] for h in h_finals},
             t_ref=t_ref, E_obc=E_obc)
    print(f"saved {outfile}")
    return hs, I, series


if __name__ == "__main__":
    run()
