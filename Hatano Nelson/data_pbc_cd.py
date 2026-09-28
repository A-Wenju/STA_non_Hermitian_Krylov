"""
Physics for HN_PBC_CD: excess energy at the bare-ramp oscillation maxima,
periodic boundaries, bare ramp vs. truncated-Krylov ground-state-Galerkin CD
(and the ground-state exact AGP as a reference); see core.py for the Krylov AGP
construction.

Step 1  Scan |E(tau)| for the bare ramp on a dense LINEAR tau grid and pick
        out the local maxima. The PBC oscillation has period ~2.9, so a log
        grid aliases it; the maxima are the envelope plotted in the figure.
        This step uses the fast sparse propagator (core.sparse_ops /
        evolve_fast), since only the LOCATION of the maxima is needed.

Step 2  At those tau only, recompute the excess energy with the dense
        stepper (core.evolve_cd), for the bare ramp and for the truncated
        Krylov AGP fixed by the ground-state Galerkin projection
        (core.agp_grid, mode='gs_galerkin') at M = M_VALUES (1, 3, 5, 7),
        plus the ground-state-projected exact AGP as a reference. Bare and CD
        go through the SAME propagator so the comparison is not
        contaminated by stepper differences.

Parallelism: the scan in Step 1 evaluates independent tau points, and the
variants in Step 2 (different M, plus "no CD" and "exact") are independent of
each other, so both loops are farmed out to a process pool. Within a single
variant, agp_grid still walks h sequentially -- it tracks the ground state by
branch continuation, an inherently serial dependency that must not be
parallelized.

Reference energy is E_ad(h_f), the eigenvalue adiabatically continued from
the t=0 ground state, not E_0(h=0): at PBC the difference is a constant
offset larger than the excess itself.

Output: HN_PBC_CD.npz, consumed by plot.py (pbc_cd).

Usage:  python3 data_pbc_cd.py [h_f] [tau_hi]
"""

import os
import sys
import time
import concurrent.futures as cf

import numpy as np

from core import (model_operators, hamiltonian, adiabatic_reference, envelope,
                  sparse_ops, excess_fast, evolve_cd, agp_grid)

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

H_F = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
TAU_LO, TAU_HI = 0.5, float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
DTAU_SCAN = 0.5          # dense linear spacing for maxima detection
DT_SCAN = 0.15           # step size for the (cheap) maxima scan
DT_EVAL = 0.05           # step size for the reported numbers
N_GRID = 100             # grid in h on which A^(M)(h) is precomputed
N, NP = 10, 5

N_WORKERS = os.cpu_count() or 4

M_VALUES = np.array([1, 3, 5, 7])

VARIANTS = ([("no CD", None, None)]
            + [(f"g.s. Galerkin, M={M}", "gs_galerkin", int(M)) for M in M_VALUES]
            + [("exact g.s. AGP", "gs_exact", 1)])


# ----------------------------------------------------------------------
# Per-worker state, set once by the pool initializer so ops/H_f/E_ad are
# pickled across the process boundary only when a worker starts, not on
# every task.
# ----------------------------------------------------------------------

_W = {}


def _init_scan(ops, h_f, H_f, E_ad):
    _W["sops"] = sparse_ops(ops)
    _W["H_f"] = H_f
    _W["E_ad"] = E_ad
    _W["h_f"] = h_f


def _scan_one(t):
    w = _W
    return abs(excess_fast(t, w["sops"], w["H_f"], w["E_ad"], w["h_f"],
                           dt_target=DT_SCAN))


def _init_variant(ops, h_f, t_max, E_ad):
    _W["ops"] = ops
    _W["h_f"] = h_f
    _W["t_max"] = t_max
    _W["E_ad"] = E_ad


def _variant_one(spec):
    lab, mode, M = spec
    ops, h_f, t_max, E_ad = _W["ops"], _W["h_f"], _W["t_max"], _W["E_ad"]
    if mode is None:
        E = excess_at(ops, h_f, t_max, E_ad)
        nrm = 0.0
    else:
        hs, Ag = agp_grid(ops, h_f, M, mode, n_grid=N_GRID)
        nrm = float(np.linalg.norm(Ag[-1]))
        E = excess_at(ops, h_f, t_max, E_ad, hs, Ag)
    return lab, E, nrm


# ----------------------------------------------------------------------
# Step 1: locate the maxima of the bare curve (parallel scan over tau)
# ----------------------------------------------------------------------

def bare_maxima(ops, h_f, tau_lo, tau_hi, dtau=DTAU_SCAN, w=2):
    H_f = hamiltonian(h_f, ops)
    E_ad = adiabatic_reference(ops, h_f)
    taus = np.arange(tau_lo, tau_hi + dtau, dtau)
    with cf.ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_scan,
                                initargs=(ops, h_f, H_f, E_ad)) as ex:
        vals = np.array(list(ex.map(_scan_one, taus, chunksize=4)))
    t_max, v_max = envelope(taus, vals, w=w)
    return taus, vals, np.array(t_max), np.array(v_max), E_ad


# ----------------------------------------------------------------------
# Step 2: bare and CD excess energy at those tau
# ----------------------------------------------------------------------

def excess_at(ops, h_f, taus, E_ad, hs=None, Ag=None, dt_target=DT_EVAL):
    H_f = hamiltonian(h_f, ops)
    out = np.empty(len(taus), dtype=complex)
    for i, t in enumerate(taus):
        psi = evolve_cd(t, ops, h_f, hs, Ag, dt_target=dt_target)
        out[i] = np.vdot(psi, H_f @ psi) / np.vdot(psi, psi) - E_ad
    return out


def run(h_f=H_F, tau_hi=TAU_HI, outfile="HN_PBC_CD.npz"):
    t0 = time.time()
    ops = model_operators(N, NP, pbc=True)
    print(f"N = {N}, D = {ops[-1]}, PBC, h_f = {h_f}, "
          f"tau in [{TAU_LO}, {tau_hi}], {N_WORKERS} workers, "
          f"M in {list(M_VALUES)}")

    taus_s, vals_s, t_max, v_max, E_ad = bare_maxima(ops, h_f, TAU_LO, tau_hi)
    print(f"  scanned {len(taus_s)} ramps, found {len(t_max)} maxima "
          f"(spacing {np.median(np.diff(t_max)):.2f})   E_ad = {E_ad:.6f}"
          f"   [{time.time()-t0:.0f}s]")

    with cf.ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_variant,
                                initargs=(ops, h_f, t_max, E_ad)) as ex:
        results = {lab: (E, nrm) for lab, E, nrm in
                   ex.map(_variant_one, VARIANTS, chunksize=1)}

    for lab, mode, M in VARIANTS:
        E, nrm = results[lab]
        print(f"  {lab:24s} |E| at tau_first = {abs(E[0]):.3e}, "
              f"tau_last = {abs(E[-1]):.3e}, ||A||_F(h_f) = {nrm:7.3f}"
              f"   [{time.time()-t0:.0f}s]")

    E_bare, _ = results["no CD"]
    E_exact, exact_norm = results["exact g.s. AGP"]
    E_gal = np.array([results[f"g.s. Galerkin, M={M}"][0] for M in M_VALUES])
    norm_gal = np.array([results[f"g.s. Galerkin, M={M}"][1] for M in M_VALUES])

    out = {"taus": t_max, "scan_taus": taus_s, "scan_vals": vals_s,
           "E_ad": np.array([E_ad]), "h_f": np.array([h_f]),
           "no_CD": E_bare,
           "galerkin_M": M_VALUES, "galerkin_E": E_gal, "galerkin_norm": norm_gal,
           "exact_E": E_exact, "exact_norm": np.array([exact_norm])}
    np.savez(outfile, **out)
    print(f"saved {outfile}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    run()
