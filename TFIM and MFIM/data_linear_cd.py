"""
Data for MFIM_linear_ramp_CD.pdf (Fig. 4, middle panel, of the manuscript).

Krylov counterdiabatic driving applied directly to the *linear* ramp at
h_f = 0.40: end-of-ramp infidelity 1 - P_R against ramp time tau, with no
control, with the ground-state-tailored truncated AGP of Eqs. (F1)-(F3) at
M = 1, 2, 3, 5, and with the exact ground-state AGP of Eq. (20). The figure
shows no CD and M = 1, 2, 3.

Parameters: L = 8, J = 1, eps = 0.5, gamma = 1.5, PBC, h_PT = 0.22.

Usage
-----
    python data_linear_cd.py            # all methods, resumable
    python data_linear_cd.py M3         # one method at a time

Plotting lives in plot.py; nothing here draws.
"""
import os, sys, time
import numpy as np

from core import H_of, dH_of, split, evolve2, tracked_frames, agp_exact_gs, agp_krylov_gs

# ------------------------------------------------------------------ parameters
L, GAM, J, EPS = 8, 1.5, 1.0, 0.5
HF = 0.40
DH = 0.002                      # CD control mesh in h
TAU_MIN, TAU_MAX = 2.0, 300.0
N_BASE = 44
N_ROUNDS = 3
DLOG = 0.30
RATIO_MIN = 1.015
DT = 0.01
CACHE = "fig3.npz"

#: label -> (M, description). M = None is no control, "ex" is the exact GS AGP.
METHODS = {"none": (None, "no CD"),
           "M1":   (1,    r"$M=1$"),
           "M2":   (2,    r"$M=2$"),
           "M3":   (3,    r"$M=3$"),
           "M5":   (5,    r"$M=5$"),
           "exact": ("ex", "exact GS AGP")}

HGRID = np.round(np.arange(0.0, HF + 0.5 * DH, DH), 10)


# ------------------------------------------------------------------ CD operators
def cd_grid(key, frames, dH):
    """A_h on HGRID for one method. Returns None for the uncontrolled ramp."""
    M = METHODS[key][0]
    if M is None:
        return None
    A = np.empty((len(HGRID), 2 ** L, 2 ** L), complex)
    t0 = time.time()
    for i, (Hm, E, R, Li, i0) in enumerate(frames):
        A[i] = (agp_exact_gs(E, R, Li, i0, dH) if M == "ex"
                else agp_krylov_gs(Hm, dH, E, R, Li, i0, M)[0])
    print(f"  CD grid [{key}]: {len(HGRID)} points, {time.time()-t0:.0f}s", flush=True)
    return A


# ------------------------------------------------------------------ dynamics
_A, _B = split(L, GAM, J=J, eps=EPS)


def infidelity(tau, Agrid, frame_hf, dt=DT):
    """1 - P_R at the end of the linear ramp, Eq. (93), with control h-dot A_h."""
    _, E, R, Li, i0 = frame_hf
    hdot = HF / tau

    def hfun(t):
        return HF * min(t / tau, 1.0)

    if Agrid is None:
        acd = None
    else:
        nH = len(HGRID) - 1

        def acd(t):                       # nearest index on the control mesh
            return hdot * Agrid[min(int(round(hfun(t) / DH)), nH)]

    psi = evolve2(_A, _B, hfun, tau, PSI0, max(3000, int(tau / dt)), Acd=acd)
    if psi is None:
        return 1.0
    c = Li @ psi
    return float(1.0 - abs(c[i0]) ** 2 / np.sum(abs(c) ** 2))


# ------------------------------------------------------------------ adaptive sweep
def sweep(Agrid, frame_hf, taus=None, vals=None, budget=None, verbose=True):
    """Log-spaced base grid, then midpoint refinement on steep segments.

    Resumable: pass the cached (taus, vals) to continue refining. `budget` is a
    wall-clock limit in seconds -- the sweep returns whatever it has when the
    budget runs out, so a long run can be continued by calling again.
    """
    t_start = time.time()
    left = lambda: budget is None or (time.time() - t_start) < budget

    if taus is None:
        taus, vals = [], []
        t0 = time.time()
        for t in np.geomspace(TAU_MIN, TAU_MAX, N_BASE):
            if not left():
                break
            taus.append(t)
            vals.append(infidelity(t, Agrid, frame_hf))
        if verbose:
            print(f"    base grid {len(taus)} pts, {time.time()-t0:.0f}s", flush=True)
        if len(taus) < N_BASE:
            return np.asarray(taus), np.asarray(vals), False
    else:
        taus, vals = list(taus), list(vals)

    for r in range(N_ROUNDS):
        lv = np.log10(np.clip(vals, 1e-14, 2.0))
        new = [np.sqrt(taus[i] * taus[i + 1]) for i in range(len(taus) - 1)
               if abs(lv[i + 1] - lv[i]) > DLOG and taus[i + 1] / taus[i] > RATIO_MIN]
        if not new:
            return np.asarray(taus), np.asarray(vals), True
        t0, added = time.time(), 0
        for t in new:
            if not left():
                break
            taus.append(t)
            vals.append(infidelity(t, Agrid, frame_hf))
            added += 1
        o = np.argsort(taus)
        taus, vals = list(np.asarray(taus)[o]), list(np.asarray(vals)[o])
        if verbose:
            print(f"    round {r+1}: +{added}/{len(new)} -> {len(taus)} pts, "
                  f"{time.time()-t0:.0f}s", flush=True)
        if added < len(new):
            return np.asarray(taus), np.asarray(vals), False
    return np.asarray(taus), np.asarray(vals), True


def run_sweep(only=None, budget=230.0):
    """Compute the missing methods, saving after each so the run is resumable.

    Returns True when every requested method is finished. Re-run until it does.
    """
    store = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    keys = [k for k in METHODS if only is None or k == only]
    todo = [k for k in keys if not store.get(f"d_{k}", np.array([False]))[0]]
    if not todo:
        print("all requested methods complete")
        return True
    print("building frames ...", flush=True)
    frames = list(tracked_frames(L, HGRID, GAM, J, EPS))
    dH = dH_of(L, GAM, J=J, eps=EPS).toarray()
    done_all = True
    for key in todo:
        print(f"[{key}] {METHODS[key][1]}", flush=True)
        A = cd_grid(key, frames, dH)
        t_prev = store.get(f"t_{key}")
        v_prev = store.get(f"v_{key}")
        taus, vals, done = sweep(A, frames[-1], t_prev, v_prev, budget=budget)
        del A
        store[f"t_{key}"], store[f"v_{key}"] = taus, vals
        store[f"d_{key}"] = np.array([done])
        np.savez(CACHE, **store)
        print(f"  cached {len(taus)} pts, complete={done}", flush=True)
        done_all &= done
        if not done:
            break
    return done_all


def _psi0():
    """Right ground state of H(0), the initial state for every ramp."""
    e, R = np.linalg.eig(H_of(L, 0.0, GAM, J=J, eps=EPS).toarray())
    return (R / np.linalg.norm(R, axis=0))[:, int(np.argmin(e.real))]


PSI0 = _psi0()


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 230.0
    sys.exit(0 if run_sweep(only, budget) else 1)
