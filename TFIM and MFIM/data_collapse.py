"""
Data for MFIM_collapse_Gamma.pdf (Fig. 4, left panel, of the manuscript).

Infidelity 1 - P_R at the end of a linear ramp, for four values of h_f, plus
the scaling variable Gamma = tau * I(h_f), with

    I(h_f) = (1/h_f) int_0^{h_f} max_n mu_n(h) dh ,   mu_n = Im[E_n(h) - E_0(h)] ,

as in Eq. (59): the max over branches at each h, integrated over the ramp.
`max_I` below uses Hungarian assignment to track branches consistently
step to step so the *ground* branch's energy E_0(h_f) (used for the dynamics
in infidelity() below) stays continuous -- the max itself doesn't care about
branch identity, since max is invariant to how the other columns are
labelled.

Parameters: L = 8, J = 1, eps = 0.5, gamma = 1.5, PBC, h_PT = 0.22.

Usage
-----
    python data_collapse.py            # sweep, caches to fig2a.npz (+ Idat.npz), resumable

Plotting lives in plot.py; nothing here draws.
"""
import os, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from core import H_of, dH_of, split, evolve2, frame

# ------------------------------------------------------------------ parameters
L, GAM, J, EPS = 8, 1.5, 1.0, 0.5
H_PT = 0.22
HFS = (0.26, 0.32, 0.40, 0.50)

TAU_MIN, TAU_MAX = 0.5, 400.0
N_BASE = 56            # base log-spaced points per curve
N_ROUNDS = 3            # rounds of adaptive refinement
DLOG = 0.30              # refine when |d log10(1-P_R)| exceeds this
RATIO_MIN = 1.015        # stop refining when tau_{i+1}/tau_i falls below this
DT = 0.01                # RK4 step; dt*||H|| ~ 0.12
CACHE = "fig2a.npz"


# ------------------------------------------------------------------ frames
def max_I(hmax=0.55, npts=551):
    """I(h_f) for each h_f, Eq. (59): max_n mu_n(h), integrated
    over h. Branches are Hungarian-tracked so the ground branch's own
    energy E_0(h_f) is correct -- but the max itself needs no branch identity,
    since max over a row is invariant to how the other columns are ordered."""
    hs = np.linspace(0.0, hmax, npts)
    D = 2 ** L
    E_all = np.zeros((npts, D), complex)
    prevR = None
    for i, h in enumerate(hs):
        Hm = H_of(L, h, GAM, J=J, eps=EPS).toarray()
        E, R = np.linalg.eig(Hm)
        R = R / np.linalg.norm(R, axis=0)
        Li = np.linalg.inv(R)
        if prevR is None:
            order = np.argsort(E.real)
        else:
            row, col = linear_sum_assignment(-np.abs(Li @ prevR))
            order = np.empty(D, dtype=int)
            order[col] = row
        E, R = E[order], R[:, order]
        E_all[i] = E
        prevR = R.copy()
    mu_max = (E_all - E_all[:, [0]]).imag.max(axis=1)     # max over branches, Eq. (59)
    out = {}
    for hf in HFS:
        k = np.searchsorted(hs, hf) + 1
        I = np.trapezoid(mu_max[:k], hs[:k]) / hs[k - 1]
        out[hf] = (float(I), complex(E_all[k - 1, 0]))
    return out


# ------------------------------------------------------------------ dynamics
_A, _B = split(L, GAM, J=J, eps=EPS)


_, _e0, _R0, _, _ = frame(L, 0.0, GAM, J=J, eps=EPS)
PSI0 = _R0[:, int(np.argmin(_e0.real))]


def infidelity(hf, tau, E_tracked_hf, dt=DT):
    """1 - P_R(tau) at the end of the linear ramp h(t) = h_f t/tau, Eq. (93)."""
    _, e, R, Li, _ = frame(L, hf, GAM, J=J, eps=EPS)
    i0 = int(np.argmin(np.abs(e - E_tracked_hf)))
    psi = evolve2(_A, _B, lambda t: hf * min(t / tau, 1.0), tau, PSI0,
                  max(3000, int(tau / dt)))
    if psi is None:
        return 1.0
    c = Li @ psi
    return float(1.0 - abs(c[i0]) ** 2 / np.sum(abs(c) ** 2))


# ------------------------------------------------------------------ adaptive sweep
def sweep_hf(hf, E_hf, verbose=True):
    """Log-spaced base grid plus N_ROUNDS of midpoint refinement on steep segments."""
    taus = list(np.geomspace(TAU_MIN, TAU_MAX, N_BASE))
    vals = []
    t0 = time.time()
    for t in taus:
        vals.append(infidelity(hf, t, E_hf))
    if verbose:
        print(f"  h_f={hf}: base grid {len(taus)} pts, {time.time()-t0:.0f}s", flush=True)

    for r in range(N_ROUNDS):
        lv = np.log10(np.clip(vals, 1e-13, 2.0))
        new = []
        for i in range(len(taus) - 1):
            steep = abs(lv[i + 1] - lv[i]) > DLOG
            wide = taus[i + 1] / taus[i] > RATIO_MIN
            if steep and wide:
                new.append(np.sqrt(taus[i] * taus[i + 1]))
        if not new:
            break
        t0 = time.time()
        for t in new:
            taus.append(t)
            vals.append(infidelity(hf, t, E_hf))
        o = np.argsort(taus)
        taus = list(np.asarray(taus)[o])
        vals = list(np.asarray(vals)[o])
        if verbose:
            print(f"    round {r+1}: +{len(new)} pts -> {len(taus)} total, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return np.asarray(taus), np.asarray(vals)


def run_sweep(only=None):
    if os.path.exists("Idat.npz"):
        z = np.load("Idat.npz")
        Idat = {float(h): (float(i), complex(e)) for h, i, e in zip(z["hf"], z["Imax"], z["Ehf"])}
    else:
        Idat = max_I()
        np.savez("Idat.npz", hf=np.array(HFS), Imax=np.array([Idat[h][0] for h in HFS]),
                 Ehf=np.array([Idat[h][1] for h in HFS]))
    store = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    changed = False
    for hf in (HFS if only is None else [only]):
        Imax, E_hf = Idat[hf]
        if store.get(f"I{hf}") is None or float(store[f"I{hf}"][0]) != Imax:
            store[f"I{hf}"] = np.array([Imax])   # keep in sync with Idat.npz even if t/v are cached
            changed = True
        key = f"t{hf}"
        if key in store:
            print(f"  h_f={hf}: cached ({len(store[key])} pts)", flush=True)
            continue
        print(f"h_f={hf}  I(h_f) = {Imax:.5f}", flush=True)
        taus, vals = sweep_hf(hf, E_hf)
        store[key] = taus
        store[f"v{hf}"] = vals
        np.savez(CACHE, **store)          # save after each h_f: resumable
        changed = False
    if changed:
        np.savez(CACHE, **store)
    return store


if __name__ == "__main__":
    run_sweep()
    print("done")
