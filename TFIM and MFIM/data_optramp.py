"""Data for MFIM_CD_g5.pdf (Fig. 4, right panel, of the manuscript): Krylov CD
on the optimal ramp of Eq. (F8), at
three amplification budgets Gamma_tol = 3, 5, 10, with the linear-ramp
counterparts as a reference.

Curves: no CD, M = 1, 2, 3, each on four schedules -- the linear ramp and the
optimal ramp at Gamma_tol = 3, 5, 10. The control is built once per M and
reused across schedules. Writes optramp2.npz after every curve.

Two stages, both cached:
  1. metrics.npz -- the drive metrics mu(h) (max instability rate) and g(h)
     (biorthogonal GS fidelity susceptibility) that the optimal-ramp solver
     (core._fit_tau) needs. 
  2. optramp2.npz -- the tau sweep for every (schedule, M) pair.

Usage
-----
    python data_optramp.py

Plotting lives in plot.py; nothing here draws.
"""
import os, time
import numpy as np
from scipy.interpolate import interp1d

from core import H_of, dH_of, split, evolve2, tracked_frames, agp_krylov_gs, _fit_tau

L, GAM, J, EPS, HF = 8, 1.5, 1.0, 0.5, 0.40
DH = 0.001                       # control grid
GTOLS = {"g3": 3.0, "g5": 5.0, "g10": 10.0}
CACHE = "optramp2.npz"
TAUS = np.unique(np.round(np.geomspace(1.0, 400.0, 32), 4))


# ------------------------------------------------------------------ drive metrics (was metrics.py)
def build_metrics(hmax=0.55, n=551, fn="metrics.npz"):
    """mu(h) = max_n Im[E_n(h)-E_0(h)] (Eq. 59) and g(h) (Eq. F5), the
    biorthogonal GS fidelity susceptibility, on a fixed h-grid. Feeds the
    optimal-ramp solver core._fit_tau."""
    if os.path.exists(fn):
        return dict(np.load(fn))
    hs = np.linspace(0.0, hmax, n)
    dH = dH_of(L, GAM, J, EPS).toarray()
    mu = np.zeros(n); g = np.zeros(n); imE0 = np.zeros(n); gap = np.zeros(n)
    for k, (H, E, R, Li, i0) in enumerate(tracked_frames(L, hs, GAM, J, EPS)):
        d = E - E[i0]
        mu[k] = np.max(d.imag)            # max over branches, Eq. (59)
        imE0[k] = E[i0].imag
        dd = d.copy(); dd[i0] = np.inf
        gap[k] = np.min(np.abs(dd))
        v = Li @ dH @ R[:, i0]
        c = v / dd; c[i0] = 0.0
        g[k] = np.sum(np.abs(c) ** 2)     # biorthogonal GS fidelity susceptibility
    d = dict(hs=hs, mu=mu, g=g, imE0=imE0, gap=gap)
    np.savez(fn, **d)
    return d


_m = build_metrics()
_k = np.searchsorted(_m["hs"], HF) + 1
HS, MU, G = _m["hs"][:_k], _m["mu"][:_k], _m["g"][:_k]
BR = MU > 1e-12
ZB = np.trapezoid(np.sqrt(MU[BR] * G[BR]), HS[BR])
RRATIO = np.trapezoid(np.sqrt(G[BR] / MU[BR]), HS[BR]) / ZB

A_OP, B_OP = split(L, GAM, J, EPS)
DHM = dH_of(L, GAM, J, EPS).toarray()
HGRID = np.round(np.arange(0.0, HF + 0.5 * DH, DH), 10)
NG = len(HGRID) - 1


# ------------------------------------------------------------------ frames / control
_frames_cache = None


def frames():
    global _frames_cache
    if _frames_cache is None:
        _frames_cache = list(tracked_frames(L, HGRID, GAM, J, EPS))
    return _frames_cache


_H0 = H_of(L, 0.0, GAM, J, EPS).toarray()
_E, _R = np.linalg.eig(_H0)
PSI0 = _R[:, int(np.argmin(_E.real))].astype(complex)
PSI0 /= np.linalg.norm(PSI0)
LIF, I0 = frames()[-1][3], frames()[-1][4]


def build_control(M):
    return [agp_krylov_gs(H, DHM, E, R, Li, i0, M)[0] for (H, E, R, Li, i0) in frames()]


def profile(tau, Gt):
    """Slowness u(h); Gt=None gives the linear ramp."""
    if Gt is None:
        return np.full_like(HS, tau / HF)
    return _fit_tau(HS, G, MU, (ZB / Gt) ** 2, float(tau))


def run(tau, Gt, control=None):
    u = profile(tau, Gt)
    t = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(HS))])
    t = t / t[-1] * tau
    h_of_t = interp1d(t, HS, bounds_error=False, fill_value=(HS[0], HS[-1]))
    hd_of_t = interp1d(t, 1.0 / u, bounds_error=False, fill_value=(1 / u[0], 0.0))
    acd = None
    if control is not None:
        def acd(tt):
            i = min(int(round(float(h_of_t(tt)) / DH)), NG)
            return float(hd_of_t(tt)) * control[i]
    psi = evolve2(A_OP, B_OP, lambda tt: float(h_of_t(tt)), tau, PSI0,
                  max(3000, int(tau / 0.01)), Acd=acd)
    if psi is None:
        return 1.0
    c = LIF @ psi
    return float(1 - abs(c[I0]) ** 2 / np.sum(abs(c) ** 2))


if __name__ == "__main__":
    d = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
    d["tau"] = TAUS
    d["meta"] = np.array([GTOLS["g3"], GTOLS["g10"], RRATIO, DH])
    np.savez(CACHE, **d)
    print(f"R = {RRATIO:.4f}   t_broken = {RRATIO*3:.2f} / {RRATIO*10:.2f}")

    for M, tag in ((None, "noCD"), (1, "M1"), (2, "M2"), (3, "M3")):
        keys = [f"{r}_{tag}" for r in ("lin", "g3", "g5", "g10")]
        if all(k in np.load(CACHE).files for k in keys):
            continue
        t0 = time.time()
        ctrl = None if M is None else build_control(M)
        for ramp, key in zip(("lin", "g3", "g5", "g10"), keys):
            if key in np.load(CACHE).files:
                continue
            Gt = None if ramp == "lin" else GTOLS[ramp]
            vals = np.array([run(tau, Gt, ctrl) for tau in TAUS])
            d = dict(np.load(CACHE)); d[key] = vals; np.savez(CACHE, **d)
            print(f"  {key:9s} min={np.nanmin(vals):.2e}", flush=True)
        del ctrl
        print(f"{tag:5s} block done in {time.time()-t0:5.0f}s", flush=True)
    print("done")
