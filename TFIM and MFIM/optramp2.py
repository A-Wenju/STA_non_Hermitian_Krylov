"""Krylov CD on the optimal ramp at two amplification budgets, with the
linear-ramp counterparts as a reference.

Curves: no CD, M = 1, 2, 3, each on three schedules -- the linear ramp and the
optimal ramp of Eq. (16) at Gamma_tol = 3 and 10. The control is built once per
M and reused across schedules. Writes optramp2.npz after every curve.
"""
import os, time
import numpy as np
from scipy.interpolate import interp1d

from core import H_of, dH_of, split, evolve2
from cd_agp import agp_krylov_gs
from brach import _fit_tau

L, GAM, J, EPS, HF = 8, 1.5, 1.0, 0.5, 0.40
DH = 0.001                       # control grid
GTOLS = {"g3": 3.0, "g5": 5.0, "g10": 10.0}
CACHE = "optramp2.npz"
TAUS = np.unique(np.round(np.geomspace(1.0, 400.0, 32), 4))

_m = np.load("metrics.npz")
_k = np.searchsorted(_m["hs"], HF) + 1
HS, MU, G = _m["hs"][:_k], _m["mu"][:_k], _m["g"][:_k]
BR = MU > 1e-12
ZB = np.trapezoid(np.sqrt(MU[BR] * G[BR]), HS[BR])
RRATIO = np.trapezoid(np.sqrt(G[BR] / MU[BR]), HS[BR]) / ZB

A_OP, B_OP = split(L, GAM, J, EPS)
DHM = dH_of(L, GAM, J, EPS).toarray()
HGRID = np.round(np.arange(0.0, HF + 0.5 * DH, DH), 10)
NG = len(HGRID) - 1


def frames():
    prev = None
    for h in HGRID:
        H = H_of(L, h, GAM, J, EPS).toarray()
        E, R = np.linalg.eig(H)
        R = R / np.linalg.norm(R, axis=0)
        Li = np.linalg.inv(R)
        i0 = int(np.argmin(E.real)) if prev is None else int(np.argmax(np.abs(Li @ prev)))
        prev = R[:, i0].copy()
        yield H, E, R, Li, i0


_H0 = H_of(L, 0.0, GAM, J, EPS).toarray()
_E, _R = np.linalg.eig(_H0)
PSI0 = _R[:, int(np.argmin(_E.real))].astype(complex)
PSI0 /= np.linalg.norm(PSI0)
_last = None
for _f in frames():
    _last = _f
LIF, I0 = _last[3], _last[4]


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
