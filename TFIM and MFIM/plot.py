"""Unified plotting for the three MFIM figures -- one file, pick which to draw.

Each section below reads the .npz cache written by the matching data_*.py
script and renders it in the shared house style (figstyle.py); nothing here
runs any physics.

Usage
-----
    python plot.py collapse     # MFIM_collapse_Gamma.pdf   <- fig2a.npz      (data_collapse.py)
    python plot.py linear_cd    # MFIM_linear_ramp_CD.pdf   <- fig3.npz       (data_linear_cd.py)
    python plot.py optramp      # MFIM_CD_g5.pdf            <- optramp2.npz   (data_optramp.py)
    python plot.py all          # all three (default)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs

FLOOR = 1e-13                                    # clip for the log axis
FADE = 0.30
#: I(h_f) at h_f = 0.40 of Eq. (59) -- max_n mu_n(h), integrated
#: over h (see data_collapse.py:max_I). From data_collapse.py's Idat.npz;
#: this is only the fallback if that cache isn't present. A fixed constant of
#: the Hamiltonian path, so it just rescales tau -> Gamma, independent of
#: ramp shape or CD order.
I_MAX_DEFAULT = 0.3940260046190286
#: M = 1, 2, 3 sample viridis (perceptually uniform, colorblind-safe) at
#: well-separated stops; shared by the linear-ramp and optramp figures so
#: the same M maps to the same color in both.
_VIRIDIS = [plt.cm.viridis(x) for x in (0.90, 0.55, 0.15)]


def _cl(y):
    return np.clip(np.asarray(y, float), FLOOR, 2.0)


def _i_max(h_f=0.40):
    try:
        z = np.load("Idat.npz")
        return float(z["Imax"][np.argmin(np.abs(z["hf"] - h_f))])
    except FileNotFoundError:
        return I_MAX_DEFAULT


def half_crossing(tau, val, level=0.5):
    """Largest tau below which 1-P_R last crosses `level` upward; None if never."""
    v = np.asarray(val, float)
    above = np.where(v >= level)[0]
    if len(above) == 0 or above[0] == 0:
        return None
    i = above[0]
    x0, x1 = np.log(tau[i - 1]), np.log(tau[i])
    y0, y1 = v[i - 1], v[i]
    return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))


# ================================================================ collapse (data_collapse.py)
def _load_collapse(npz_file):
    d = np.load(npz_file)
    h_finals = sorted(float(k[1:]) for k in d.files if k.startswith("t"))
    return {hf: (d[f"t{hf}"], d[f"v{hf}"], float(d[f"I{hf}"][0])) for hf in h_finals}


def plot_collapse(npz_file="fig2a.npz", outfile="MFIM_collapse_Gamma.pdf"):
    series = _load_collapse(npz_file)
    print("points per curve:", {hf: len(v[0]) for hf, v in series.items()})

    fs.use_style()
    fig, ax = plt.subplots(figsize=fs.FIGSIZE1)
    for (hf, (tau, val, Imax)), c in zip(sorted(series.items()), fs.PALETTE):
        fs.series(ax, tau * Imax, _cl(val), color=c, label=rf"$h_{{\rm f}}={hf:.2f}$", ms=2.2)
    ax.set_ylim(3e-10, 3.0)
    fs.vline(ax, 10.0, ls="--")
    ax.set_xlabel(r"$\Gamma=\tau\,I(h_f)$")
    ax.set_ylabel(r"$1-P_{R}(\tau)$")
    ax.legend(loc="lower right", ncol=1, fontsize=8, handlelength=1.2,
              borderpad=0.3, labelspacing=0.2, handletextpad=0.4, markerscale=0.8)
    fs.save(fig, outfile)


# ================================================================ linear ramp CD (data_linear_cd.py)
#: "no CD" is drawn in red (fs.PALETTE[2]).
_LINEAR_STYLE = [("none", "no CD",  fs.PALETTE[2], "-"),
                  ("M1",   r"$M=1$", _VIRIDIS[0],   "-"),
                  ("M2",   r"$M=2$", _VIRIDIS[1],   "-"),
                  ("M3",   r"$M=3$", _VIRIDIS[2],   "-")]
TAU_REF = 40.0          # reference ramp time for the printed diagnostic


def _load_linear_cd(npz_file):
    d = np.load(npz_file)
    keys = [k[2:] for k in d.files if k.startswith("t_")]
    return {k: (d[f"t_{k}"], d[f"v_{k}"]) for k in keys}


def _at(tau, val, t0):
    """Infidelity at ramp time t0, by log-log interpolation."""
    return float(np.exp(np.interp(np.log(t0), np.log(tau), np.log(_cl(val)))))


def plot_linear_cd(npz_file="fig3.npz", outfile="MFIM_linear_ramp_CD.pdf"):
    series = _load_linear_cd(npz_file)
    I_max = _i_max()
    shown = {k: v for k, v in series.items() if k in [s[0] for s in _LINEAR_STYLE]}
    print("points per curve:", {k: len(v[0]) for k, v in sorted(shown.items())})
    print(f"1-P_R at tau={TAU_REF:g} (Gamma={TAU_REF*I_max:.1f}):",
          {k: f"{_at(*v, TAU_REF):.1e}" for k, v in sorted(shown.items())})

    fs.use_style()
    fig, ax = plt.subplots(figsize=fs.FIGSIZE1)
    for key, label, color, ls in _LINEAR_STYLE:
        if key not in series:
            continue
        tau, val = series[key]
        fs.series(ax, tau * I_max, _cl(val), color=color, label=label, ms=2.2)
    ax.set_ylim(1e-11, 4.0)
    ax.set_xlabel(r"$\Gamma=\tau\,I(h_f)$")
    ax.set_ylabel(r"$1-P_{R}(\tau)$")
    ax.legend(loc="lower right", fontsize=9, handlelength=1.4, borderpad=0.4,
              labelspacing=0.3, handletextpad=0.4)
    fs.save(fig, outfile)


# ================================================================ optimal ramp, Gamma_tol=5 (data_optramp.py)
_OPTRAMP_SPEC = [("noCD", fs.GRAY_CURVE, "opt. ramp"),
                  ("M1", _VIRIDIS[0], r"$M=1$"),
                  ("M2", _VIRIDIS[1], r"$M=2$"),
                  ("M3", _VIRIDIS[2], r"$M=3$")]


def _load_optramp(npz_file):
    d = np.load(npz_file)
    g3, g10, rr, dh = d["meta"]
    series = {k: d[k] for k in d.files if k not in ("tau", "meta")}
    return d["tau"], series, float(g3), float(g10), float(rr), float(dh)


def plot_optramp(npz_file="optramp2.npz", outfile="MFIM_CD_g5.pdf"):
    tau, series, g3, g10, rr, dh = _load_optramp(npz_file)
    g5 = 5.0
    I_max = _i_max()
    tau_c = half_crossing(tau, series["lin_noCD"])
    shown = ["lin_noCD"] + [f"g5_{tag}" for tag, _, _ in _OPTRAMP_SPEC]
    lo = min(np.nanmin(series[k]) for k in shown)

    print(f"control grid dh = {dh:g},  {len(tau)} ramp times,  I(h_f) = {I_max:.5f}")
    print(f"R = {rr:.4f}   t_broken = {rr*g5:.2f} ({g5:g})")
    if tau_c:
        print(f"linear ramp, half-infidelity crossing: tau_c = {tau_c:.1f} "
              f"(Gamma_c = {tau_c*I_max:.2f})")
    for k in sorted(k for k in series if k.startswith(("lin_", "g5_"))):
        print(f"  {k:10s} floor {np.nanmin(series[k]):.2e} "
              f"at tau={tau[int(np.nanargmin(series[k]))]:.1f}")

    fs.use_style()
    fig, ax = plt.subplots(1, 1, figsize=fs.figsize(1))
    fs.series(ax, tau, _cl(series["lin_noCD"]), color=fs.PALETTE[2], lw=0.9, ms=1.6,
              alpha=FADE, label="lin. ramp")
    for tag, col, lab in _OPTRAMP_SPEC:
        k = f"g5_{tag}"
        if k in series:
            fs.series(ax, tau, _cl(series[k]), color=col, label=lab, ms=2.4)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$1-P_{R}(\tau)$")
    ax.set_ylim(lo / 3, 40.0)

    handles = [Line2D([], [], color=fs.PALETTE[2], lw=0.9, marker="o", ms=1.6, mew=0,
                      alpha=FADE, label="lin. ramp")]
    handles += [Line2D([], [], color=col, lw=1.1, marker="o", ms=2.4, mew=0, label=lab)
                for _, col, lab in _OPTRAMP_SPEC]
    ax.legend(handles=handles, loc="upper left", ncol=3, columnspacing=0.9,
              handlelength=1.5, borderpad=0.35, labelspacing=0.22, handletextpad=0.45,
              fontsize=8.5)
    fs.save(fig, outfile)


# ================================================================ dispatch
FIGURES = {
    "collapse":  (plot_collapse,  "fig2a.npz",    "MFIM_collapse_Gamma.pdf"),
    "linear_cd": (plot_linear_cd, "fig3.npz",     "MFIM_linear_ramp_CD.pdf"),
    "optramp":   (plot_optramp,   "optramp2.npz", "MFIM_CD_g5.pdf"),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in FIGURES and which != "all":
        sys.exit(f"unknown figure {which!r}; choose from {list(FIGURES)} or 'all'")
    todo = FIGURES if which == "all" else {which: FIGURES[which]}
    for name, (draw, npz_file, outfile) in todo.items():
        print(f"--- {name} ---")
        draw(npz_file, outfile)
