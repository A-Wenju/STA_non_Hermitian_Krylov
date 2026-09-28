"""
Draw the three manuscript figures from their cached .npz files. Run the
matching data_*.py script first (or `python plot.py all` after all three
caches already exist) to (re)generate the data; plotting itself does no
simulation, so it is cheap to iterate on styling.

Usage:
    python plot.py obc_cd        -> HN_OBC_CD_fig.pdf       (reads data_obc_cd.py's cache)
    python plot.py pbc_excess    -> HN_PBC_Excess_Energy.pdf (reads data_pbc_excess.py's cache)
    python plot.py pbc_cd        -> HN_PBC_CD.pdf            (reads data_pbc_cd.py's cache)
    python plot.py all           -> all three
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import figstyle as fs

# ----------------------------------------------------------------------
# HN_OBC_CD_fig -- exact-CD excess energy, open boundaries
# ----------------------------------------------------------------------

def plot_obc_cd(npz="HN_OBC_CD_fig.npz", outfile="HN_OBC_CD_fig.pdf"):
    d = np.load(npz)
    taus, E_bare, E_cd = d["taus"], d["E_bare"], d["E_cd"]

    # Normalisation: the maximum is taken over |E_bare|, so both the real and
    # imaginary parts lie in [-1, 1]. To make a panel in which the real and
    # imaginary parts each individually touch +-1, normalise them separately:
    #     norm_re = np.abs(E_bare.real).max(); norm_im = np.abs(E_bare.imag).max()
    norm = np.abs(E_bare).max()

    fs.use_style()
    colors = fs.PALETTE
    styles = ["-o", "-o", "-o", "--s"]
    labels = ["without CD, Real", "without CD, Imag",
              "with CD, Real", "with CD, Imag"]
    series = [E_bare.real / norm, E_bare.imag / norm,
              E_cd.real / norm,   E_cd.imag / norm]
    # the two "with CD" curves both sit flat at ~0 and exactly overlap; give
    # them different markers/linewidths and offset which points get a marker
    # so both colors stay visible instead of one fully hiding the other
    linewidths = [1.2, 1.2, 2.2, 1.4]
    markersizes = [3.2, 3.2, 3.6, 3.2]
    markevery = [None, None, (0, 2), (1, 2)]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for y, c, ls, lab, lw, ms, me in zip(series, colors, styles, labels,
                                          linewidths, markersizes, markevery):
        ax.plot(taus, y, ls, color=c, label=lab, linewidth=lw,
                markersize=ms, markeredgewidth=0, markevery=me)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$E(\tau)\,/\,\max|E(\tau)|$")
    ax.set_xlim(0, taus[-1] * 1.02)
    ax.set_ylim(-1.08, 1.12)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax.legend(loc="upper right", fontsize=8.5, handlelength=1.6,
              borderpad=0.35, labelspacing=0.25, handletextpad=0.45)

    fs.save(fig, outfile, pad=0.3)


# ----------------------------------------------------------------------
# HN_PBC_Excess_Energy -- excess energy inside and past the condition-(17) window
# ----------------------------------------------------------------------

def _load_pbc_excess(npz):
    d = np.load(npz)
    h_finals = sorted({float(k[1:]) for k in d.files if k.startswith("s")})
    series = {hf: (d[f"t{hf}"], d[f"s{hf}"]) for hf in h_finals}
    hs, I, tm = d["hs"], d["I"], d["tm"]

    def tau_max_at(hf):
        k = np.argmin(np.abs(hs - hf))
        return tm[k]

    series = {hf: (t, tau_max_at(hf), s) for hf, (t, s) in series.items()}
    return hs, I, tm, float(d["h_c"]), series, d["t_ref"], d["E_obc"]


def _draw_excess_panel(ax, series, t_ref, E_obc, hf0, cols, title=True,
                       vline="tau_max", tau_label=True):
    for hf, c in zip(series, cols):
        t, tmax, E = series[hf]
        ax.loglog(t, np.abs(E), "-", color=c, lw=0.7, alpha=0.35)
        te, ye = fs.envelope(t, np.abs(E), w=2)
        label = rf"PBC, $h_{{\rm f}}={hf}$"
        if tau_label:
            label += rf" ($\tau_{{\max}}={tmax:.0f}$)"
        ax.loglog(te, ye, "o", color=c, ms=4.5, mew=0, label=label)
        if vline == "tau_max":
            vpos = tmax
        elif vline == "inflection":
            vpos = te[np.argmin(ye)]     # last envelope point before decay turns up
        else:
            vpos = None
        if vpos is not None and np.isfinite(vpos) and vpos <= t[-1]:
            ax.axvline(vpos, color=c, ls=":", lw=1.0, alpha=0.6)
    ax.loglog(t_ref, np.abs(E_obc), "-", color="0.45", lw=0.7, alpha=0.5)
    te, ye = fs.envelope(t_ref, np.abs(E_obc), w=2)
    ax.loglog(te, ye, "s", color="0.35", ms=4.5, mew=0,
              label=rf"OBC, $h_{{\rm f}}={hf0}$")
    # tau^-1 guide, amplitude anchored to the OBC curve at its left edge
    g0 = max(5.0 * t_ref[0], 0.15 * t_ref[-1])
    g1 = min(t_ref[-1], g0 * 10)
    amp = np.abs(E_obc[np.argmin(np.abs(t_ref - g0))]) * g0 * 7.0
    fs.power_guide(ax, g0, g1, amp, p=-1.0, label=None)
    # place the label at the guide's geometric centre, lifted well above the
    # envelope: the OBC markers sit right on the guide line by construction
    # (it approximates their envelope), and horizontal gaps between them
    # shrink and vary with how far the sweep extends, so dodging sideways is
    # fragile -- a large vertical lift clears any marker regardless of tau range
    gx = np.sqrt(g0 * g1)
    ax.text(gx, amp * gx ** -1.0 * 1.7, r"$\tau^{-1}$", fontsize=11,
            color="0.15", ha="center", va="bottom")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$|E(\tau)|$")
    ax.legend(loc="lower left", fontsize=9, handlelength=1.6, borderpad=0.35,
              labelspacing=0.22, handletextpad=0.45)
    if title:
        fs.panel_title(ax, "b", r"excess energy (dotted: $\tau_{\max}$)")


def plot_pbc_excess(npz="HN_PBC_Excess_Energy.npz", outfile="HN_PBC_Excess_Energy.pdf"):
    """Single-axes excess-energy figure."""
    _, _, _, _, series, t_ref, E_obc = _load_pbc_excess(npz)
    hf0 = sorted(series)[-1]
    fs.use_style()
    fig, ax = plt.subplots(figsize=fs.FIGSIZE1)
    _draw_excess_panel(ax, series, t_ref, E_obc, hf0, fs.PALETTE,
                       title=False, vline="inflection", tau_label=False)
    fs.save(fig, outfile)


# ----------------------------------------------------------------------
# HN_PBC_CD -- excess energy at the oscillation maxima, bare vs Krylov AGP
# ----------------------------------------------------------------------

# condition-(17) scale and measured bare turn-around at h_f = 0.1 (N = 10, PBC)
TAU_MAX = {0.10: 20.5}       # Gamma = 1, I(h_f) of Eq. (59)
TAU_STAR = {0.10: 48.2}      # measured envelope minimum

PLOT_M = [1, 3, 5]           # subset of galerkin_M to draw (M=7 dropped)

GALERKIN_COLORS = {1: "#bfe08a", 3: "#5cb85c", 5: "#146b34"}


def _marks(ax, h_f, y):
    t = TAU_STAR.get(round(h_f, 3))
    if t is None:
        return
    ax.axvline(t, color="0.6", ls="--", lw=1.1)
    ax.annotate(r"$\tau^\star$", xy=(t, y), xytext=(t * 1.06, y), fontsize=10,
                color="0.35")


def plot_pbc_cd(npz="HN_PBC_CD.npz", outfile="HN_PBC_CD.pdf"):
    d = np.load(npz, allow_pickle=True)
    taus = d["taus"]
    h_f = float(d["h_f"][0]) if "h_f" in d.files else 0.10
    no_cd = np.abs(d["no_CD"])
    fs.use_style()
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.75))

    ax.loglog(taus, no_cd, "-s", color=fs.PALETTE[0], lw=1.2, ms=4.5, mew=0, label="no CD")

    M_vals = d["galerkin_M"]
    E_gal = d["galerkin_E"]
    for M, E in zip(M_vals, E_gal):
        if int(M) not in PLOT_M:
            continue
        ax.loglog(taus, np.abs(E), "--o", color=GALERKIN_COLORS[int(M)],
                  lw=1.15, ms=4.0, mew=0, label=rf"$M={M}$")

    if "exact_E" in d.files:
        exact = np.abs(d["exact_E"])
        ax.loglog(taus, exact, "-d", color="#e8746b", lw=1.2, ms=4.5, mew=0,
                  label="exact GS AGP")

    ax.set_xlim(taus[0] * 0.9, taus[-1] * 1.15)
    _marks(ax, h_f, 2e-9)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$|E(\tau)|$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
              fontsize=8.5, handlelength=1.4, borderpad=0.3,
              labelspacing=0.2, handletextpad=0.35, columnspacing=0.8)

    fs.save(fig, outfile)


def table_pbc_cd(npz="HN_PBC_CD.npz"):
    """Print median suppression factors relative to the bare ramp -- a text
    diagnostic, not a figure."""
    d = np.load(npz, allow_pickle=True)
    h_f = float(d["h_f"][0]) if "h_f" in d.files else 0.10
    taus, base = d["taus"], np.abs(d["no_CD"])
    M_vals = d["galerkin_M"]
    E_gal = np.abs(d["galerkin_E"])

    print(f"{'tau':>7}" + "".join(f"{'M='+str(M):>14}" for M in M_vals))
    for i, t in enumerate(taus):
        row = f"{t:7.1f}"
        for j in range(len(M_vals)):
            row += f"{E_gal[j, i]:14.3e}"
        print(row)
    print()

    suppression = base[None, :] / E_gal
    print(f"{'M':>4} {'median suppression':>19} {'at tau_max':>11} "
          f"{'at tau*':>9}")
    i_m = int(np.argmin(np.abs(taus - TAU_MAX.get(round(h_f, 3), taus[-1]))))
    i_s = int(np.argmin(np.abs(taus - TAU_STAR.get(round(h_f, 3), taus[-1]))))
    for j, M in enumerate(M_vals):
        s = suppression[j]
        print(f"{M:4d} {np.median(s):18.1f}x {s[i_m]:10.1f}x {s[i_s]:8.1f}x")

    if "exact_E" in d.files:
        exact = np.abs(d["exact_E"])
        s = base / exact
        print(f"{'exact':>4} {np.median(s):18.1f}x {s[i_m]:10.1f}x "
              f"{s[i_s]:8.1f}x")


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

FIGURES = {
    "obc_cd": plot_obc_cd,
    "pbc_excess": plot_pbc_excess,
    "pbc_cd": plot_pbc_cd,
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for name, fn in FIGURES.items():
            fn()
        table_pbc_cd()
    elif which in FIGURES:
        FIGURES[which]()
        if which == "pbc_cd":
            table_pbc_cd()
    else:
        print(f"unknown figure '{which}'; choose from {list(FIGURES)} or 'all'")
        sys.exit(1)


if __name__ == "__main__":
    main()
