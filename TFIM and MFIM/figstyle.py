"""
figstyle.py -- house figure standard for the STA / non-Hermitian Krylov manuscript.

Shared by every plot script in this directory so
that all figures use one definition of the style rather than a copy per script.

Typical use::

    import figstyle as fs

    fs.use_style()
    fig, ax = plt.subplots(figsize=fs.FIGSIZE1)
    fs.series(ax, tau, err, color=fs.PALETTE[0], label=r"$h_f=0.40$")
    ax.set_xlabel(r"$\\tau$"); ax.set_ylabel(r"$1-P_R(\\tau)$")
    ax.legend(loc="lower left")
    fs.save(fig, "fig.pdf")

Everything here is plain matplotlib; nothing is required at import time.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants

#: Qualitative series palette, in order of use.
PALETTE = ["#1f6fb4", "#2ca02c", "#e8746b", "#f2b134"]

#: Greys for reference curves, guides and annotations.
GRAY_CURVE, GRAY_GUIDE, GRAY_TEXT = "0.35", "0.55", "0.45"

#: Single-column and two-panel figure sizes, in inches.
FIGSIZE1 = (3.6, 2.75)
FIGSIZE2 = (7.0, 2.75)


def figsize(ncols=1, height=2.75):
    """Figure size for `ncols` panels side by side at the house panel width."""
    return (3.6 if ncols == 1 else 3.5 * ncols, height)


# ---------------------------------------------------------------- style

def use_style():
    """Apply the house rcParams. Call once, before creating the figure."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 12, "axes.linewidth": 0.9,
        "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 3.6, "ytick.major.size": 3.6,
        "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
        "axes.grid": False,
        "legend.frameon": True, "legend.framealpha": 1.0,
        "legend.edgecolor": "0.5", "legend.fancybox": False,
        "legend.fontsize": 10})


# ---------------------------------------------------------------- helpers

def series(ax, x, y, color, label=None, loglog=True,
           lw=1.1, ms=2.2, marker="o", alpha=1.0):
    """Default series style: a solid line with a small marker on every computed
    point.
    """
    draw = ax.loglog if loglog else ax.plot
    draw(x, y, marker + "-", color=color, lw=lw, ms=ms, mew=0,
         alpha=alpha, label=label)


def envelope(x, y, w=2, min_sep=0.0, logx=True):
    """Upper envelope of an oscillatory series: points that are the local
    maximum over a window of half-width `w`.

    Optional, and off by default -- an envelope hides where the samples
    actually are, so use it only when asked for, or when the notch structure
    is genuinely a sampling artefact that would mislead. `min_sep` thins the
    result so consecutive markers are at least that far apart along x, in
    decades if `logx`. Returns (x_env, y_env).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = [i for i in range(len(y))
            if y[i] >= y[max(0, i - w):min(len(y), i + w + 1)].max()]
    if min_sep > 0 and keep:
        pos = np.log10(x) if logx else x
        thinned, last = [keep[0]], pos[keep[0]]
        for i in keep[1:]:
            if pos[i] - last >= min_sep:
                thinned.append(i)
                last = pos[i]
        keep = thinned
    return x[keep], y[keep]


def oscillatory(ax, x, y, color, label=None, w=2, min_sep=0.0, loglog=True,
                lw=0.7, alpha=0.35, ms=4.5, marker="o"):
    """Faint raw line plus envelope markers. Opt-in; see `envelope`. Prefer
    `series` unless an envelope has been explicitly requested."""
    draw = ax.loglog if loglog else ax.plot
    draw(x, y, "-", color=color, lw=lw, alpha=alpha)
    xe, ye = envelope(x, y, w=w, min_sep=min_sep, logx=loglog)
    draw(xe, ye, marker, color=color, ms=ms, mew=0, label=label)
    return xe, ye


def panel_title(ax, letter, text, **kw):
    """Left-aligned panel title in the house format, e.g. ``(a) infidelity``."""
    ax.set_title(rf"({letter}) {text}", fontsize=13, loc="left", **kw)


def power_guide(ax, x0, x1, amp, p=-1.0, label=None, offset=1.6, npts=8):
    """Dashed power-law guide ``amp * x**p`` over [x0, x1], annotated in place.

    Guides are labelled on the axes rather than in the legend so the legend
    carries only data series.
    """
    g = np.geomspace(x0, x1, npts)
    ax.loglog(g, amp * g ** p, "k--", lw=1.1)
    if label:
        gm = g[len(g) // 2]
        ax.text(gm, amp * gm ** p * offset, label, fontsize=11,
                color="0.15", ha="center", va="bottom")


def vline(ax, x, color="0.55", label=None, y=None, ls=":", lw=1.0, alpha=1.0):
    """Dotted vertical marker (a threshold, a critical time), annotated in place."""
    ax.axvline(x, color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ylo, yhi = ax.get_ylim()
        ax.text(x * 1.05, y if y is not None else np.sqrt(ylo * yhi), label,
                fontsize=11, color=GRAY_TEXT)


def band(ax, x0, x1, label=None, y=None, color="0.87"):
    """Shaded vertical band marking an interval, e.g. a collapse threshold."""
    ax.axvspan(x0, x1, color=color, zorder=0)
    if label:
        ylo, yhi = ax.get_ylim()
        ax.text(np.sqrt(x0 * x1), y if y is not None else np.sqrt(ylo * yhi),
                label, fontsize=11, color=GRAY_TEXT, ha="center")


# ---------------------------------------------------------------- output

def save(fig, outfile, pad=0.4, dpi=400, png=True):
    """Tight-layout, then write the PDF (for LaTeX) and a matching PNG (for
    quick viewing). Returns the path written."""
    fig.tight_layout(pad=pad)
    fig.savefig(outfile, bbox_inches="tight")
    if png and outfile.endswith(".pdf"):
        fig.savefig(outfile[:-4] + ".png", dpi=dpi, bbox_inches="tight")
    print(f"written: {outfile}")
    return outfile
