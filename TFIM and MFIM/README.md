# TFIM and MFIM

Two non-Hermitian Ising figures: the closed-form AGP norm of the exactly
solvable non-Hermitian transverse-field Ising model (NH-TFIM, `AGP_NHTFIM.pdf`),
and the Krylov counterdiabatic-driving figures for the mixed-field Ising model
(MFIM, Fig. 4).

## NH-TFIM: AGP norm (`agp-norm-tfim.ipynb` -> `AGP_NHTFIM.pdf`)

Self-contained Julia notebook (Plots, LaTeXStrings; no local includes, no
data cache). It evaluates the closed-form norm of the Krylov AGP truncated at
dA = 50 terms, with J = 1:

    g = h sqrt(1 - gamma^2),   nu = ln(J/g)
    ||A||^2 = 1/(32 |g|^2) * sum_{k=1}^{dA} |sinh((dA+1-k) nu) / sinh((dA+1) nu)|^2

It plots this against gamma in [0, 2] for h = 0.5, 0.99, 1.01, 1.5, 10. The
dashed lines mark the PT point gamma = 1 and the h = 1.5 critical points
gamma_c = sqrt(1 -+ 1/h^2). `sinhratio` evaluates the sinh ratio stably for
complex nu.

## MFIM: Krylov CD driving (Fig. 4)

Krylov counterdiabatic driving on the non-Hermitian mixed-field Ising model
(Eq. 85; L=8, gamma=1.5, J=1, eps=0.5, PBC; ramp h: 0 -> h_f, spectrum real
for h <= h_c = 0.22). Produces the three panels of Fig. 4 of the manuscript.
All scripts run from this directory (imports and cache filenames are
relative, not package paths).

The truncated AGP is the ground-state-tailored one of App. F, Eqs. (F1)-(F3),
built on the Arnoldi Krylov basis; I(h_f) is the Eq. (59) definition; the
optimal ramp is Eq. (F8) with nu_1 = (Z_b/Gamma_tol)^2 and nu_2 fitted to tau.

## Layout

```
core.py           physics core -- everything the three data scripts depend on:
                    Hamiltonian / dynamics   H_of, dH_of, split, evolve2
                    branch tracking          frame, tracked_frames
                    Krylov ground-state AGP  arnoldi_ops, agp_exact_gs, agp_krylov_gs
                    optimal-ramp solver      _fit_tau, solve, schedule
figstyle.py       shared plot style (colors, save, series helpers)

data_collapse.py   -> fig2a.npz, Idat.npz    Fig. 4 left: linear-ramp collapse vs Gamma=tau*I(h_f)
data_linear_cd.py  -> fig3.npz               Fig. 4 middle: Krylov CD on the linear ramp
data_optramp.py    -> metrics.npz, optramp2.npz   Fig. 4 right: Krylov CD on the optimal ramp
                                                    (Gamma_tol = 3, 5, 10; figure uses 5); also builds
                                                    the mu(h)/g(h) drive metrics it needs

plot.py           -> MFIM_collapse_Gamma.pdf   (python plot.py collapse)
                  -> MFIM_linear_ramp_CD.pdf   (python plot.py linear_cd)
                  -> MFIM_CD_g5.pdf            (python plot.py optramp)
                     python plot.py all draws all three.
```

## Reproducing the three figures from scratch

```
python data_collapse.py  && python plot.py collapse      # MFIM_collapse_Gamma.pdf
python data_linear_cd.py && python plot.py linear_cd     # MFIM_linear_ramp_CD.pdf
python data_optramp.py   && python plot.py optramp       # MFIM_CD_g5.pdf

# or, once every data_*.py has been run at least once:
python plot.py all
```

Each `data_*.py` script caches to its own `.npz` and is resumable (re-running
skips whatever's already cached). `data_collapse.py` also writes `Idat.npz`
(I(h_f) per h_f), which `plot.py` reads (via `_i_max()`) for every figure's
x-axis, to convert tau to Gamma = tau * I(h_f); if that cache is missing,
`plot.py` falls back to a hardcoded value for h_f=0.40.
