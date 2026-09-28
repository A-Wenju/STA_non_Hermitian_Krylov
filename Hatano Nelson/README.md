# Hatano Nelson

Krylov counterdiabatic driving of the interacting Hatano-Nelson model (Eq. 55;
open and periodic boundaries; N=10, N/2 particles, J=U=a=1, ramp h: 0 -> 0.1).
Produces the three panels of Fig. 1 of the manuscript. All scripts run from
this directory (imports and cache filenames are relative, not package paths).

Open boundaries use the exact AGP A_h = -i a X (Eq. 56). Periodic boundaries
use the ground-state-tailored truncated AGP of App. F, Eqs. (F1)-(F3), built on
the Arnoldi Krylov basis (`agp` mode 'gs_galerkin'); I(h_f) is the Eq. (59)
definition. The excess energy (Eq. 57) is measured against E_0(h=0) for OBC
and against the eigenvalue adiabatically continued to h_f for PBC.

## Layout

```
core.py               physics core -- everything the three data scripts depend on:
                        model / Hamiltonian          basis, model_operators, hamiltonian, dH
                        time evolution                evolve, evolve_obc_exact, evolve_obc_stepper, evolve_cd
                        spectral diagnostics          adiabatic_reference, imaginary_spread,
                                                       tracked_spectrum, condition17, tau_max
                        Krylov ground-state AGP       arnoldi, agp, agp_grid, residual_table  (PBC only;
                                                       OBC has the exact A_h = -i a X instead)
                        fast sparse propagator        sparse_ops, evolve_fast, excess_fast

figstyle.py            shared plot style (palette, rcParams, envelope, save)

data_obc_cd.py       -> HN_OBC_CD_fig.npz             Fig. 1 left: bare vs. exact-CD excess energy, OBC
data_pbc_excess.py   -> HN_PBC_Excess_Energy.npz      Fig. 1 middle: condition-(17) window + excess-energy ramps, PBC
data_pbc_cd.py       -> HN_PBC_CD.npz                 Fig. 1 right: excess energy at the oscillation maxima, bare vs.
                                                       truncated-Krylov ground-state-Galerkin AGP, PBC

plot.py               -> HN_OBC_CD_fig.pdf            (python plot.py obc_cd)
                       -> HN_PBC_Excess_Energy.pdf     (python plot.py pbc_excess)
                       -> HN_PBC_CD.pdf                (python plot.py pbc_cd)
                          python plot.py all draws all three (and prints the PBC-CD suppression table)
```

## Reproducing the three figures from scratch

```
python data_obc_cd.py       && python plot.py obc_cd        # HN_OBC_CD_fig.pdf
python data_pbc_excess.py   && python plot.py pbc_excess    # HN_PBC_Excess_Energy.pdf
python data_pbc_cd.py       && python plot.py pbc_cd        # HN_PBC_CD.pdf

# or, once every data_*.py has been run at least once:
python plot.py all
```

Each `data_*.py` script writes its own `.npz` cache; plotting reads the cache
and does no simulation, so figure styling can be iterated on cheaply via
`plot.py` alone. `data_pbc_cd.py` takes optional `h_f` and `tau_hi` CLI args
(default 0.10, 100.0) and uses a process pool (`os.cpu_count()` workers) for
both the oscillation-maxima scan and the per-M AGP evaluation.
