# PT Heisenberg

Adiabatic gauge potential (AGP) norm of the PT-symmetric Heisenberg chain
(open chain, boundary fields h_1 = 1/(ξ + iχ), h_N = conj(h_1); N=9, D=512,
J=1, χ=1e-7 so H is non-Hermitian but its spectrum is real to machine
precision). Produces the AGP-norm figure `heisenberg_agp_cascade.pdf`
(Fig. 2 of the manuscript): exact ‖A_ξ‖² across the B2 → A2
transition (exceptional point ξ = -1/2), together with truncated Krylov
AGPs built from the first M = ⌊K/2⌋ odd-indexed Arnoldi basis operators,
M = 1, 2, 4, 8, 16, 32, 64, 124, 249. All scripts run from this directory
(`include` and cache filenames are relative).

## Layout

```
core.jl                 physics core: everything data_cascade.jl depends on (module HeisenbergCore):
                          model / Hamiltonian          embed_operator, embed_two_site, build_hamiltonian, dH_dxi
                          exact AGP norm               biorthogonal_eigensystem, agp_norm_exact_generic, agp_norm_exact
                          truncated Krylov AGP         KrylovKit_Arnoldi_full, AGP_from_basis

data_cascade.jl      -> heisenberg_agp_cascade.csv   exact norm + odd-K norms (K = 3, 5, ..., 499) at 60 ξ ∈ [-0.9, -0.4]

plot.jl              -> heisenberg_agp_cascade.pdf   (julia plot.jl; also writes a .png)
```

## Reproducing the figure from scratch

```
julia data_cascade.jl && julia plot.jl
```

`data_cascade.jl` writes the CSV cache row by row, one row per ξ. It takes
about 25 s per ξ point, so about 25 min in total. `plot.jl` only reads the
cache and runs no simulation, so you can iterate on figure styling cheaply by
rerunning `plot.jl`. The CSV keeps all 249 odd-K columns, so you can plot a
different set of truncation depths by editing `K_VALUES` in `plot.jl`.

Julia packages: `KrylovKit` (core), `Plots` (GR backend) + `LaTeXStrings` (plot);
`LinearAlgebra`, `SparseArrays`, `Printf`, `DelimitedFiles` are stdlib.
