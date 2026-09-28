# data_cascade.jl -- odd-K Krylov-truncation scan of the AGP norm for the
# PT-symmetric Heisenberg chain (N=9, χ=1e-7, ξ ∈ [-0.9, -0.4], 60 points).
#
# For each ξ: exact ‖A_ξ‖² by full biorthogonal diagonalisation, plus the
# truncated-Krylov norm for every odd K in 3:2:499 (M = ⌊K/2⌋ = 1..249),
# all sliced from ONE Arnoldi build to depth K_MAX per ξ.
#
# Writes heisenberg_agp_cascade.csv (columns: xi, exact_norm_sq, K3, K5, ..., K499)
# row by row, flushed after every ξ, so an interrupted run keeps completed
# points. ~25 s per ξ point on one core (≈25 min total).
#
# Usage:
#     julia data_cascade.jl

include("core.jl")
using .HeisenbergCore

using Printf

const N        = 9
const CHI      = 1e-7
const XI_RANGE = (-0.9, -0.4)
const N_POINTS = 60
const K_MAX    = 500
const K_LIST   = collect(3:2:499)   # all odd K from 3 to 499 (249 values)
const OUTFILE  = "heisenberg_agp_cascade.csv"

xi_vals = collect(range(XI_RANGE[1], XI_RANGE[2], length=N_POINTS))

io = open(OUTFILE, "w")
println(io, "xi,exact_norm_sq," * join(["K$(k)" for k in K_LIST], ","))
flush(io)

println("Scanning N=$N, chi=$CHI, xi in $XI_RANGE, n_points=$N_POINTS, K_max=$K_MAX")
println("Odd K values: $(length(K_LIST)) points from $(K_LIST[1]) to $(K_LIST[end])")
println("Output: $OUTFILE")
println("-"^70)

t_start = time()
for (i, xi) in enumerate(xi_vals)
    t0 = time()

    H  = Matrix(build_hamiltonian(N, xi, CHI))
    dH = Matrix(dH_dxi(N, xi, CHI))
    dim_H = size(H, 1)

    exact_norm_sq = agp_norm_exact(N, xi, CHI).norm_sq

    basis_ops, h10, H_hess, init_norm = KrylovKit_Arnoldi_full(H, dH, K_MAX)
    k_actual = length(basis_ops)

    norms = Float64[]
    for K in K_LIST
        if K > k_actual
            push!(norms, NaN)
            continue
        end
        A = AGP_from_basis(basis_ops, H_hess, h10, K, init_norm, dim_H)
        push!(norms, sum(abs2, A))
    end

    println(io, @sprintf("%.10f,%.10e,", xi, exact_norm_sq) *
                join([@sprintf("%.10e", n) for n in norms], ","))
    flush(io)

    @printf("[%2d/%2d] xi=%+.4f  exact=%.4e  k_actual=%d  (%.1f s, elapsed %.1f min)\n",
            i, N_POINTS, xi, exact_norm_sq, k_actual, time() - t0, (time() - t_start)/60)
end

close(io)
println("-"^70)
println("Done. Wrote $OUTFILE")
