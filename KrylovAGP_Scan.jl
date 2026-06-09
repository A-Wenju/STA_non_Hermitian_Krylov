"""
KrylovAGP_Scan
==============
Scans ξ across the B2 → A2 phase transition and computes the AGP
Frobenius norm squared using the Krylov-Arnoldi algorithm at several
finite Krylov dimensions, alongside the exact result from full
diagonalization for benchmarking.

Dependencies (included automatically):
    PT_Heisenberg.jl    — Hamiltonian, dH/dξ
    B2_A2_Transition.jl
    AGP_Norm.jl         — exact AGP norm via biorthogonal eigensystem
    KrylovKitAGP.jl     — Krylov-Arnoldi AGP

Usage (Jupyter notebook):
    include("KrylovAGP_Scan.jl")
    using .KrylovAGP_Scan

    results = scan_krylov_agp_norm(7, 1e-5)
    print_krylov_scan_table(results)

Notes on convergence
--------------------
The Krylov AGP at finite dimension k is NOT guaranteed to be a
variational lower bound on ‖A_exact‖²_F. It can overshoot or undershoot
depending on ξ and k, because the truncated Hessenberg system does not
correspond to a projection of the full operator equation. Convergence is
therefore non-monotone in general; the correct diagnostic is the relative
error |‖A_k‖² - ‖A_exact‖²| / ‖A_exact‖², which decreases toward zero
as k → dim(H)².
"""
module KrylovAGP_Scan

using LinearAlgebra
using SparseArrays
using Printf

# ── Load all dependencies ─────────────────────────────────────────────────────
include("PT_Heisenberg.jl")
include("B2_A2_Transition.jl")
include("AGP_Norm.jl")
include("KrylovKitAGP.jl")

using .PT_Heisenberg
using .AGP_Norm
using .KrylovKitAGP

export scan_krylov_agp_norm, print_krylov_scan_table

# ─────────────────────────────────────────────────────────────────────────────
# Single-point: Krylov AGP norm squared for several k_dims
# ─────────────────────────────────────────────────────────────────────────────

"""
    krylov_agp_norm_sq(H, dH, k_dims) -> Vector{Float64}

Compute ‖A_k‖²_F from KrylovKit_AGP_finite at each dimension in k_dims.
Each entry is computed independently by calling KrylovKit_AGP_finite
with the corresponding target_dim.

Returns a Vector of ‖A_k‖²_F values in the same order as k_dims.

Note: the Krylov norm can exceed the exact value at small k — this is
expected behaviour of the truncated Hessenberg solve, not a bug.
"""
function krylov_agp_norm_sq(H::Matrix{ComplexF64},
                             dH::Matrix{ComplexF64},
                             k_dims::Vector{Int})::Vector{Float64}

    norms_sq = zeros(Float64, length(k_dims))
    for i in eachindex(k_dims)
        A_k          = KrylovKitAGP.KrylovKit_AGP_finite(H, dH, k_dims[i])
        norms_sq[i]  = sum(abs2, A_k)
    end
    return norms_sq
end

# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────

"""
    scan_krylov_agp_norm(N, χ;
                         ξ_range     = (-0.9, -0.05),
                         n_points    = 60,
                         krylov_dims = [5, 10, 20, 40, 80],
                         degen_tol   = 1e-8)
        -> NamedTuple

Sweep ξ across the B2 → A2 transition at fixed χ and compute:
  • The exact AGP Frobenius norm squared (full diagonalization)
  • The Krylov AGP norm squared at each dimension in krylov_dims
  • The relative error |‖A_k‖² - ‖A_exact‖²| / ‖A_exact‖² at each k

Parameters
----------
N           : chain length (N=7 recommended for χ=1e-5)
χ           : imaginary boundary parameter (use small value, e.g. 1e-5)
ξ_range     : (ξ_start, ξ_end), should straddle -1/2
n_points    : number of ξ values in the scan
krylov_dims : sorted vector of Krylov dimensions ≤ (2^N)², keep ≤ 100
degen_tol   : near-degeneracy cutoff for exact AGP sum

Returns a NamedTuple:
  xi          : Vector{Float64}     ξ values
  exact       : Vector{Float64}     exact ‖A_ξ‖²_F
  krylov      : Dict{Int,Vector{Float64}}   Krylov ‖A_k‖²_F keyed by k
  rel_error   : Dict{Int,Vector{Float64}}   relative error keyed by k
  krylov_dims : Vector{Int}
  phase       : Vector{String}      "B2" or "A2"
  N           : Int
  chi         : Float64

Plotting example
----------------
    using Plots
    results = scan_krylov_agp_norm(7, 1e-5)

    # Norm comparison
    p1 = plot(xlabel="ξ", ylabel="‖A_ξ‖²_F", yscale=:log10, legend=:topleft,
              title="AGP norm: Krylov vs exact  (N=\$(results.N), χ=\$(results.chi))")
    for k in results.krylov_dims
        plot!(p1, results.xi, results.krylov[k], label="k=\$k", linestyle=:dash)
    end
    plot!(p1, results.xi, results.exact, label="Exact", lw=2, color=:black)
    vline!(p1, [-0.5], linestyle=:dot, color=:red, label="ξ=-1/2")

    # Relative error convergence
    p2 = plot(xlabel="ξ", ylabel="relative error", yscale=:log10, legend=:topleft,
              title="Krylov convergence")
    for k in results.krylov_dims
        plot!(p2, results.xi, results.rel_error[k], label="k=\$k")
    end
    vline!(p2, [-0.5], linestyle=:dot, color=:red, label="ξ=-1/2")
"""
function scan_krylov_agp_norm(N::Int, χ::Real;
                               ξ_range::Tuple{Real,Real} = (-0.9, -0.4),
                               n_points::Int             = 60,
                               krylov_dims::Vector{Int}  = [5,17,39,199],
                               degen_tol::Real           = 1e-8)

    @assert issorted(krylov_dims) "krylov_dims must be in ascending order"
    @assert maximum(krylov_dims) <= (2^N)^2 "Krylov dim exceeds Hilbert space squared"

    ξ_vals = collect(range(ξ_range[1], ξ_range[2], length=n_points))

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    exact_arr = zeros(Float64, n_points)
    phase_arr = Vector{String}(undef, n_points)

    # Build dicts with explicit loops to avoid generator-scoping issues
    krylov_arr   = Dict{Int, Vector{Float64}}()
    rel_err_arr  = Dict{Int, Vector{Float64}}()
    for k in krylov_dims
        krylov_arr[k]  = zeros(Float64, n_points)
        rel_err_arr[k] = zeros(Float64, n_points)
    end

    println("Scanning N=$N, χ=$χ,  ξ ∈ [$(ξ_range[1]), $(ξ_range[2])],  n_points=$n_points")
    println("Krylov dims: $krylov_dims")
    println("─"^68)

    for (i, ξ) in enumerate(ξ_vals)

        # ── Hamiltonian and derivative ────────────────────────────────────────
        H  = Matrix(PT_Heisenberg.build_hamiltonian(N, ξ, χ))
        dH = Matrix(PT_Heisenberg.dH_dxi(N, ξ, χ))

        # ── Exact norm ────────────────────────────────────────────────────────
        res          = AGP_Norm.agp_norm_exact(N, ξ, χ; param=:xi, degen_tol=degen_tol)
        exact_arr[i] = res.norm_sq
        phase_arr[i] = ξ < -0.5 ? "B2" : "A2"

        # ── Krylov norms at each requested dimension ──────────────────────────
        kn = krylov_agp_norm_sq(H, dH, krylov_dims)
        for (j, k) in enumerate(krylov_dims)
            krylov_arr[k][i]  = kn[j]
            rel_err_arr[k][i] = abs(kn[j] - exact_arr[i]) /
                                 max(exact_arr[i], eps(Float64))
        end

        @printf("  ξ=%+6.4f [%s]  exact=%.4e  k=%d: %.4e  (rel_err=%.2e)\n",
                ξ, phase_arr[i], exact_arr[i],
                krylov_dims[end], krylov_arr[krylov_dims[end]][i],
                rel_err_arr[krylov_dims[end]][i])
    end

    println("─"^68)
    println("Done.")

    return (
        xi          = ξ_vals,
        exact       = exact_arr,
        krylov      = krylov_arr,
        rel_error   = rel_err_arr,
        krylov_dims = krylov_dims,
        phase       = phase_arr,
        N           = N,
        chi         = Float64(χ),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

"""
    print_krylov_scan_table(results; n_show=20)

Print a formatted table showing exact and Krylov norms, and relative
errors, at evenly spaced ξ values across the scan.
"""
function print_krylov_scan_table(results; n_show::Int=20)
    n   = length(results.xi)
    idx = unique(round.(Int, range(1, n, length=min(n_show, n))))
    ks  = results.krylov_dims

    # ── Norm table ────────────────────────────────────────────────────────────
    header = "  ξ        Ph    Exact       " *
             join([@sprintf(" k=%-4d     ", k) for k in ks])
    println()
    println(header)
    println("  " * "─"^length(header))
    for i in idx
        row = @sprintf("  %+6.4f   %-2s    %.4e  ", results.xi[i], results.phase[i], results.exact[i])
        for k in ks
            row *= @sprintf(" %.4e  ", results.krylov[k][i])
        end
        println(row)
        if i < n && results.phase[i] == "B2" && results.phase[i+1] == "A2"
            println("  " * "─"^18 * "  ξ = −1/2  " * "─"^18)
        end
    end

    # ── Relative error table ──────────────────────────────────────────────────
    println()
    println("  Relative errors  |‖A_k‖² - ‖A_exact‖²| / ‖A_exact‖²")
    header2 = "  ξ        Ph    " * join([@sprintf(" k=%-4d     ", k) for k in ks])
    println(header2)
    println("  " * "─"^length(header2))
    for i in idx
        row = @sprintf("  %+6.4f   %-2s   ", results.xi[i], results.phase[i])
        for k in ks
            row *= @sprintf(" %.4e  ", results.rel_error[k][i])
        end
        println(row)
        if i < n && results.phase[i] == "B2" && results.phase[i+1] == "A2"
            println("  " * "─"^18 * "  ξ = −1/2  " * "─"^18)
        end
    end
    println()
end

end  # module KrylovAGP_Scan