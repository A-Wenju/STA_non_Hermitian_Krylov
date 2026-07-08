
module KrylovAGP_Scan

using LinearAlgebra
using SparseArrays
using Printf
using Base.Threads

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

function scan_krylov_agp_norm(N::Int, χ::Real;
                               ξ_range::Tuple{Real,Real} = (-0.9, -0.4),
                               n_points::Int             = 60,
                               krylov_dims::Vector{Int}  = [5,17,69,399],
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
    println("Using $(Threads.nthreads()) thread(s)")
    println("─"^68)

    for i in eachindex(ξ_vals)
        ξ = ξ_vals[i]

        # ── Hamiltonian and derivative ────────────────────────────────────────
        H  = Matrix(PT_Heisenberg.build_hamiltonian(N, ξ, χ))
        dH = Matrix(PT_Heisenberg.dH_dxi(N, ξ, χ))

        # ── Exact norm ────────────────────────────────────────────────────────
        res          = AGP_Norm.agp_norm_exact(N, ξ, χ; param=:xi, degen_tol=degen_tol)

    # for (i, ξ) in enumerate(ξ_vals)
    #     n_up = PT_Heisenberg.ground_state_n_up(N, ξ)   # ← NEW (one line)
    #     H  = Matrix(PT_Heisenberg.build_hamiltonian_sector(N, ξ, χ, n_up))  # ← changed
    #     dH = Matrix(PT_Heisenberg.dH_dxi_sector(N, ξ, χ, n_up))            # ← changed
    #     res = AGP_Norm.agp_norm_exact(N, ξ, χ;                              # ← add n_up=
    #             param=:xi, degen_tol=degen_tol, n_up=n_up)
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