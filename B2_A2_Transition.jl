
module B2_A2_Transition

using LinearAlgebra
using SparseArrays
using Printf

# ── Import everything exported by PT_Heisenberg ───────────────────────────
# PT_Heisenberg must be included before this module in the notebook:
include("PT_Heisenberg.jl")
using .PT_Heisenberg

export build_sz_operator,
       find_unbroken_ground_state,
       ground_state_sz,
       spectral_gaps,
       scan_B2_to_A2,
       print_scan_table,
       detailed_report,
       recommend_parameters

# ─────────────────────────────────────────────────────────────────────────────
# Sz operator
# ─────────────────────────────────────────────────────────────────────────────

function build_sz_operator(N::Int)
    dim  = 2^N
    # For basis index i (0-based), the number of up spins = popcount(i)
    vals = [0.5 * (count_ones(i) - (N - count_ones(i))) for i in 0:(dim-1)]
    return spdiagm(0 => ComplexF64.(vals))
end

# ─────────────────────────────────────────────────────────────────────────────
# Sector classification and ground state
# ─────────────────────────────────────────────────────────────────────────────

function find_unbroken_ground_state(H::AbstractMatrix; tol::Real=1e-6)
    F            = eigen(Matrix(H))
    idx_unbroken = findall(e -> abs(imag(e)) < tol, F.values)
    idx_broken   = findall(e -> abs(imag(e)) >= tol, F.values)

    E_unbroken   = real.(F.values[idx_unbroken])
    V_unbroken   = F.vectors[:, idx_unbroken]
    E_broken     = F.values[idx_broken]

    perm         = sortperm(E_unbroken)
    return E_unbroken[perm], V_unbroken[:, perm], E_broken
end


function ground_state_sz(H::AbstractMatrix, Sz_op::AbstractMatrix; tol::Real=1e-6)
    FR = eigen(Matrix(H))
    FL = eigen(Matrix(H'))

    real_mask = findall(e -> abs(imag(e)) < tol, FR.values)
    isempty(real_mask) && return NaN, NaN

    # Ground state = lowest real eigenvalue
    gs_local = argmin(real.(FR.values[real_mask]))
    gs_idx   = real_mask[gs_local]
    E_gs     = real(FR.values[gs_idx])
    ψR       = FR.vectors[:, gs_idx]

    # Match the left eigenvector by closest eigenvalue
    lft_idx  = argmin(abs.(FL.values .- FR.values[gs_idx]))
    ψL       = conj.(FL.vectors[:, lft_idx])   # bra form

    norm_LR   = dot(ψL, ψR)
    Sz_expval = real(dot(ψL, Matrix(Sz_op) * ψR) / norm_LR)

    return E_gs, Sz_expval
end

# ─────────────────────────────────────────────────────────────────────────────
# Spectral gaps
# ─────────────────────────────────────────────────────────────────────────────


function spectral_gaps(H::AbstractMatrix; tol::Real=1e-6)
    E_u, _, E_b = find_unbroken_ground_state(H; tol)

    Δ_unbroken  = length(E_u) >= 2 ? E_u[2] - E_u[1] : Inf
    Δ_to_broken = isempty(E_b) ? Inf : minimum(real.(E_b)) - E_u[1]

    return Δ_unbroken, Δ_to_broken
end

# ─────────────────────────────────────────────────────────────────────────────
# Full scan across B2 → A2
# ─────────────────────────────────────────────────────────────────────────────

function scan_B2_to_A2(N::Int, χ::Real;
                        ξ_range::Tuple{Real,Real}=(-0.9, -0.05),
                        n_points::Int=60,
                        tol::Real=1e-6)

    ξ_vals        = range(ξ_range[1], ξ_range[2], length=n_points)
    Sz_op         = build_sz_operator(N)

    E0_arr        = zeros(Float64, n_points)
    Sz_arr        = zeros(Float64, n_points)
    n_broken_arr  = zeros(Int,     n_points)
    gap_unbroken  = zeros(Float64, n_points)
    gap_to_broken = fill(NaN,      n_points)
    phase_arr     = Vector{String}(undef, n_points)

    for (k, ξ) in enumerate(ξ_vals)
        H              = PT_Heisenberg.build_hamiltonian(N, ξ, χ)
        E_gs, Sz_val   = ground_state_sz(H, Sz_op; tol)
        Δu, Δb         = spectral_gaps(H; tol)
        _, _, E_b      = find_unbroken_ground_state(H; tol)

        E0_arr[k]      = E_gs
        Sz_arr[k]      = Sz_val
        n_broken_arr[k]= length(E_b)
        gap_unbroken[k]= Δu
        gap_to_broken[k]= isinf(Δb) ? NaN : Δb
        phase_arr[k]   = ξ < -0.5 ? "B2" : "A2"
    end

    return (
        xi            = collect(ξ_vals),
        E0            = E0_arr,
        Sz            = Sz_arr,
        n_broken      = n_broken_arr,
        gap_unbroken  = gap_unbroken,
        gap_to_broken = gap_to_broken,
        phase         = phase_arr,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Display utilities
# ─────────────────────────────────────────────────────────────────────────────

function print_scan_table(scan)
    header = "  ξ        Phase  |  E0               ⟨Sz⟩    n_broken" *
             "  Δ(unbroken)   Δ(→broken)"
    rule   = "  " * "─"^88
    println()
    println(header)
    println(rule)

    for i in eachindex(scan.xi)
        ξ  = scan.xi[i]
        ph = scan.phase[i]

        gap_b_str = isnan(scan.gap_to_broken[i]) ?
                    "        ∞" :
                    @sprintf("%+9.5f", scan.gap_to_broken[i])

        @printf("  %+6.3f   %-3s   |  %+12.6f    %+5.2f    %4d      %+9.5f   %s\n",
                ξ, ph,
                scan.E0[i],
                scan.Sz[i],
                scan.n_broken[i],
                scan.gap_unbroken[i],
                gap_b_str)

        # Print separator at the B2 → A2 boundary
        if i < length(scan.xi) && scan.phase[i] == "B2" && scan.phase[i+1] == "A2"
            println("  " * "─"^26 * "  ξ = −1/2  exceptional point  " * "─"^28)
        end
    end
    println()
end


function detailed_report(N::Int, ξ::Real, χ::Real;
                          n_show::Int=10, tol::Real=1e-6)
    H     = PT_Heisenberg.build_hamiltonian(N, ξ, χ)
    Sz_op = build_sz_operator(N)
    F     = eigen(Matrix(H))
    h1    = 1.0 / complex(ξ, χ)

    println("="^66)
    println("  Spectrum:  N=$N,  ξ=$ξ,  χ=$χ")
    @printf("  h1 = %+.6f %+.6fi,   hN = conj(h1)\n", real(h1), imag(h1))
    println("  Phase: $(PT_Heisenberg.phase_label(ξ))")
    println("  Showing lowest $n_show eigenvalues by Re(E):")
    println("  " * "─"^52)
    println("  Rank   Re(E)            Im(E)          Sector")
    println("  " * "─"^52)

    perm = sortperm(F.values, by=real)
    for (rank, idx) in enumerate(perm[1:min(n_show, length(perm))])
        E      = F.values[idx]
        sector = abs(imag(E)) < tol ? "unbroken" : "BROKEN"
        @printf("  %3d    %+12.7f   %+12.7f   %s\n",
                rank, real(E), imag(E), sector)
    end

    E_gs, Sz_val = ground_state_sz(H, Sz_op; tol)
    println("  " * "─"^52)
    @printf("  Ground state (unbroken):  E0 = %+.7f,   ⟨Sz⟩ = %+.4f\n",
            E_gs, Sz_val)
    println("="^66)
end

"""
    recommend_parameters(χ)

Print a table of recommended (ξ, h, γ) values spanning B2 → A2
at a given χ, useful for setting up CD driving protocols.
"""
function recommend_parameters(χ::Real)
    println("\nRecommended scan parameters for χ = $χ")
    println("  Phase   ξ           h = ξ/(ξ²+χ²)     γ = χ/(ξ²+χ²)")
    println("  " * "─"^60)
    for (ξ, label) in [(-0.80, "B2"),
                        (-0.65, "B2"),
                        (-0.55, "B2"),
                        (-0.50, "── exceptional point ──"),
                        (-0.40, "A2"),
                        (-0.25, "A2"),
                        (-0.10, "A2")]
        h, γ = PT_Heisenberg.xi_chi_to_h_gamma(ξ, χ)
        @printf("  %-26s  ξ = %+5.2f   h = %+7.4f   γ = %+7.4f\n",
                label, ξ, h, γ)
    end
    println()
end

function main(χ)
    N = 5        # chain length (keep small for exact diagonalization)
 
    println("\n" * "█"^65)
    println("  B2 → A2 phase transition study")
    println("  N = $N sites,  χ = $χ")
    println("█"^65)
 
    # ── 1. Parameter recommendations ──────────────────────────────────
    recommend_parameters(χ)
 
    # ── 2. Detailed report at three representative points ──────────────
    println("\n\n── Representative points ──")
    detailed_report(N, -0.70, χ)   # deep in B2
    detailed_report(N, -0.50, χ)   # at the exceptional point
    detailed_report(N, -0.20, χ)   # in A2
 
    # ── 3. Full scan across the transition ────────────────────────────
    println("\n\n── Full scan: B2 → A2  (N=$N, χ=$χ) ──")
    scan = scan_B2_to_A2(N, χ; ξ_range=(-0.85, -0.4), n_points=25)
    print_scan_table(scan)
 
    # ── 4. Check PT-unbroken ground state is continuous ───────────────
    println("\n\n── Continuity check of E0 across the transition ──")
    println("  (E0 should be smooth; any jump indicates a level crossing)")
    ξ_fine = range(-0.55, -0.45, length=11)
    println("  ξ           E0")
    for ξ in ξ_fine
        H = PT_Heisenberg.build_hamiltonian(N, ξ, χ)
        E_gs, _ = ground_state_sz(H, build_sz_operator(N))
        marker = abs(ξ + 0.5) < 0.01 ? "  ← ξ ≈ -1/2" : ""
        @printf("  %+7.4f     %+.8f%s\n", ξ, E_gs, marker)
    end
end

end  # module B2_A2_Transition
