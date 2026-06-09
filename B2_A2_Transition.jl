"""
B2_A2_Transition
================
Module for studying the B2 → A2 phase transition in the PT-symmetric
Heisenberg spin chain.

Reference: Kattel, Pasnoori & Andrei, J. Phys. A 56, 325001 (2023)

Usage (Jupyter notebook):
    include("PT_Heisenberg.jl")
    include("B2_A2_Transition.jl")
    using .B2_A2_Transition

    scan = scan_B2_to_A2(6, 0.3)
    print_scan_table(scan)
    detailed_report(6, -0.70, 0.3)

Phase recap (Tables 1 & 2 of the paper):
─────────────────────────────────────────────────────────────────────
  B2  (ξ < -1/2):  No boundary strings. ALL states have real energy.
                   Ground state: Sz = +1/2 (odd N), Sz = 0 or 1 (even N).
                   Built from all-spins-up reference state.

  A2  (-1/2 < ξ < 0):  Boundary strings exist → PT-broken sector appears.
                        BUT the ground state remains in the PT-unbroken
                        sector with real energy E0 throughout.
                        Ground state: Sz = +1/2 (odd N), Sz = 0 or 1 (even N).

  Transition at ξ = -1/2:  exceptional point. Boundary strings first
  appear. E0 is continuous across the transition but the wavefunction
  structure changes — this makes CD driving across this point non-trivial.

Strategy for CD driving:
  Fix χ, sweep ξ from B2 → A2 (e.g. -0.8 → -0.1).
  The instantaneous ground state is real and well-defined throughout.
  The AGP norm ‖A_ξ‖² will peak sharply near ξ = -1/2, signalling
  the exceptional point and the associated critical slowing-down.
"""
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

"""
    build_sz_operator(N) -> SparseMatrixCSC

Build the total Sz = (1/2) Σ_j σz_j operator as a diagonal sparse matrix
in the 2^N-dimensional computational basis.

Basis ordering convention (matching PT_Heisenberg):
  index 0  →  |↑↑...↑⟩  (all up),   Sz = +N/2
  index 2^N-1  →  |↓↓...↓⟩ (all down), Sz = -N/2
"""
function build_sz_operator(N::Int)
    dim  = 2^N
    # For basis index i (0-based), the number of up spins = popcount(i)
    vals = [0.5 * (count_ones(i) - (N - count_ones(i))) for i in 0:(dim-1)]
    return spdiagm(0 => ComplexF64.(vals))
end

# ─────────────────────────────────────────────────────────────────────────────
# Sector classification and ground state
# ─────────────────────────────────────────────────────────────────────────────

"""
    find_unbroken_ground_state(H; tol=1e-6)
        -> (E_unbroken, V_unbroken, E_broken)

Diagonalize H and split the spectrum into the PT-unbroken sector
(|Im(E)| < tol, real eigenvalues) and the PT-broken sector (complex
eigenvalues). Both are sorted by Re(E).

Returns
-------
E_unbroken  : Vector{Float64}  — real parts of unbroken eigenvalues, sorted
V_unbroken  : Matrix{ComplexF64} — corresponding right eigenvectors (columns)
E_broken    : Vector{ComplexF64} — complex eigenvalues of the broken sector
"""
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

"""
    ground_state_sz(H, Sz_op; tol=1e-6) -> (E_gs, ⟨Sz⟩)

Find the PT-unbroken ground state and return its energy and biorthogonal
expectation value of Sz.

For a non-Hermitian H the correct expectation value is the biorthogonal one:
    ⟨Sz⟩ = ⟨ψ_L | Sz | ψ_R⟩ / ⟨ψ_L | ψ_R⟩
where |ψ_R⟩ is the right eigenvector of H and ⟨ψ_L| is the left
eigenvector (= right eigenvector of H†, conjugate-transposed).

Returns (NaN, NaN) if no real eigenvalue exists within tolerance.
"""
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

"""
    spectral_gaps(H; tol=1e-6) -> (Δ_unbroken, Δ_to_broken)

Compute two spectral gaps relevant to the AGP norm:

  Δ_unbroken  : E1 - E0 within the PT-unbroken sector (both real).
                This controls transitions within the real sector.

  Δ_to_broken : Re(E_broken_min) - E0, the gap from the unbroken ground
                state to the lowest PT-broken (complex) state.
                Returns Inf in the B phases where no broken states exist.

Both gaps closing near ξ = -1/2 causes the AGP norm to diverge.
"""
function spectral_gaps(H::AbstractMatrix; tol::Real=1e-6)
    E_u, _, E_b = find_unbroken_ground_state(H; tol)

    Δ_unbroken  = length(E_u) >= 2 ? E_u[2] - E_u[1] : Inf
    Δ_to_broken = isempty(E_b) ? Inf : minimum(real.(E_b)) - E_u[1]

    return Δ_unbroken, Δ_to_broken
end

# ─────────────────────────────────────────────────────────────────────────────
# Full scan across B2 → A2
# ─────────────────────────────────────────────────────────────────────────────

"""
    scan_B2_to_A2(N, χ; ξ_range=(-0.9, -0.05), n_points=60, tol=1e-6)
        -> NamedTuple

Sweep ξ from deep in B2 through the exceptional point at ξ = -1/2 into A2,
recording at each point:
  - xi           : the ξ value
  - E0           : ground state energy (PT-unbroken sector)
  - Sz           : biorthogonal ⟨Sz⟩ of the ground state
  - n_broken     : number of PT-broken (complex) eigenvalues
  - gap_unbroken : E1 - E0 within the unbroken sector
  - gap_to_broken: Re(E_broken_min) - E0  (NaN in B2 where no broken states)
  - phase        : "B2" or "A2" string label

Parameters
----------
N        : chain length
χ        : fixed imaginary boundary parameter (kept constant during scan)
ξ_range  : (ξ_start, ξ_end), should straddle -1/2
n_points : number of ξ values in the scan
tol      : threshold for |Im(E)| to classify as unbroken

Example
-------
    scan = scan_B2_to_A2(6, 0.3; ξ_range=(-0.85, -0.05), n_points=40)
    print_scan_table(scan)
"""
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

"""
    print_scan_table(scan)

Pretty-print the NamedTuple returned by scan_B2_to_A2, with a horizontal
rule marking the B2/A2 transition point.
"""
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

"""
    detailed_report(N, ξ, χ; n_show=10, tol=1e-6)

Print a detailed spectrum at a single (ξ, χ) point: all eigenvalues
labelled by sector (unbroken/broken), the boundary fields, the phase,
and the biorthogonal ground state ⟨Sz⟩.
"""
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
