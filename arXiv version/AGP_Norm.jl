"""
AGP_Norm
========
Exact adiabatic gauge potential (AGP) norm for the PT-symmetric
Heisenberg spin chain via full diagonalization.

References:
  - Kolodrubetz, Sels, Mehta, Polkovnikov, Phys. Rep. 697, 1 (2017)
  - Pandey, Claeys, Campbell, Polkovnikov, Bhattacharjee, PRX 10, 041017 (2020)

Usage (Jupyter notebook):
    include("PT_Heisenberg.jl")
    include("B2_A2_Transition.jl")
    include("AGP_Norm.jl")
    using .PT_Heisenberg, .B2_A2_Transition, .AGP_Norm

    result = agp_norm_exact(6, -0.3, 0.3)
    scan   = scan_agp_norm(6, 0.3; n_points=80)

Theory
------
For H(λ) with biorthogonal eigenbasis {|n_R⟩, |n_L⟩} satisfying
    H|n_R⟩ = E_n|n_R⟩,   ⟨n_L|H = E_n⟨n_L|,   ⟨m_L|n_R⟩ = δ_mn,

the exact AGP operator is:
    A_λ = i Σ_{n≠m} |m_R⟩ ⟨m_L|∂_λH|n_R⟩⟨n_L| / (E_n - E_m)

Its Frobenius norm is:
    ‖A_λ‖²_F = Tr(A_λ† A_λ)
             = Σ_{m≠n} |⟨m_L|∂_λH|n_R⟩|² / |E_n - E_m|²

This sums over ALL off-diagonal pairs, not just from the ground state.
This is the correct extensive quantity that:
  • Scales with system size N (making it the right object for finite-size scaling)
  • Diverges at the exceptional point ξ = -1/2 for any N
  • Reduces to the standard AGP Frobenius norm when H is Hermitian
  • Is the correct benchmark target for approximate AGP algorithms
    (nested commutators, variational, etc.)

Driving parameter choice:
  param = :xi  → λ = ξ,  ∂_λH uses dH_dxi  (drives across B2/A2 transition)
  param = :chi → λ = χ,  ∂_λH uses dH_dchi
"""
module AGP_Norm

using LinearAlgebra
using SparseArrays
using Printf

include("PT_Heisenberg.jl")
include("B2_A2_Transition.jl")
using .PT_Heisenberg

export agp_norm_exact,
       agp_matrix_exact,
       scan_agp_norm,
       print_agp_scan_table

# ─────────────────────────────────────────────────────────────────────────────
# Biorthogonal eigensystem
# ─────────────────────────────────────────────────────────────────────────────

"""
    biorthogonal_eigensystem(H) -> (E, VR, VL)

Compute the full biorthogonal eigensystem of a non-Hermitian matrix H.

Algorithm:
  A single eigen call gives H = VR * diag(E) * VR⁻¹, so the left
  eigenvectors (as kets) are exactly the rows of VR⁻¹, conjugated:

      |n_L⟩  ←→  row n of VR⁻¹  (stored as column n of VL below)

  i.e. VL = conj((VR⁻¹)ᵀ) = conj(inv(VR))ᵀ

  Biorthonormality ⟨m_L|n_R⟩ = (VR⁻¹ * VR)[m,n] = δ_mn is then
  exact to machine precision by construction — no matching, no
  rescaling, no division by small overlaps.

  This is robust for any χ including χ → 0 (nearly Hermitian) and
  for large N, because it avoids:
    • Two separate eigen calls with independent eigenvalue orderings
    • Greedy closest-eigenvalue matching that fails near degeneracies
    • Division by ⟨n_L|n_R⟩ overlaps that vanish near exceptional points

  The only failure mode is exact non-diagonalisability (H defective),
  which occurs only precisely at the exceptional point itself.

Returns
-------
E   : Vector{ComplexF64}  eigenvalues
VR  : Matrix{ComplexF64}  right eigenvectors as columns
VL  : Matrix{ComplexF64}  left eigenvectors as columns (ket form |n_L⟩),
                          satisfying conj(VL[:,m])' * VR[:,n] = δ_mn
"""
function biorthogonal_eigensystem(H::AbstractMatrix)
    F  = eigen(Matrix(H))
    E  = F.values
    VR = F.vectors

    # Left eigenvectors: rows of VR⁻¹, stored as columns of VL
    # conj(VL[:,n])' * VR[:,m] = (VR⁻¹)[n,:] * VR[:,m] = δ_nm  ✓
    VL = conj.(inv(VR)')   # VL[:,n] = conj of n-th column of (VR⁻¹)ᵀ

    return E, VR, VL
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact AGP matrix
# ─────────────────────────────────────────────────────────────────────────────

"""
    agp_matrix_exact(N, ξ, χ; param=:xi, degen_tol=1e-8) -> Matrix{ComplexF64}

Construct the full 2^N × 2^N AGP operator:

    A_λ = i Σ_{n≠m} |m_R⟩ ⟨m_L|∂_λH|n_R⟩⟨n_L| / (E_n - E_m)

Parameters
----------
N         : chain length
ξ, χ      : boundary parameters
param     : :xi or :chi — which parameter λ to differentiate w.r.t.
degen_tol : pairs with |E_n - E_m| < degen_tol are skipped (near-degenerate)

Returns the full dense AGP matrix. Use for:
  • Direct Frobenius norm: norm(A, 2) or sqrt(sum(abs2, A))
  • Benchmarking approximate AGP algorithms (nested commutators, etc.)
  • Checking anti-Hermiticity: A_λ should satisfy A_λ† = -A_λ
    when H is Hermitian; for non-Hermitian H this no longer holds exactly.

Note: O(4^N) memory. Practical up to N ≈ 12.
"""
function agp_matrix_exact(N::Int, ξ::Real, χ::Real;
                           param::Symbol    = :xi,
                           degen_tol::Real  = 1e-8)

    H  = Matrix(PT_Heisenberg.build_hamiltonian(N, ξ, χ))
    dH = _get_dH(N, ξ, χ, param)

    E, VR, VL = biorthogonal_eigensystem(H)
    dim = length(E)

    A = zeros(ComplexF64, dim, dim)

    # Pre-compute dH * VR once: columns are dH|n_R⟩
    dH_VR = dH * VR          # dim × dim matrix; column n = ∂H|n_R⟩

    # conj(VL)' = bra matrix; row m = ⟨m_L|
    braL  = conj.(VL)'       # dim × dim; braL[m,:] = ⟨m_L|

    # mel[m,n] = ⟨m_L|∂H|n_R⟩  — all matrix elements at once
    mel   = braL * dH_VR     # dim × dim

    for n in 1:dim
        for m in 1:dim
            m == n && continue
            ΔE = E[n] - E[m]
            abs(ΔE) < degen_tol && continue

            # A += (i * mel[m,n] / ΔE) * |m_R⟩⟨n_L|
            # Outer product: VR[:,m] * conj(VL[:,n])'
            coeff = im * mel[m, n] / ΔE
            @inbounds for r in 1:dim, c in 1:dim
                A[r, c] += coeff * VR[r, m] * conj(VL[c, n])
            end
        end
    end

    return A
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact AGP Frobenius norm  (efficient: no need to build full matrix)
# ─────────────────────────────────────────────────────────────────────────────

"""
    agp_norm_exact(N, ξ, χ; param=:xi, degen_tol=1e-8) -> NamedTuple

Compute the exact AGP Frobenius norm squared at a single point (ξ, χ):

    ‖A_λ‖²_F = Σ_{m≠n} |⟨m_L|∂_λH|n_R⟩|² / |E_n - E_m|²

This is computed directly from the eigendecomposition without building
the full AGP matrix, making it significantly faster and more memory
efficient for large N.

Parameters
----------
N         : chain length
ξ, χ      : boundary parameters
param     : :xi (default) or :chi
degen_tol : pairs with |E_n - E_m| < degen_tol are excluded from the sum

Returns a NamedTuple with fields:
  norm_sq          : ‖A_λ‖²_F  (real, non-negative)
  n_pairs          : number of (m,n) pairs included in the sum
  n_degen_skipped  : number of near-degenerate pairs excluded
  min_gap          : minimum |E_n - E_m| over all included pairs
  E                : full eigenvalue vector (for diagnostics)
  phase            : phase label string
"""
# function agp_norm_exact(N::Int, ξ::Real, χ::Real;
#                         param::Symbol   = :xi,
#                         degen_tol::Real = 1e-8)

#     H  = Matrix(PT_Heisenberg.build_hamiltonian(N, ξ, χ))
#     dH = _get_dH(N, ξ, χ, param)

  function agp_norm_exact(N::Int, ξ::Real, χ::Real;
                          param::Symbol        = :xi,
                          degen_tol::Real      = 1e-8,
                          n_up::Union{Int,Nothing} = nothing)

      if isnothing(n_up)
          H  = Matrix(PT_Heisenberg.build_hamiltonian(N, ξ, χ))
          dH = _get_dH(N, ξ, χ, param)
      else
          H  = Matrix(PT_Heisenberg.build_hamiltonian_sector(N, ξ, χ, n_up))
          dH = _get_dH_sector(N, ξ, χ, param, n_up)
      end

    E, VR, VL = biorthogonal_eigensystem(H)
    dim = length(E)

    # All matrix elements at once: mel[m,n] = ⟨m_L|∂H|n_R⟩
    braL  = conj.(VL)'      # braL[m,:] = ⟨m_L|
    dH_VR = dH * VR         # dH_VR[:,n] = ∂H|n_R⟩
    mel   = braL * dH_VR    # mel[m,n] = ⟨m_L|∂H|n_R⟩

    norm_sq         = 0.0
    n_pairs         = 0
    n_degen_skipped = 0
    min_gap         = Inf

    for n in 1:dim
        for m in 1:dim
            m == n && continue
            ΔE  = E[n] - E[m]
            gap = abs(ΔE)
            if gap < degen_tol
                n_degen_skipped += 1
                continue
            end
            min_gap  = min(min_gap, gap)
            norm_sq += abs2(mel[m, n]) / abs2(ΔE)
            n_pairs += 1
        end
    end

    return (
        norm_sq         = norm_sq,
        n_pairs         = n_pairs,
        n_degen_skipped = n_degen_skipped,
        min_gap         = min_gap,
        E               = E,
        phase           = PT_Heisenberg.phase_label(ξ),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Scan across B2 → A2
# ─────────────────────────────────────────────────────────────────────────────

"""
    scan_agp_norm(N, χ; ξ_range=(-0.9, -0.05), n_points=80,
                  param=:xi, degen_tol=1e-8)
        -> NamedTuple

Sweep ξ across the B2 → A2 exceptional point at ξ = -1/2 and compute
‖A_ξ‖²_F at each point. This is the main function for studying the AGP
norm as a detector of the PT phase transition.

Returns a NamedTuple with vectors:
  xi             : ξ values
  norm_sq        : ‖A_λ‖²_F at each ξ
  min_gap        : minimum |E_n - E_m| at each ξ
  n_degen_skipped: degenerate pairs skipped at each ξ
  phase          : phase label at each ξ ("B2" or "A2")

Example
-------
    scan = scan_agp_norm(6, 0.3; n_points=80)
    print_agp_scan_table(scan)

    # Plotting in a Jupyter notebook (requires Plots.jl):
    using Plots
    plot(scan.xi, scan.norm_sq,
         xlabel = "ξ",
         ylabel = "‖A_ξ‖²_F",
         title  = "AGP norm across B2 → A2  (N=6, χ=0.3)",
         label  = "N=6",
         marker = :circle,
         yscale = :log10)
    vline!([-0.5], linestyle=:dash, color=:red, label="ξ = −1/2 (EP)")
"""
function scan_agp_norm(N::Int, χ::Real;
                       ξ_range::Tuple{Real,Real} = (-0.9, -0.05),
                       n_points::Int             = 80,
                       param::Symbol             = :xi,
                       degen_tol::Real           = 1e-8)

    ξ_vals          = collect(range(ξ_range[1], ξ_range[2], length=n_points))
    norm_sq_arr     = zeros(Float64, n_points)
    min_gap_arr     = zeros(Float64, n_points)
    n_degen_arr     = zeros(Int,     n_points)
    phase_arr       = Vector{String}(undef, n_points)

    for (k, ξ) in enumerate(ξ_vals)
        res = agp_norm_exact(N, ξ, χ; param=param, degen_tol=degen_tol)

        norm_sq_arr[k]  = res.norm_sq
        min_gap_arr[k]  = res.min_gap
        n_degen_arr[k]  = res.n_degen_skipped
        phase_arr[k]    = ξ < -0.5 ? "B2" : "A2"
    end

    return (
        xi              = ξ_vals,
        norm_sq         = norm_sq_arr,
        min_gap         = min_gap_arr,
        n_degen_skipped = n_degen_arr,
        phase           = phase_arr,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

"""
    print_agp_scan_table(scan)

Pretty-print the NamedTuple returned by scan_agp_norm.
"""
function print_agp_scan_table(scan)
    println()
    println("  ξ         Phase   ‖A_ξ‖²_F          min|ΔE|        skipped")
    println("  " * "─"^62)
    for i in eachindex(scan.xi)
        ξ  = scan.xi[i]
        ph = scan.phase[i]
        @printf("  %+7.4f   %-3s     %+14.6e   %+12.6e   %4d\n",
                ξ, ph,
                scan.norm_sq[i],
                scan.min_gap[i],
                scan.n_degen_skipped[i])
        if i < length(scan.xi) && scan.phase[i] == "B2" && scan.phase[i+1] == "A2"
            println("  " * "─"^22 * "  ξ = −1/2 exceptional point  " * "─"^10)
        end
    end
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

function _get_dH(N::Int, ξ::Real, χ::Real, param::Symbol)
    if param == :xi
        return Matrix(PT_Heisenberg.dH_dxi(N, ξ, χ))
    elseif param == :chi
        return Matrix(PT_Heisenberg.dH_dchi(N, ξ, χ))
    else
        error("param must be :xi or :chi, got :$param")
    end
end

  function _get_dH_sector(N::Int, ξ::Real, χ::Real,
                          param::Symbol, n_up::Int)
      if param == :xi
          return Matrix(PT_Heisenberg.dH_dxi_sector(N, ξ, χ, n_up))
      elseif param == :chi
          return Matrix(PT_Heisenberg.dH_dchi_sector(N, ξ, χ, n_up))
      else
          error("param must be :xi or :chi, got :$param")
      end
  end

end  # module AGP_Norm