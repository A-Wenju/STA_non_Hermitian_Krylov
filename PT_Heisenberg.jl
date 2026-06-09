"""
PT-symmetric non-Hermitian Heisenberg spin chain
=================================================
Reference: Kattel, Pasnoori & Andrei, J. Phys. A 56, 325001 (2023)

Hamiltonian (paper eq. 32):
    H = Σ_{j=1}^{N-1} Σ_{α=x,y,z} σα_j σα_{j+1} + h1 σz_1 + hN σz_N

with PT-symmetric boundary fields:
    h1 = 1 / (ξ + iχ)       (left boundary)
    hN = 1 / (ξ - iχ)       (right boundary)

where ξ, χ ∈ ℝ are the two real parameters of the model.

ξ is the phase-controlling parameter (Tables 1 & 2 of the paper):
    ξ = Re(1/h1) = h / (h² + γ²)    [if one uses the h-iγ convention]

Phase diagram:
    B2:  ξ < -1/2   →  PT unbroken, no boundary strings
    A2:  -1/2 < ξ < 0   →  PT broken sector present
    A1:  0 < ξ < 1/2    →  PT broken sector present
    B1:  ξ > 1/2    →  PT unbroken, no boundary strings

Phase transitions (exceptional points) at ξ = ±1/2.
"""
module PT_Heisenberg

using LinearAlgebra, SparseArrays, Printf

# ─────────────────────────────────────────────
# Single-site Pauli matrices (2×2)
# ─────────────────────────────────────────────
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const I2 = ComplexF64[1 0; 0 1]

# ─────────────────────────────────────────────
# Kronecker embedding helpers
# ─────────────────────────────────────────────

"""
    embed_operator(op, j, N)

Embed a single-site 2×2 operator `op` at site j of an N-site chain.
Returns a sparse 2^N × 2^N matrix.
"""
function embed_operator(op::AbstractMatrix, j::Int, N::Int)
    @assert 1 <= j <= N "Site j=$j out of range [1,$N]"
    mat = (j == 1) ? op : kron(Matrix(I, 2^(j-1), 2^(j-1)), op)
    r   = N - j
    mat = (r  > 0) ? kron(mat, Matrix(I, 2^r, 2^r)) : mat
    return sparse(mat)
end

"""
    embed_two_site(op1, op2, j, N)

Embed op1_j ⊗ op2_{j+1} into the full 2^N-dimensional Hilbert space.
"""
function embed_two_site(op1::AbstractMatrix, op2::AbstractMatrix, j::Int, N::Int)
    @assert 1 <= j <= N-1 "Bond j=$j out of range [1,$(N-1)]"
    bond = kron(op1, op2)
    mat  = (j > 1)  ? kron(Matrix(I, 2^(j-1), 2^(j-1)), bond) : bond
    r    = N - j - 1
    mat  = (r  > 0) ? kron(mat, Matrix(I, 2^r, 2^r))           : mat
    return sparse(mat)
end

# ─────────────────────────────────────────────
# Phase label
# ─────────────────────────────────────────────

"""
    phase_label(ξ)

Return a string identifying which phase ξ corresponds to,
following the paper's classification (Tables 1 & 2).
"""
function phase_label(ξ::Real)
    if ξ > 0.5 + 1e-10
        return "B1  (ξ = $(round(ξ,digits=5)) > 1/2,  PT unbroken everywhere)"
    elseif ξ < -0.5 - 1e-10
        return "B2  (ξ = $(round(ξ,digits=5)) < -1/2, PT unbroken everywhere)"
    elseif 0.0 + 1e-10 < ξ < 0.5 - 1e-10
        return "A1  (0 < ξ = $(round(ξ,digits=5)) < 1/2, PT broken sector present)"
    elseif -0.5 + 1e-10 < ξ < 0.0 - 1e-10
        return "A2  (-1/2 < ξ = $(round(ξ,digits=5)) < 0, PT broken sector present)"
    elseif abs(ξ - 0.5) < 1e-10 || abs(ξ + 0.5) < 1e-10
        return "Exceptional point  (ξ ≈ ±1/2)"
    else
        return "A1/A2 boundary  (ξ ≈ 0)"
    end
end

# ─────────────────────────────────────────────
# Convert between parameterizations
# ─────────────────────────────────────────────

"""
    xi_chi_to_h_gamma(ξ, χ)

Convert the paper's (ξ, χ) parameterization to the (h, γ) convention
where h1 = h - iγ.

Since h1 = 1/(ξ + iχ):
    h = Re(h1) =  ξ / (ξ² + χ²)
    γ = -Im(h1) = χ / (ξ² + χ²)
"""
function xi_chi_to_h_gamma(ξ::Real, χ::Real)
    denom = ξ^2 + χ^2
    @assert denom > 0 "ξ and χ cannot both be zero"
    return ξ / denom, χ / denom
end

"""
    h_gamma_to_xi_chi(h, γ)

Convert (h, γ) with h1 = h - iγ to the paper's (ξ, χ) parameterization.

Since ξ + iχ = 1/h1 = 1/(h - iγ):
    ξ = Re(1/h1) =  h / (h² + γ²)
    χ = Im(1/h1) =  γ / (h² + γ²)
"""
function h_gamma_to_xi_chi(h::Real, γ::Real)
    denom = h^2 + γ^2
    @assert denom > 0 "h and γ cannot both be zero"
    return h / denom, γ / denom
end

# ─────────────────────────────────────────────
# Main Hamiltonian builder
# ─────────────────────────────────────────────

"""
    build_hamiltonian(N, ξ, χ; J=1.0)

Build the full 2^N × 2^N Hamiltonian for the PT-symmetric Heisenberg chain
using the paper's native (ξ, χ) parameterization (eq. 32):

    H = J Σ_{j=1}^{N-1} (σx_j σx_{j+1} + σy_j σy_{j+1} + σz_j σz_{j+1})
        + h1 σz_1  +  hN σz_N

where:
    h1 = 1/(ξ + iχ)    left boundary  (gain)
    hN = 1/(ξ - iχ)    right boundary (loss)

Parameters
----------
N  : number of sites (≥ 2)
ξ  : real boundary parameter (phase transitions at |ξ| = 1/2)
χ  : imaginary boundary parameter (PT-breaking strength)
J  : exchange coupling (default 1.0)

Returns
-------
H  : sparse ComplexF64 matrix of size 2^N × 2^N

Notes
-----
- H† ≠ H whenever χ ≠ 0 (non-Hermitian)
- [PT, H] = 0 for all ξ, χ (PT-symmetric by construction, paper eq. 32)
- χ = 0 recovers the Hermitian chain with real boundary fields h1 = hN = 1/ξ
- The derivative ∂H/∂ξ and ∂H/∂χ needed for the AGP are simply:
      ∂H/∂ξ = ∂h1/∂ξ σz_1 + ∂hN/∂ξ σz_N
      ∂H/∂χ = ∂h1/∂χ σz_1 + ∂hN/∂χ σz_N
  with ∂h1/∂ξ = -1/(ξ+iχ)², ∂h1/∂χ = -i/(ξ+iχ)²
"""
function build_hamiltonian(N::Int, ξ::Real, χ::Real; J::Real=1.0)
    @assert N >= 2 "Need at least N=2 sites"

    H = spzeros(ComplexF64, 2^N, 2^N)

    # ── Bulk: isotropic Heisenberg exchange ───────────────────────────
    for j in 1:(N-1)
        H += J * embed_two_site(σx, σx, j, N)
        H += J * embed_two_site(σy, σy, j, N)
        H += J * embed_two_site(σz, σz, j, N)
    end

    # ── Boundary fields (paper eq. 32) ────────────────────────────────
    h1 = 1.0 / complex(ξ,  χ)    # 1/(ξ + iχ)
    hN = 1.0 / complex(ξ, -χ)    # 1/(ξ - iχ) = conj(h1)

    H += h1 * embed_operator(σz, 1, N)
    H += hN * embed_operator(σz, N, N)

    return H
end

"""
    build_hamiltonian_from_h_gamma(N, h, γ; J=1.0)

Convenience wrapper: accepts the (h, γ) convention where h1 = h - iγ,
converts to (ξ, χ), and calls build_hamiltonian.
"""
function build_hamiltonian_from_h_gamma(N::Int, h::Real, γ::Real; J::Real=1.0)
    ξ, χ = h_gamma_to_xi_chi(h, γ)
    return build_hamiltonian(N, ξ, χ; J)
end

"""
    dH_dxi(N, ξ, χ)

Derivative of H with respect to ξ (needed for AGP computation).
∂H/∂ξ = (∂h1/∂ξ) σz_1 + (∂hN/∂ξ) σz_N
where ∂h1/∂ξ = -1/(ξ+iχ)² and ∂hN/∂ξ = -1/(ξ-iχ)²
"""
function dH_dxi(N::Int, ξ::Real, χ::Real)
    dh1 = -1.0 / complex(ξ,  χ)^2
    dhN = -1.0 / complex(ξ, -χ)^2
    return dh1 * embed_operator(σz, 1, N) + dhN * embed_operator(σz, N, N)
end

"""
    dH_dchi(N, ξ, χ)

Derivative of H with respect to χ (needed for AGP computation).
∂H/∂χ = (∂h1/∂χ) σz_1 + (∂hN/∂χ) σz_N
where ∂h1/∂χ = -i/(ξ+iχ)² and ∂hN/∂χ = +i/(ξ-iχ)²
"""
function dH_dchi(N::Int, ξ::Real, χ::Real)
    dh1 = -im / complex(ξ,  χ)^2
    dhN = +im / complex(ξ, -χ)^2
    return dh1 * embed_operator(σz, 1, N) + dhN * embed_operator(σz, N, N)
end

# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────

"""
    hamiltonian_info(N, ξ, χ; J=1.0)

Build H, diagonalize it, and print a full diagnostic summary.
Returns (H, eigenvalues).
"""
function hamiltonian_info(N::Int, ξ::Real, χ::Real; J::Real=1.0)
    H  = build_hamiltonian(N, ξ, χ; J)
    h1 = 1.0 / complex(ξ, χ)
    hN = conj(h1)
    Ev = eigvals(Matrix(H))

    thresh    = max(1e-8, 1e-8 * maximum(abs.(Ev)))
    n_complex = count(e -> abs(imag(e)) > thresh, Ev)
    n_real    = length(Ev) - n_complex
    is_herm   = norm(Matrix(H) - Matrix(H)') < 1e-10

    println("="^65)
    println("PT-symmetric Heisenberg chain  (N=$N, J=$J)")
    println("  Parameters:  ξ = $ξ,  χ = $χ")
    @printf("  Boundary:    h1 = %.5f %+.5fi,  hN = %.5f %+.5fi\n",
            real(h1), imag(h1), real(hN), imag(hN))
    println("  Phase:       $(phase_label(ξ))")
    println("  Hermitian?   $is_herm  (true only when χ=0)")
    println("  Spectrum:    $n_real real,  $n_complex complex eigenvalues")
    println("="^65)

    return H, Ev
end

# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

function demo()
    println("\n──── B1 phase (ξ=-0.8, χ=0.1): all eigenvalues should be real ────")
    _, E = hamiltonian_info(4, -0.8, 0.1)
    for e in sort(E, by=real)[1:16]
        @printf("  %+.6f  %+.6fi\n", real(e), imag(e))
    end

    println("\n──── A1 phase (ξ=0.3, χ=0.3): complex eigenvalues present ────")
    _, E = hamiltonian_info(4, 0.3, 0.3)
    for e in sort(E, by=real)[1:8]
        @printf("  %+.6f  %+.6fi\n", real(e), imag(e))
    end

    println("\n──── Hermitian limit (ξ=0.8, χ=0.0): all eigenvalues real ────")
    _, E = hamiltonian_info(4, 0.8, 0.0)
    for e in sort(real.(E))[1:8]
        @printf("  %+.6f\n", e)
    end

    println("\n──── Parameterization check ────")
    h, γ = 1.0, 0.5
    ξ, χ = h_gamma_to_xi_chi(h, γ)
    h2, γ2 = xi_chi_to_h_gamma(ξ, χ)
    @printf("  (h=%.2f, γ=%.2f) → (ξ=%.5f, χ=%.5f) → (h=%.2f, γ=%.2f)\n",
            h, γ, ξ, χ, h2, γ2)
    println("  Phase for these parameters: $(phase_label(ξ))")

    println("\n──── Scanning ξ across phase boundaries at χ=0.3 ────")
    for ξ in [-0.8, -0.45, -0.2, 0.2, 0.45, 0.8]
        println("  $(phase_label(ξ))")
    end
end

# Uncomment to run:
# demo()
end