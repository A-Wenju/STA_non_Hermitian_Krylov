"""
core.jl -- all physics for the PT-symmetric Heisenberg chain AGP-norm figure

    heisenberg_agp_cascade   (data_cascade.jl -> plot.jl)

Model (open chain, N sites, boundary fields h_1 = 1/(ξ + iχ), h_N = conj(h_1)):

    H(ξ, χ) = J Σ_{j=1}^{N-1} Σ_{α=x,y,z} σ^α_j σ^α_{j+1} + h_1 σ^z_1 + h_N σ^z_N

The spectrum is PT-unbroken in B2 (ξ < -1/2) and has a PT-broken sector in
A2 (-1/2 < ξ < 0); ξ = -1/2 is the exceptional point where the AGP norm peaks.

Exact AGP norm (biorthogonal eigenbasis H|n_R⟩ = E_n|n_R⟩, ⟨m_L|n_R⟩ = δ_mn):

    ‖A_ξ‖² = Σ_{m≠n} |⟨m_L|∂_ξH|n_R⟩|² / |E_n - E_m|²

Truncated Krylov AGP: Arnoldi on the Liouvillian L = [H, ·] seeded with ∂_ξH,
built ONCE to depth K_max; each truncation K is sliced from that single build
(Arnoldi subspaces are nested).
"""
module HeisenbergCore

using LinearAlgebra, SparseArrays
using KrylovKit

import KrylovKit: inner, scale, scale!!, add!!, zerovector

export build_hamiltonian, dH_dxi,
       biorthogonal_eigensystem, agp_norm_exact_generic, agp_norm_exact,
       KrylovKit_Arnoldi_full, AGP_from_basis

# ─────────────────────────────────────────────────────────────────────────────
# Model: Pauli matrices, Kronecker embedding, Hamiltonian and ∂_ξH
# ─────────────────────────────────────────────────────────────────────────────

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

function embed_operator(op::AbstractMatrix, j::Int, N::Int)
    @assert 1 <= j <= N "Site j=$j out of range [1,$N]"
    mat = (j == 1) ? op : kron(Matrix(I, 2^(j-1), 2^(j-1)), op)
    r   = N - j
    mat = (r  > 0) ? kron(mat, Matrix(I, 2^r, 2^r)) : mat
    return sparse(mat)
end

function embed_two_site(op1::AbstractMatrix, op2::AbstractMatrix, j::Int, N::Int)
    @assert 1 <= j <= N-1 "Bond j=$j out of range [1,$(N-1)]"
    bond = kron(op1, op2)
    mat  = (j > 1)  ? kron(Matrix(I, 2^(j-1), 2^(j-1)), bond) : bond
    r    = N - j - 1
    mat  = (r  > 0) ? kron(mat, Matrix(I, 2^r, 2^r))           : mat
    return sparse(mat)
end

"""
    build_hamiltonian(N, ξ, χ; J=1.0) -> SparseMatrixCSC{ComplexF64}

Full 2^N × 2^N PT-symmetric Heisenberg Hamiltonian (see module docstring).
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

    # ── Boundary fields ───────────────────────────────────────────────
    h1 = 1.0 / complex(ξ,  χ)    # 1/(ξ + iχ)
    hN = 1.0 / complex(ξ, -χ)    # 1/(ξ - iχ) = conj(h1)

    H += h1 * embed_operator(σz, 1, N)
    H += hN * embed_operator(σz, N, N)

    return H
end

"""
    dH_dxi(N, ξ, χ) -> SparseMatrixCSC{ComplexF64}

∂H/∂ξ -- only the boundary fields depend on ξ.
"""
function dH_dxi(N::Int, ξ::Real, χ::Real)
    dh1 = -1.0 / complex(ξ,  χ)^2
    dhN = -1.0 / complex(ξ, -χ)^2
    return dh1 * embed_operator(σz, 1, N) + dhN * embed_operator(σz, N, N)
end

# ─────────────────────────────────────────────────────────────────────────────
# Exact AGP norm (full biorthogonal diagonalisation)
# ─────────────────────────────────────────────────────────────────────────────

"""
    biorthogonal_eigensystem(H) -> (E, VR, VL)
"""
function biorthogonal_eigensystem(H::AbstractMatrix)
    F  = eigen(Matrix(H))
    E  = F.values
    VR = F.vectors

    # Left eigenvectors: rows of VR⁻¹, stored as columns of VL
    VL = conj.(inv(VR)')

    return E, VR, VL
end

"""
    agp_norm_exact_generic(H, dH; degen_tol=1e-6) -> NamedTuple

Returns (norm_sq, n_pairs, n_degen_skipped, min_gap, E).
"""
function agp_norm_exact_generic(H::AbstractMatrix, dH::AbstractMatrix;
                                 degen_tol::Real = 1e-6)
    H  = Matrix(H)
    dH = Matrix(dH)
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
    )
end

"""
    agp_norm_exact(N, ξ, χ; degen_tol=1e-8) -> NamedTuple

Heisenberg wrapper: exact ‖A_ξ‖² at (ξ, χ), driving parameter λ = ξ.
"""
function agp_norm_exact(N::Int, ξ::Real, χ::Real; degen_tol::Real = 1e-8)
    H  = Matrix(build_hamiltonian(N, ξ, χ))
    dH = Matrix(dH_dxi(N, ξ, χ))
    return agp_norm_exact_generic(H, dH; degen_tol=degen_tol)
end

# ─────────────────────────────────────────────────────────────────────────────
# Truncated Krylov AGP (Arnoldi, build once / slice many)
# ─────────────────────────────────────────────────────────────────────────────

"""
    KrylovKit_Arnoldi_full(H, dHdt, K_max; tol=1e-12)

Build the Arnoldi Krylov basis ONCE, up to depth K_max, starting from
A0 = dHdt/||dHdt||. Returns the full basis_ops (length k <= K_max),
the initial residual h_10 = first_normres (same for every truncation K,
since it's computed before any expansion), the FULL (k x k) upper-Hessenberg
matrix, and init_norm.

This is the object every truncated-K AGP is sliced from -- nestedness of
Arnoldi subspaces guarantees the leading K x K principal submatrix and first
K basis vectors are identical to what a from-scratch build to depth K would
produce (same starting vector, same orthogonalizer).
"""
@inline function KrylovKit_Arnoldi_full(H, dHdt, K_max::Int; tol::Float64 = 1e-12)

    scale(A::Matrix{ComplexF64}, α::ComplexF64) = α * A
    scale(A::Matrix{ComplexF64}, α::Real)       = α * A

    scale!!(A::Matrix{ComplexF64}, α::ComplexF64) = (A .*= α; A)
    scale!!(A::Matrix{ComplexF64}, α::Real)       = (A .*= α; A)

    inner(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}) = dot(vec(A), vec(B))

    zerovector(A::Matrix{ComplexF64})      = zero(A)
    zerovector(A::Matrix{ComplexF64}, ::Type{T}) where T = zeros(T, size(A))

    add!!(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, α::Number)         = (A .+= α .* B; A)
    add!!(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, α::Number, β::Number) = (A .= β .* A .+ α .* B; A)

    init_norm = sqrt(real(inner(dHdt, dHdt)))
    A0 = dHdt / init_norm

    L = A -> H * A - A * H

    iterator      = ArnoldiIterator(L, A0)
    factorization = initialize(iterator)

    first_normres = normres(factorization)

    while length(factorization) < K_max
        normres(factorization) < tol && break
        expand!(iterator, factorization)
    end

    k          = length(factorization)
    V          = basis(factorization)
    basis_ops  = [V[i] for i in 1:k]
    H_hess     = Matrix(rayleighquotient(factorization))

    return basis_ops, first_normres, H_hess, init_norm
end

"""
    AGP_from_basis(basis_ops, H_hess, h_10, K, init_norm, dim_H)

Slice the first K basis vectors and leading K x K block of H_hess from a
full Arnoldi build and solve for the truncated AGP at depth K (M = ⌊K/2⌋
terms) -- without rebuilding the basis.
"""
function AGP_from_basis(basis_ops, H_hess, h_10, K::Int, init_norm, dim_H::Int)
    K_eff = min(K, length(basis_ops))
    L_matrix = H_hess[1:K_eff, 1:K_eff]

    LHS_matrix = (L_matrix^2)[2:2:end, 2:2:end]
    dim_A = size(LHS_matrix, 1)

    RHS_vec = zeros(ComplexF64, dim_A)
    RHS_vec[1] = -im * h_10

    AGP_vec = LHS_matrix \ RHS_vec
    AGP_mat = zeros(ComplexF64, dim_H, dim_H)

    for i in 1:dim_A
        AGP_mat += AGP_vec[i] * basis_ops[2*i]
    end

    return init_norm * AGP_mat
end

end  # module HeisenbergCore
