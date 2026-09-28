
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

# ─────────────────────────────────────────────
# Phase label
# ─────────────────────────────────────────────

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


function xi_chi_to_h_gamma(ξ::Real, χ::Real)
    denom = ξ^2 + χ^2
    @assert denom > 0 "ξ and χ cannot both be zero"
    return ξ / denom, χ / denom
end


function h_gamma_to_xi_chi(h::Real, γ::Real)
    denom = h^2 + γ^2
    @assert denom > 0 "h and γ cannot both be zero"
    return h / denom, γ / denom
end

# ─────────────────────────────────────────────
# Main Hamiltonian builder
# ─────────────────────────────────────────────

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


function build_hamiltonian_from_h_gamma(N::Int, h::Real, γ::Real; J::Real=1.0)
    ξ, χ = h_gamma_to_xi_chi(h, γ)
    return build_hamiltonian(N, ξ, χ; J)
end


function dH_dxi(N::Int, ξ::Real, χ::Real)
    dh1 = -1.0 / complex(ξ,  χ)^2
    dhN = -1.0 / complex(ξ, -χ)^2
    return dh1 * embed_operator(σz, 1, N) + dhN * embed_operator(σz, N, N)
end


function dH_dchi(N::Int, ξ::Real, χ::Real)
    dh1 = -im / complex(ξ,  χ)^2
    dhN = +im / complex(ξ, -χ)^2
    return dh1 * embed_operator(σz, 1, N) + dhN * embed_operator(σz, N, N)
end

# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────


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

# ═══════════════════════════════════════════════════════════════════════════════
# Sz-sector restricted Hamiltonian and derivatives
#
# The full Hamiltonian commutes with total Sz = (1/2)Σ σz_j, so H is block-
# diagonal in sectors of fixed n_up (number of ↑ spins). The sector dimension
# is C(N, n_up) instead of 2^N, e.g. for N=9, Sz=+1/2: 126 vs 512.
#
# Since ∂H/∂ξ and ∂H/∂χ only touch the boundary σz terms they are diagonal
# in the Sz basis and also preserve the sector — no cross-sector mixing.
#
# Ground state sector (from Tables 1 & 2 of Kattel et al.):
#   odd N, ξ < 0 (B2/A2):  Sz = +1/2  →  n_up = (N+1)/2
#   odd N, ξ > 0 (B1/A1):  Sz = -1/2  →  n_up = (N-1)/2
# ═══════════════════════════════════════════════════════════════════════════════


function generate_sector_basis(N::Int, n_up::Int)
    @assert 0 <= n_up <= N "n_up=$n_up out of range [0,$N]"
    return [s for s in 0:(2^N - 1) if count_ones(s) == n_up]
end


function ground_state_n_up(N::Int, ξ::Real)
    @assert isodd(N) "Only implemented for odd N (non-degenerate ground state)"
    return ξ < 0 ? (N + 1) ÷ 2 : (N - 1) ÷ 2
end


function build_hamiltonian_sector(N::Int, ξ::Real, χ::Real, n_up::Int;
                                   J::Real = 1.0)
    basis    = generate_sector_basis(N, n_up)
    dim      = length(basis)
    index_of = Dict(basis[i] => i for i in 1:dim)

    h1 = 1.0 / complex(ξ,  χ)    # left  boundary field
    hN = 1.0 / complex(ξ, -χ)    # right boundary field

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    for (col, state) in enumerate(basis)
        diag = zero(ComplexF64)

        # ── Boundary terms (diagonal) ─────────────────────────────────────────
        # σz eigenvalue: ↑ (bit=1) → +1,  ↓ (bit=0) → -1
        diag += h1 * (((state >> 0)     & 1) == 1 ? 1 : -1)   # site 1 = bit 0
        diag += hN * (((state >> (N-1)) & 1) == 1 ? 1 : -1)   # site N = bit N-1

        # ── Bulk Heisenberg (sites j=0..N-2 in 0-indexed bits) ───────────────
        for j in 0:N-2
            s_j   = (state >> j)     & 1
            s_jp1 = (state >> (j+1)) & 1

            # σz_j σz_{j+1}: diagonal
            diag += J * (s_j == s_jp1 ? 1 : -1)

            # σx σx + σy σy: off-diagonal, only when spins differ
            if s_j != s_jp1
                newstate = state ⊻ (1 << j) ⊻ (1 << (j+1))   # flip both
                row = index_of[newstate]
                push!(rows, row)
                push!(cols, col)
                push!(vals, ComplexF64(2J))
            end
        end

        push!(rows, col)
        push!(cols, col)
        push!(vals, diag)
    end

    return sparse(rows, cols, vals, dim, dim)
end


function dH_dxi_sector(N::Int, ξ::Real, χ::Real, n_up::Int)
    basis = generate_sector_basis(N, n_up)
    dh1   = -1.0 / complex(ξ,  χ)^2
    dhN   = -1.0 / complex(ξ, -χ)^2
    diag  = ComplexF64[
        dh1 * (((s >> 0)     & 1) == 1 ? 1 : -1) +
        dhN * (((s >> (N-1)) & 1) == 1 ? 1 : -1)
        for s in basis
    ]
    return spdiagm(0 => diag)
end


function dH_dchi_sector(N::Int, ξ::Real, χ::Real, n_up::Int)
    basis = generate_sector_basis(N, n_up)
    dh1   = -im / complex(ξ,  χ)^2
    dhN   = +im / complex(ξ, -χ)^2
    diag  = ComplexF64[
        dh1 * (((s >> 0)     & 1) == 1 ? 1 : -1) +
        dhN * (((s >> (N-1)) & 1) == 1 ? 1 : -1)
        for s in basis
    ]
    return spdiagm(0 => diag)
end

# Uncomment to run:
# demo()
end