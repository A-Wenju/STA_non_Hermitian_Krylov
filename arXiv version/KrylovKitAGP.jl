module KrylovKitAGP
using KrylovKit, LinearAlgebra, SparseArrays

# ── Main function ──────────────────────────────────────────────────────────────
import KrylovKit: inner, scale, scale!!, add!!, zerovector

@inline function KrylovKit_Arnoldi(H,dHdt; tol::Float64 = 1e-12, η = 0.9999 )

    target_dim::Int =size(H,1)^2

     # ── Interface definitions ──────────────────────────────────────────────────────

    scale(A::Matrix{ComplexF64}, α::ComplexF64) = α * A
    scale(A::Matrix{ComplexF64}, α::Real)       = α * A

    scale!!(A::Matrix{ComplexF64}, α::ComplexF64) = (A .*= α; A)
    scale!!(A::Matrix{ComplexF64}, α::Real)       = (A .*= α; A)

    #inner(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}) = dot(vec(A), vec(B))
    inner(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}) = dot(vec(A), vec(B))

    zerovector(A::Matrix{ComplexF64})      = zero(A)
    zerovector(A::Matrix{ComplexF64}, ::Type{T}) where T = zeros(T, size(A))

    add!!(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, α::Number)         = (A .+= α .* B; A)
    add!!(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, α::Number, β::Number) = (A .= β .* A .+ α .* B; A)

    # Normalize starting vector
    init_norm = sqrt(real(inner(dHdt, dHdt)))
    A₀ = dHdt / init_norm

    # Define superoperator
    L = A -> H * A - A * H

    # Build iterator and initialize
    iterator     = ArnoldiIterator(L, A₀, ModifiedGramSchmidtIR(η))
    factorization = initialize(iterator)

    first_normres = normres(factorization)  # ← grab it right here
    #@info "Initial residual: $first_normres"

    # Expand until target dimension or convergence
    while length(factorization) < target_dim
        normres(factorization) < tol && break
        expand!(iterator, factorization)
        # After expand!, grab the new vector and reorthogonalize
        # for j in 1:length(factorization)-1
        #     vⱼ = basis(factorization)[j]
        #     vₖ = basis(factorization)[end]
        #     vₖ .-= inner(vⱼ, vₖ) .* vⱼ
        # end
        #@info "k=$(length(factorization)), normres=$(normres(factorization))"
    end

    # Extract results
    k          = length(factorization)
    V          = basis(factorization)
    basis_ops  = [V[i] for i in 1:k]
    H_hess     = Matrix(rayleighquotient(factorization))

    return basis_ops, first_normres, H_hess, init_norm
end

@inline function KrylovKit_AGP(H, dHdt)  #Calculates AGP for Arnoldi 
    K_list, h_10 , L_matrix, init_norm = KrylovKit_Arnoldi(H, dHdt)  #Get the Liouvillian in Krylov Basis

    LHS_matrix = (L_matrix^2)[2:2:end, 2:2:end]

    dim_A = size(LHS_matrix,1)

    RHS_vec = zeros(ComplexF64, dim_A) #initiate the RHS vector for AGP construction
    RHS_vec[1] = -im*h_10; #set the first element of the RHS vector

    AGP_vec = LHS_matrix\RHS_vec; #compute the AGP coefficients by multiplying the inverse of the LHS matrix with the RHS vector
    AGP_mat = zeros(ComplexF64,size(H)[1], size(H)[1]) #initiate the AGP matrix

    for i in 1:dim_A
        AGP_mat += AGP_vec[i]*K_list[2*i] #construct the AGP by summing the AGP coefficients multiplied by the corresponding Krylov basis vectors
    end

    return init_norm * AGP_mat  #return the AGP matrix
end

@inline function KrylovKit_Arnoldi_finite(H,dHdt, target_dim; tol::Float64 = 1e-12)

     # ── Interface definitions ──────────────────────────────────────────────────────

    scale(A::Matrix{ComplexF64}, α::ComplexF64) = α * A
    scale(A::Matrix{ComplexF64}, α::Real)       = α * A

    scale!!(A::Matrix{ComplexF64}, α::ComplexF64) = (A .*= α; A)
    scale!!(A::Matrix{ComplexF64}, α::Real)       = (A .*= α; A)

    #inner(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}) = dot(vec(A), vec(B))
    inner(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}) = dot(vec(A), vec(B))

    zerovector(A::Matrix{ComplexF64})      = zero(A)
    zerovector(A::Matrix{ComplexF64}, ::Type{T}) where T = zeros(T, size(A))

    add!!(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, α::Number)         = (A .+= α .* B; A)
    add!!(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, α::Number, β::Number) = (A .= β .* A .+ α .* B; A)

    # Normalize starting vector
    init_norm = sqrt(real(inner(dHdt, dHdt)))
    A₀ = dHdt / init_norm

    # Define superoperator
    L = A -> H * A - A * H

    # Build iterator and initialize
    iterator     = ArnoldiIterator(L, A₀)
    factorization = initialize(iterator)

    first_normres = normres(factorization)  # ← grab it right here
    #@info "Initial residual: $first_normres"

    # Expand until target dimension or convergence
    while length(factorization) < target_dim
        normres(factorization) < tol && break
        expand!(iterator, factorization)
        #@info "k=$(length(factorization)), normres=$(normres(factorization))"
    end

    # Extract results
    k          = length(factorization)
    V          = basis(factorization)
    basis_ops  = [V[i] for i in 1:k]
    H_hess     = Matrix(rayleighquotient(factorization))

    return basis_ops, first_normres, H_hess, init_norm
end

@inline function KrylovKit_AGP_finite(H, dHdt, target_dim)  #Calculates AGP for Arnoldi 
    K_list, h_10 , L_matrix, init_norm = KrylovKit_Arnoldi_finite(H, dHdt, target_dim)  #Get the Liouvillian in Krylov Basis

    LHS_matrix = (L_matrix^2)[2:2:end, 2:2:end]

    dim_A = size(LHS_matrix,1)

    RHS_vec = zeros(ComplexF64, dim_A) #initiate the RHS vector for AGP construction
    RHS_vec[1] = -im*h_10; #set the first element of the RHS vector

    AGP_vec = LHS_matrix\RHS_vec; #compute the AGP coefficients by multiplying the inverse of the LHS matrix with the RHS vector
    AGP_mat = zeros(ComplexF64,size(H)[1], size(H)[1]) #initiate the AGP matrix

    for i in 1:dim_A
        AGP_mat += AGP_vec[i]*K_list[2*i] #construct the AGP by summing the AGP coefficients multiplied by the corresponding Krylov basis vectors
    end

    return init_norm * AGP_mat  #return the AGP matrix
end

end