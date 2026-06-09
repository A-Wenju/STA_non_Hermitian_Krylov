module Hatano_Nelson
using SparseArrays, LinearAlgebra, Combinatorics

@inline function build_H_and_dHdt(N, J, U, a, h0, τ, t)
    Np   = Int64(N/2)              # half filling

    #define the helper function to generate the basis states
    function generate_basis(N, Np)
        states = []
        for occ in combinations(1:N, Np) # generate all combinations of Np occupied sites out of N total sites
            state = 0
            for site in occ 
                state |= 1 << (site-1)
            end
            push!(states, state)
        end
        return states
    end

    basis = generate_basis(N, Np)
    dim = length(basis)

    #This program converts all the possible binaries (like 000111, 001101, ...)to decimal and stores them

    # Map state -> index and create a dictionary for quick lookup
    index_of = Dict(basis[i] => i for i in 1:dim);

    #Take the fermion exchange statistics into account when hopping from site i to j. The sign is given by (-1)^(number of fermions between i and j)
    function fermion_sign(state, i, j)
        if i == j
            return 1
        end
        left  = min(i, j) # find the left and right sites
        right = max(i, j)
        
        mask = ((1 << (right-1)) - 1) ⊻ ((1 << left) - 1) # create a mask that has 1s between the two sites and 0s elsewhere

        n_between = count_ones(state & mask)

        return (-1)^n_between
    end

    # -----------------------
    # Build Hamiltonian
    # -----------------------
    function build_H(t)  #build the Hamiltonian matrix at time t
        h = h0 * t / τ

        rows = Int[]  #store the row indices of the nonzero matrix elements
        cols = Int[]
        vals = ComplexF64[]

        for (col, state) in enumerate(basis)  #loop over all basis states

            # nearest-neighbor terms
            for n in 1:N-1

                occ_n   = (state >> (n-1)) & 1 #for the nth lattice site, see if the site and its neighbour is occupied or not
                occ_np1 = (state >> n) & 1

                # interaction term     #Add the interaction term if the two neighboring sites are both occupied
                if occ_n == 1 && occ_np1 == 1
                    push!(rows, col)
                    push!(cols, col)
                    push!(vals, U)
                end

                # hopping n -> n+1   #Add the hopping term if one of the two neighboring sites is occupied and the other is empty   
                if occ_n == 1 && occ_np1 == 0
                    newstate = state ⊻ (1<<(n-1)) ⊻ (1<<n)
                    row = index_of[newstate]
                    sign = fermion_sign(state, n, n+1)  #this isi always one, but is useful if we want to extend the model to include longer-range hopping

                    amp = (J/2) * exp(a*h) * sign

                    push!(rows, row) #Add the matrix element to the Hamiltonian
                    push!(cols, col)
                    push!(vals, amp)
                end

                # hopping n+1 -> n  #Add the hopping term if one of the two neighboring sites is occupied and the other is empty   
                if occ_n == 0 && occ_np1 == 1
                    newstate = state ⊻ (1<<(n-1)) ⊻ (1<<n)
                    row = index_of[newstate]
                    sign = fermion_sign(state, n+1, n)

                    amp = (J/2) * exp(-a*h) * sign

                    push!(rows, row) #Add the matrix element to the Hamiltonian
                    push!(cols, col)
                    push!(vals, amp)
                end

            end
        end

        return sparse(rows, cols, vals, dim, dim) # construct the sparse Hamiltonian matrix
    end

    function build_dHdt(t)  #compute the time derivative of the Hamiltonian
        h = h0 * t / τ
        dhdt = h0 / τ   #the time derivative function
    
        rows = Int[]  #store the row indices of the nonzero matrix elements
        cols = Int[]
        vals = ComplexF64[]

        for (col, state) in enumerate(basis) #loop over all basis states
            for n in 1:N-1

                occ_n   = (state >> (n-1)) & 1
                occ_np1 = (state >> n) & 1

                # hopping n -> n+1
                if occ_n == 1 && occ_np1 == 0  # Add the hopping term if one of the two neighboring sites is occupied and the other is empty
                    newstate = state ⊻ (1<<(n-1)) ⊻ (1<<n)
                    row = index_of[newstate]
                    sign = fermion_sign(state, n, n+1)

                    amp = (J/2) * a * dhdt * exp(a*h) * sign

                    push!(rows, row)
                    push!(cols, col)
                    push!(vals, amp)
                end

                # hopping n+1 -> n  #Add the hopping term if one of the two neighboring sites is occupied and the other is empty
                if occ_n == 0 && occ_np1 == 1
                    newstate = state ⊻ (1<<(n-1)) ⊻ (1<<n)
                    row = index_of[newstate]
                    sign = fermion_sign(state, n+1, n)

                    amp = -(J/2) * a * dhdt * exp(-a*h) * sign 

                    push!(rows, row) #Add the matrix element to the time derivative of the Hamiltonian
                    push!(cols, col)
                    push!(vals, amp)
                end
            end
        end

        return sparse(rows, cols, vals, dim, dim) # construct the sparse matrix for the time derivative of the Hamiltonian
    end

    return build_H(t), build_dHdt(t)
end


end # module
