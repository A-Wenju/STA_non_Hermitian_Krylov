module biLanczos
using LinearAlgebra, SparseMatricesCSR
@inline function algorithm(H, dHdt)
    #define the required functions 
    dim_H = size(H,1) #get the dimension of the Hamiltonian
    ip(a,b) = tr(a'b)/dim_H #define the inner product function for the biorthogonal Lanczos algorithm
    L(Ham,a) = Ham*a-a*Ham   #set the Liouvillian superoperator 
    L_dagger(Ham,a) = Ham'*a-a*Ham'  #set the adjoint of the Liouvillian superoperator 
    a_n(Qn,Ham,Pn) = ip(Qn,L(Ham,Pn)); #compute the a_n coefficient for the biorthogonal Lanczos algorithm

    #This cell performs the biorthogonal Lanczos algorithm to construct the Krylov subspace and compute the necessary coefficients for the STA protocol.
    #setup the initial vectors for the biorthogonal Lanczos algorithm
    P0 = Q0 = dHdt #set the initial P and Q vectors
    init_norm = sqrt(ip(Q0,P0))
    P0 = Q0 = dHdt/init_norm #normalize the initial P and Q vectors

    R0 = L(H, P0) - a_n(Q0,H,P0)*P0 #compute the first R vector using the Liouvillian superoperator and the a_n coefficient
    S0 = L_dagger(H, Q0) - conj(a_n(Q0,H,P0))*Q0

    dim_K = dim_H^2 #Int(length(H) - sqrt(length(H)) + 2) #dimension of the Krylov subspace

    P_list = Vector{Matrix{ComplexF64}}(undef, dim_K) #initialize a list to store the P vectors
    Q_list = Vector{Matrix{ComplexF64}}(undef, dim_K) #initialize a list to store the Q vectors

    a_list = Vector{ComplexF64}(undef, dim_K) #initialize a list to store the a coefficients
    ω_list = Vector{ComplexF64}(undef, dim_K) #initialize a list to store the ω coefficients
    c_list = Vector{ComplexF64}(undef, dim_K) #initialize a list to store the c coefficients
    b_list = Vector{ComplexF64}(undef, dim_K) #initialize a list to store the b coefficients

    P_list[1], Q_list[1] = P0, Q0 #store the first P and Q vectors
    ω_list[1], c_list[1], b_list[1] = 1.0,1.0,1.0 #store the first ω, c, and b coefficients
    a_list[1] = a_n(Q0,H,P0) #store the first a coefficient

    J=1 #initialize the iteration counter
    for j in 2:dim_K
        ω_j = ip(S0,R0) #compute the ω coefficient
        c_j = sqrt(abs(ω_j)) #compute the c coefficient
        b_j = ω_j/c_j #compute the b coefficient

        P_j = R0/c_j #compute the P vector
        Q_j = S0/conj(b_j) #compute the Q vector

        for _ in 1:3
            P_j = P_j - sum([ip(Q_list[k],P_j)*P_list[k] for k in 1:j-1]) #orthogonalize P_j against previous P vectors
            Q_j = Q_j - sum([ip(P_list[k],Q_j)*Q_list[k] for k in 1:j-1]) #orthogonalize Q_j against previous Q vectors
        end

        # num_error_test = maximum(abs.([ip(P_list[k],Q_j) for k in 1:j-1])) #check the orthogonalization
        # count = 1
        # while num_error_test > 1e-10  #unless you meet the threshold, do explicit orthogonalization
        #     # println(num_error_test)
        #     # println(count)
        #     count +=1
        #     P_j = P_j - sum([ip(Q_list[k],P_j)*P_list[k] for k in 1:j-1]) #orthogonalize P_j against previous P vectors
        #     Q_j= Q_j - sum([ip(P_list[k],Q_j)*Q_list[k] for k in 1:j-1]) #orthogonalize Q_j against previous Q vectors

        #     num_error_test = maximum(abs.([ip(P_list[k],Q_j) for k in 1:j-1]))

        #     if count > 3  #Put an emergency break
        #         break
        #     end
        # end


        #println("Iteration $j: ω = $ω_j, c = $c_j, b = $b_j")
        if c_j < 1e-5 #check for convergence
            #println("Convergence reached at iteration $j with ω = $ω_j")
            break
        end
        J+=1 #increment the iteration counter

        P_list[j], Q_list[j] = P_j, Q_j #store the P and Q vectors
        ω_list[j], c_list[j], b_list[j] = ω_j, c_j, b_j #store the ω, c, and b coefficients

        a_j = a_n(Q_j,H,P_j) #store the a coefficient
        a_list[j] = a_j
        R0 = L(H, P_j) - a_j*P_j - b_j*P0 #compute the next R and S vector
        S0 = L_dagger(H, Q_j) - conj(a_j)*Q_j - c_j*Q0
        
        P0, Q0 = P_j, Q_j #Reset P0 and Q0 for the next iteration
    end

    dim_K_true = J #get the actual dimension of the Krylov subspace constructed

    P_list, Q_list = P_list[1:dim_K_true], Q_list[1:dim_K_true] #truncate the list of P and Q vectors to the actual dimension
    ω_list, c_list, b_list = ω_list[1:dim_K_true], c_list[1:dim_K_true], b_list[1:dim_K_true]; #truncate the list of ω, c, and b coeffs
    a_list = a_list[1:dim_K_true]; #truncate the list of a coeffs

    return P_list, Q_list, ω_list, c_list, b_list, a_list, init_norm
end

end #module
