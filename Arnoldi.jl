module Arnoldi
using LinearAlgebra, Base.Threads
@inline function algorithm(H, dHdt) #Define the main function
    #define the required functions 
    dim_H = size(H,1) #get the dimension of the Hamiltonian
    ip(a,b) = tr(a'b)/dim_H #define the inner product function for the Arnoldi iteration
    L(Ham,a) = Ham*a-a*Ham   #set the Liouvillian superoperator 
    h_jk(K_j,Ham,K_k) = ip(K_j,L(Ham,K_k)); #compute the h_j_k coeff for Arnoldi iteration 

       #This cell performs the Arnoldi iteration to construct the Krylov subspace.
    #setup the initial vectors for the Arnoldi iteration
    K0 = dHdt #set the initial vector
    init_norm = sqrt(ip(K0,K0))
    K0 = dHdt/init_norm #normalize the initial vector

    dim_K = dim_H^2  #set the initial dimension of Krylov Space

    K_list = Vector{Matrix{ComplexF64}}(undef, dim_K)  #set the list to Krylov Basis Vectors
    norm_list = Vector{Float64}(undef, dim_K)
    K_list[1], norm_list[1] = K0, 1.0  #Put the initial vectors

    count = 1 
    for n in 2:dim_K
        K_n = L(H,K_list[n-1]) - sum([h_jk(K_list[k], H, K_list[n-1])*K_list[k] for k in 1:n-1]) #Do the iterative calculation
        K_n = K_n - sum([ip(K_list[k],K_n)*K_list[k] for k in 1:n-1])  #Do explicit orthogonalization
        K_n = K_n - sum([ip(K_list[k],K_n)*K_list[k] for k in 1:n-1])  #Do explicit orthogonalization

        norm_K_n = abs(sqrt(ip(K_n,K_n)))  #calculate the norm and normalize
        K_n = K_n/norm_K_n  

        if norm_K_n < 1e-10  #break the algorithm once the full Krylov Space is explored
            break
        end
        count+=1
    
        K_list[n], norm_list[n] = K_n, norm_K_n    #return the basis vectors list 
    end

    dim_K_true = count  #set the actual dimension of Krylov Space

    K_list, norm_list = K_list[1:dim_K_true], norm_list[1:dim_K_true]  #trim the list accordingly

    L_matrix = Matrix{ComplexF64}(undef, dim_K_true, dim_K_true)  #calculate the Liouillian matrix in Krylov Basis

    @threads for j in 1:dim_K_true  #Calculate elements of L in Krylov Basis
        for k in 1:dim_K_true
            L_matrix[j,k] = h_jk(K_list[j], H, K_list[k])
        end
    end

    return K_list, norm_list[2], L_matrix, init_norm  #return the required list
end

function AGP_Arnoldi(H, dHdt)  #Calculates AGP for Arnoldi 
    K_list, h_10 , L_matrix, init_norm = algorithm(H, dHdt)  #Get the Liouvillian in Krylov Basis

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

@inline function algorithm_finite(H, dHdt, dim_K) #this is for finite dimensional Arnoldi 
    #define the required functions 
    dim_H = size(H,1) #get the dimension of the Hamiltonian
    ip(a,b) = tr(a'b)/dim_H #define the inner product function for the Arnoldi iteration
    L(Ham,a) = Ham*a-a*Ham   #set the Liouvillian superoperator 
    h_jk(K_j,Ham,K_k) = ip(K_j,L(Ham,K_k)); #compute the h_j_k coeff for Arnoldi iteration 

       #This cell performs the Arnoldi iteration to construct the Krylov subspace.
    #setup the initial vectors for the Arnoldi iteration
    K0 = dHdt #set the initial vector
    init_norm = sqrt(ip(K0,K0))
    K0 = dHdt/init_norm #normalize the initial vector

    #dim_K = dim_H^2  #set the initial dimension of Krylov Space

    K_list = Vector{Matrix{ComplexF64}}(undef, dim_K)  #set the list to Krylov Basis Vectors
    norm_list = Vector{Float64}(undef, dim_K)
    K_list[1], norm_list[1] = K0, 1.0  #Put the initial vectors

    count = 1 
    for n in 2:dim_K
        K_n = L(H,K_list[n-1]) - sum([h_jk(K_list[k], H, K_list[n-1])*K_list[k] for k in 1:n-1]) #Do the iterative calculation
        K_n = K_n - sum([ip(K_list[k],K_n)*K_list[k] for k in 1:n-1])  #Do explicit orthogonalization
        #K_n = K_n - sum([ip(K_list[k],K_n)*K_list[k] for k in 1:n-1])  #Do explicit orthogonalization

        norm_K_n = abs(sqrt(ip(K_n,K_n)))  #calculate the norm and normalize
        K_n = K_n/norm_K_n  

        if norm_K_n < 1e-10  #break the algorithm once the full Krylov Space is explored
            break
        end
        count+=1
    
        K_list[n], norm_list[n] = K_n, norm_K_n    #return the basis vectors list 
    end

    dim_K_true = count  #set the actual dimension of Krylov Space

    K_list, norm_list = K_list[1:dim_K_true], norm_list[1:dim_K_true]  #trim the list accordingly

    L_matrix = Matrix{ComplexF64}(undef, dim_K_true, dim_K_true)  #calculate the Liouillian matrix in Krylov Basis

    for j in 1:dim_K_true  #Calculate elements of L in Krylov Basis
        for k in 1:dim_K_true
            L_matrix[j,k] = h_jk(K_list[j], H, K_list[k])
        end
    end

    return K_list, norm_list[2], L_matrix, init_norm  #return the required list
end
end

function AGP_Arnoldi_finite(H, dHdt, dim_K)  #Calculates AGP for Arnoldi 
    K_list, h_10 , L_matrix, init_norm = algorithm_finite(H, dHdt, dim_K)  #Get the Liouvillian in Krylov Basis

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