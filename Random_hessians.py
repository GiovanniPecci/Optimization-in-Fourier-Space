import time
import numpy as np
import os 
import scipy.sparse.linalg
from scipy.stats import special_ortho_group as og
from numba import jit

# Generate N_samples random positive definite matrices dim x dim dimensional.
# The structure of the spectrum is : eigenvalues[i] = base*i**(exponent) + q
# A constant seed fixes the unitary transformation
def generate_rndm_hessian_from_poly(base, exp, q, dim, seed = None):
    np.random.seed(seed)


    eigenvalues = [base*i**(exp) + q for i in range(dim)] # Random spectrum in real space
    hessian_diagonal = np.diag(eigenvalues)

    U = og.rvs(dim) # Random unitary
    random_hessian = U @ hessian_diagonal @ (np.conj(U)).T # Random hessians in real space
        

    return random_hessian , eigenvalues


@jit(nopython = True)
def outer_numba(a, b):
    n = len(a)
    m = len(b)
    result = np.zeros((n, m), dtype=a.dtype)
    
    for i in range(n):
        for j in range(m):
            result[i, j] = a[i] * b[j]
    
    return result


def fi_func(m, P, r_i):
    return np.sin( np.pi/P*m*r_i)


@jit(nopython=True)
def functional_define(dim, r):
    P = dim//2
    Nc = len(r)
    
    Ln_ = np.zeros(P)    
    fi_ = np.zeros((P, Nc))
    
    for m in range(P):
      for i in range(Nc):
          fi_[m,i] = np.sin( np.pi/P*m*r[i]) 
      Ln_[m] = m/P   
                
    return fi_ , Ln_


def hessian_fourier_(H, r):
    N_c = len(r)
    dim = H.shape[0]

    dim_fou = 2*N_c
    
    F = np.zeros((dim, dim, dim_fou, dim_fou))
    fi_mat, Ln_mat = functional_define(dim, r)

    # m n i j : Even Even Even Even     
    for m in range(0,dim,2):
        for n in range(0,dim,2):
            F[m,n, ::2, ::2] = H[m,n] * Ln_mat[m//2] * Ln_mat[n//2] *  np.outer(fi_mat[m//2,:], fi_mat[n//2,:])


    # m n i j  Even Odd Even Odd          
    for m in range(0, dim, 2):
        for n in range(1, dim, 2):
                    F[m,n,::2,1::2] = H[m,n] * Ln_mat[m//2] * (1 - Ln_mat[(n-1)//2]) * np.outer(fi_mat[m//2,:], fi_mat[(n-1)//2,:])

    # m n i j  Odd Even Odd Even          
    for m in range(1, dim, 2):
        for n in range(0, dim, 2):
                    F[m,n,1::2,::2] = H[m,n] * (1 - Ln_mat[(m-1)//2]) * Ln_mat[n//2] * np.outer(fi_mat[(m-1)//2, :], fi_mat[n//2, :])


    # m n i j  Odd Odd Odd Odd          
    for m in range(1, dim, 2):
        for n in range(1, dim, 2):
            F[m,n,1::2,1::2] = H[m,n] * (1 - Ln_mat[(m-1)//2]) * (1 - Ln_mat[(n-1)//2]) * np.outer( fi_mat[(m-1)//2,:], fi_mat[(n-1)//2, :] )

    HF = np.einsum('mnij -> ij' , F )  
    
    ei_HF = scipy.linalg.eigvalsh(HF)

    return ei_HF 

@jit(nopython=True)
def hessian_fourier(H, r):
    N_c = len(r)
    dim = H.shape[0]

    dim_fou = 2*N_c
    
    F = np.zeros((dim, dim, dim_fou, dim_fou))
    fi_mat, Ln_mat = functional_define(dim, r)

    # m n i j : Even Even Even Even     
    for m in range(0,dim,2):
        for n in range(0,dim,2):
            F[m,n, ::2, ::2] = H[m,n] * Ln_mat[m//2] * Ln_mat[n//2] *  outer_numba(fi_mat[m//2,:], fi_mat[n//2,:])


    # m n i j  Even Odd Even Odd          
    for m in range(0, dim, 2):
        for n in range(1, dim, 2):
                    F[m,n,::2,1::2] = H[m,n] * Ln_mat[m//2] * (1 - Ln_mat[(n-1)//2]) * outer_numba(fi_mat[m//2,:], fi_mat[(n-1)//2,:])

    # m n i j  Odd Even Odd Even          
    for m in range(1, dim, 2):
        for n in range(0, dim, 2):
                    F[m,n,1::2,::2] = H[m,n] * (1 - Ln_mat[(m-1)//2]) * Ln_mat[n//2] * outer_numba(fi_mat[(m-1)//2, :], fi_mat[n//2, :])


    # m n i j  Odd Odd Odd Odd          
    for m in range(1, dim, 2):
        for n in range(1, dim, 2):
            F[m,n,1::2,1::2] = H[m,n] * (1 - Ln_mat[(m-1)//2]) * (1 - Ln_mat[(n-1)//2]) * outer_numba( fi_mat[(m-1)//2,:], fi_mat[(n-1)//2, :] )

    HF = np.einsum('mnij -> ij' , F )  
    
    ei_HF = scipy.linalg.eigvalsh(HF)

    return ei_HF 



def fourier_transf(Nc, RAND, h, seed_iter = None):
    np.random.seed(seed_iter)

    r = np.arange(1,Nc+1)
    if RAND:
        r = np.array([ r[j]*(1 + np.random.rand() - .5) for j in range(Nc)] ) # Multiplicative noise

    eigenvalues_fou = hessian_fourier_(h, r)
    
    return r, eigenvalues_fou  #, eigenvectors_fou


    



N_samples = 100 # Number of random hessians in real space
dim = 100 # = 2P dimension of hessian in real space
seed_hess = 10 # seed of the random hessian in real space

current_dir =  os. getcwd()
hess_dir = current_dir+'/Random_hessians/dim_{}/jit/'.format(dim)
if not os.path.exists(hess_dir):os.makedirs(hess_dir)


Ncs = [dim//2] # -> 2*Ncs = dimension of hessian in Fourier space 
n_iter = 10 # Iteration of different random frequencies



#### HERE WE START ####

# Here we generate the N_samples hessians in real space
hess_samples , ei_list = np.zeros((N_samples, dim, dim)) , np.zeros((N_samples, dim))

for j in range(N_samples):
    np.random.seed(None)
    
    base = 0.01*np.random.rand()
    exp = (2*np.random.rand())+3
    q = 0#0.0015*np.random.rand()
    hess_samples[j] , ei_list[j] = generate_rndm_hessian_from_poly(base, exp, q, dim, seed = seed_hess)


file_real_space = hess_dir+'REAL_eigs_seedhess_{}.dat'.format(seed_hess)
np.savetxt(file_real_space, ei_list.T)


for Nc in Ncs:
    print('Nc = ', Nc)
    st = time.time()


    eigval_fou = np.zeros((N_samples, 2*Nc)) 
    eigval_rand_j =  np.zeros((n_iter, N_samples, 2*Nc)) 
   
    for p, h in enumerate(hess_samples):
        print(p+1, '/', N_samples)
    
        RAND = False    
        r, eigval_fou[p] = fourier_transf(Nc, RAND, h, seed_iter = None)  
        file_fou_space = hess_dir+'FOURIER_Nc_{}_eigs_seedhess_{}.dat'.format(Nc, seed_hess)
        

        RAND = True
        seed_iter = None
        for j in range(n_iter):
            print('random iter n = ', j+1, '/', n_iter)
            np.random.seed(seed_iter)
         
            r, eigval_rand_j[j,p] = fourier_transf(Nc, RAND, h, seed_iter = None)
             
        file_rand_fou_space = hess_dir+'RANDFOURIER_Nc_{}_eigs_seedhess_{}.dat'.format(Nc, seed_hess)
         
        
        np.savetxt(file_fou_space, eigval_fou.T)
        
        with open(file_rand_fou_space, 'w') as f:
            for i in range(n_iter):
                np.savetxt(f, eigval_rand_j[i].T)
                f.write('\n')
       
    print('Elapsed time = ' + str(time.time() - st), 'seconds')







