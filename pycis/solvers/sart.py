import numpy as np
import scipy.sparse as sps

def solve(A,b,iterations=50,lam_start=1.0):
    '''
    Slightly modified (code tidying, but functionally identical)
    version of James Harrison's SART sparse matrix solver. 
    Algorithm based on Scott Silburn's MATLAB code.

    Solves A*x = b with non-negativity constraint, using
    a fixed number of iterations.

    Parameters:

        A (scipy sparse matrix) : Matrix 

        b (scipy sparse matrix) : Data vector

        iterations (int): Number of iterations

        lam_start (float) : Initial value of relaxation parameter lambda

    Returns:

        Numpy matrix containing x

        Array of length n_iterations indicating convergence behaviour
    '''
    shap = A.shape
    lam = lam_start
    colsum = (A.transpose()).dot(sps.csc_matrix(np.ones(shap[0])).transpose())
    lamda = colsum
    #lamda = lamda.multiply(colsum != 0)
    np.reciprocal(lamda.data,out=lamda.data)
    np.multiply(lamda.data,lam,out=lamda.data)
    
    # Initialise output
    sol = sps.csc_matrix(np.zeros((shap[1],1))+np.exp(-1))
    # Create an array to monitor the convergence
    conv = np.zeros(iterations)
    
    for i in range(iterations):
        # Calculate sol_new = sol+lambda*(x'*(b-Ax))
        tmp = b.transpose()-A.dot(sol)
        tmp2 = A.transpose().dot(tmp)
        #newsol = sol+tmp2*lamda
        newsol = sol+tmp2.multiply(lamda)
        # Eliminate negative values
        newsol = newsol.multiply(newsol > 0.0)
        newsol.eliminate_zeros()
        # Calculate how quickly the code is converging
        conv[i] = (sol.multiply(sol).sum()-newsol.multiply(newsol).sum())/sol.multiply(sol).sum()
        # Set the new solution to be the old solution and repeat
        sol = newsol
        # Clear up memory leaks
        tmp = None
        tmp2 = None
      
    sol = None	  
    return newsol.todense(), conv