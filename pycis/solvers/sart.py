import numpy as np
import scipy.sparse as sps

def solve(A, b, non_neg=True, max_iter=2000, tol=1e-2, lam=None, verbose=True):
    """
    Scott Silburn's Simultaneous Algebraic Reconstruction Technique (SART) solver.
    Ported from MATLAB PhD code by James Harrison & Scott Silburn

    Iteratively solves a linear equations A*x = b

    Written for tomographic reconstruction of camera data

    Parameters:

        A (scipy sparse matrix)  : Matrix

        b (scipy sparse matrix)  : Data vector

        non_neg (bool)           : Whether to enforce non-negativity on the solution

        max_iter (int)           : Maximum allowed number of iterations

        tol (float)              : Difference in fractional error between consecutive iterations where the \
                                   result is considered converged and the solver stops.

        lam (float)              : Starting value for lambda parameter ("strength" of each iteration). \
                                   This will be sanity checked by the solver anyway.

        verbose (bool)           : Whether to print status messages

    Returns:

        Numpy matrix containing solution vector x

        Array with as many elements as the number of iterations containing the relative error
        at each iteration. Can be used to see how the convergence goes.
    """

    equations, unknowns = A.shape

    if verbose:
        print('SART Solver: solving system of {:d} equations with {:d} unknowns.'.format(equations, unknowns))

    # Initialise output matrix
    x = sps.csc_matrix(np.ones((unknowns, 1)) * np.exp(-1))

    # Check if we're given lambda or want to optimise it automatically
    if not lam:
        optimise_lam = True
        lam = 1.
    else:
        optimise_lam = False
        if verbose:
            print('Using given lambda = {:.1e}'.format(lam))

    # Lambda is weighted by the structure of the geometry matrix
    colsum = np.abs(np.array(A.T * np.ones((equations, 1))))
    lamda = np.ones(colsum.shape) * lam / colsum
    lamda[colsum == 0] = 0
    lamda = sps.csc_matrix(lamda)

    if optimise_lam:
        # Run 2 iterations and see if the norm of the difference between successive
        # iterations is going up or down. Decrease lambda until it goes down.
        if verbose:
            print('Finding maximum starting lambda value...')
        while True:
            deltas = [0, 0]
            for i in range(3):
                x1 = x + lamda.multiply(A.T * (b.T - A * x))
                if i > 0:
                    deltas[i - 1] = sps.linalg.norm(x1 - x) / sps.linalg.norm(x)
                x = x1

            if deltas[1] > deltas[0]:
                lamda = np.divide(lamda, 1.1)
                lam = lam / 1.1
                x[:] = np.exp(-1)
            else:
                break
        if verbose:
            print('   Using lambda = {:.1e}'.format(lam))

    # List to store the running errror
    err = []
    x[:] = np.exp(-1)

    iteration_number = 1
    converged = False

    print('Starting iterations...')
    while iteration_number <= max_iter:

        # Calculate updated solution
        x1 = x + lamda.multiply(A.T * (b.T - A * x))

        # Non-negativity constraint, if required
        if non_neg:
            x1[x1 < 0] = 0
            x1.eliminate_zeros()

        # Current relative error
        err.append(sps.linalg.norm(b.T - A * x1) / sps.linalg.norm(x))

        # Update the current solution
        x = x1

        # Check if we are converged enough
        if iteration_number >= 2:
            if np.abs(err[-1] - err[-2]) < tol:
                if verbose:
                    print('   Reached convergence criterion after {:d} iterations.'.format(iteration_number))
                converged = True
                break

        iteration_number += 1

    if not converged and verbose:
        print('   Stopped at {:d} iteration limit without reaching convergence criterion.'.format(max_iter))

    return x.todense(), np.array(err)


def save(filename, data, time):

    """
    Save solved profile in a format readable by PyCIS later.

    Parameters:

        filename (string)        : Desired save filename

        data (numpy.ndarray)     : sart.solve() Output

    """

    np.savez(filename, data=data[0], err=data[1], time=time)

    return
