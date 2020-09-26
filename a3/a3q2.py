#CSC 336 Summer 2020 A3Q2 starter code

#Note: you may use the provided code or write your own, it is your choice.

#some general imports
import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt

def get_dn(fn):
    """
    Returns dn, given a value for fn, where Ad = b + f (see handout)

    The provided implementation uses the solve_banded approach from A2,
    but feel free to modify it if you want to try to more efficiently obtain
    dn.

    Note, this code uses a global variable for n.
    """

    #the matrix A in banded format
    diagonals = [np.hstack([0,0,np.ones(n-2)]),#zeros aren't used, so can be any value.
                 np.hstack([0,-4*np.ones(n-2),-2]),
                 np.hstack([9,6*np.ones(n-3),[5,1]]),
                 np.hstack([-4*np.ones(n-2),-2,0]),#make sure this -2 is in correct spot
                 np.hstack([np.ones(n-2),0,0])] #zeros aren't used, so can be any value.
    A = np.vstack(diagonals) * n**3

    b = -(1/n) * np.ones(n)

    b[-1] += fn

    sol = solve_banded((2, 2), A, b)
    dn = sol[-1]

    return dn

def newton_iter(fno):
    diagonals = [np.hstack([0,0,np.ones(n-2)]),#zeros aren't used, so can be any value.
                 np.hstack([0,-4*np.ones(n-2),-2]),
                 np.hstack([9,6*np.ones(n-3),[5,1]]),
                 np.hstack([-4*np.ones(n-2),-2,0]),#make sure this -2 is in correct spot
                 np.hstack([np.ones(n-2),0,0])] #zeros aren't used, so can be any value.
    A = np.vstack(diagonals) * n**3
    E = np.zeros(n)
    E[-1] = 1
    A_inv = solve_banded((2,2), A, E)
    b = -(1 / n) * np.ones(n)
    # return np.subtract(fno, np.divide((np.add(np.dot(A_inv.T, b), np.dot(A_inv[-1], fno))), A_inv[-1]))
    return fno - (np.dot(A_inv, b) + fno * A_inv[-1]) / A_inv[-1]

if __name__ == "__main__":
    #experiment code
    nn = []
    time_bre = []
    time_fso = []
    time_new = []

    for i in range(5,17):
        n = 2**i
        nn.append(n)

        start = time.perf_counter()
        fn_brentq = brentq(get_dn, -3, 3)
        end = time.perf_counter()
        time_bre.append(end - start)

        start = time.perf_counter()
        fn_fsolve = fsolve(get_dn, np.array(1))[0]
        end = time.perf_counter()
        time_fso.append(end - start)

        start = time.perf_counter()
        fn_newton = newton_iter(0)
        end = time.perf_counter()
        time_new.append(end - start)

        print("n: " + str(n) + ", relerr_brentq: " + str(format((fn_brentq - 0.375103623599399)/0.375103623599399, '.6e')) + ", relerr_fsolve: " + str(
            format((fn_fsolve - 0.375103623599399)/0.375103623599399, '.6e')) + ", relerr_newton: " + str(format((fn_newton - 0.375103623599399)/0.375103623599399, '.6e')))

    plt.plot(nn, time_bre, 'r', nn, time_fso, 'g', nn, time_new, 'b')
    plt.xlabel("n")
    plt.ylabel("time")
    plt.legend(("brentq", "fsolve", "newton"))
    plt.show()
