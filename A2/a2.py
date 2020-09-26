#CSC336 Assignment #2 starter code

import numpy as np
import scipy.linalg as sla
#import time
#import matplotlib.pyplot as plt

#Q4a
def p_to_q(p):
    """
    return the permutation vector, q, corresponding to
    the pivot vector, p.
    >>> p_to_q(np.array([2,3,2,3]))
    array([2, 3, 0, 1])
    >>> p_to_q(np.array([2,4,8,3,9,7,6,8,9,9]))
    array([2, 4, 8, 3, 9, 7, 6, 0, 1, 5])
    """
    q = np.array([i for i in range(p.shape[0])]) #replace with your code
    for i in range(p.shape[0]):
        q[i], q[p[i]] = q[p[i]], q[i]
    return q

#Q4b
def solve_plu(A,b):
    """
    return the solution of Ax=b. The solution is calculated
    by calling scipy.linalg.lu_factor, converting the piv
    vector using p_to_q, and solving two triangular linear systems
    using scipy.linalg.solve_triangular.
    """
    lu, piv = sla.lu_factor(A)
    q = p_to_q(piv)
    y = sla.solve_triangular(lu, b[q], lower=True, unit_diagonal=True)
    x = sla.solve_triangular(lu, y)

    return x

def q3_f1(A, B, C, b):
    return np.dot(np.dot(np.dot(np.linalg.inv(B), 2 * A + np.identity(A.shape[0])), np.linalg.inv(C) + A), b)

def q3_f2(A, B, C, b):
    M = 2 * A + np.identity(B.shape[0])
    N = sla.solve(B, M)
    O = sla.solve(C, b) + np.dot(A, b)
    return np.dot(N, O)

def print_graph(min_n, max_n):
    """
    Code for q3c, uncomment line 5 and line 6 before using it.
    """
    n_lst = []
    f1 = []
    f2 = []
    for i in range(min_n, max_n + 1):
        n = 20 * i
        n_lst.append(n)

        A = np.random.uniform(-1, 1, [n, n])
        B = np.random.uniform(-1, 1, [n, n])
        C = np.random.uniform(-1, 1, [n, n])
        b = np.random.uniform(-1, 1, n)

        f1_s = time.perf_counter()
        q3_f1(A, B, C, b)
        f1_e = time.perf_counter()
        f1.append(f1_e - f1_s)

        f2_s = time.perf_counter()
        q3_f2(A, B, C, b)
        f2_e = time.perf_counter()
        f2.append(f2_e - f2_s)
    plt.plot(n_lst, f1, 'b', n_lst, f2, 'r--')
    plt.xlabel("n")
    plt.ylabel("time")
    plt.legend(("formula 1", "formula 2"))
    plt.savefig("a2.png")
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    #test your solve_plu function on a random system
    n = 10
    A = np.random.uniform(-1,1,[n,n])
    b = np.random.uniform(-1,1,n)
    xtrue = sla.solve(A,b)
    x = solve_plu(A,b)
    print("solve_plu works:",np.allclose(x,xtrue,rtol=1e-10,atol=0))

    #print_graph(1, 50)
