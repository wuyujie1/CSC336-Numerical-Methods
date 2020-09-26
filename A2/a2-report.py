#CSC336 Assignment #2 starter code for the report question

#These are some basic imports you will probably need,
#but feel free to add others here if you need them.
import numpy as np
from scipy.sparse import diags
import scipy.sparse
import scipy.linalg as sla
from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.linalg import solve, solve_triangular, solve_banded
import time
import matplotlib
import matplotlib.pyplot as plt

"""
See the examples in class this week or ask on Piazza if you
aren't sure how to start writing the code
for the report questions.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.
"""

"""
timing code sample (feel free to use timeit if you find it easier)
#you might want to run this in a loop and record
#the average or median time taken to get a more reliable timing
start_time = time.perf_counter()
#your code here to time
time_taken = time.perf_counter() - start_time
"""

#1(a)
def LU(n):
    A = diags(np.array(
        [np.ones(n - 2), np.append(-4 * np.ones(n - 2), [-2]), np.append(9, np.append(6 * np.ones(n - 3), [5, 1])),
         np.append(-4 * np.ones(n - 2), [-2]), np.ones(n - 2)]), (-2, -1, 0, 1, 2), [n, n]).toarray()
    start = time.perf_counter()
    x = sla.solve(A, (-1 / (n ** 4)) * np.ones(n))
    end = time.perf_counter()
    return x, end - start

#1(b)
def LU_band(n):
    ab = np.array(
        [np.append([0, 0], np.ones(n - 2)), np.append([0], np.append(-4 * np.ones(n - 2), [-2])), np.append(9, np.append(6 * np.ones(n - 3), [5, 1])),
         np.append(-4 * np.ones(n - 2), [-2, 0]), np.append(np.ones(n - 2), [0, 0])])
    start = time.perf_counter()
    x = sla.solve_banded((2, 2), ab, (-1 / (n ** 4)) * np.ones(n))
    end = time.perf_counter()
    return x, end - start

#1(c)
def LU_sparse(n):
    start = time.perf_counter()
    A = diags(np.array(
        [np.ones(n - 2), np.append(-4 * np.ones(n - 2), [-2]), np.append(9, np.append(6 * np.ones(n - 3), [5, 1])),
         np.append(-4 * np.ones(n - 2), [-2]), np.ones(n - 2)]), (-2, -1, 0, 1, 2), [n, n], format="csr")
    x = spsolve(A, (-1 / (n ** 4)) * np.ones(n))
    end = time.perf_counter()
    return x, end - start

#1(d)
def prefactored(n):
    R = diags(np.array(
        [np.append([2], np.ones(n - 1)),
         -2 * np.ones(n - 1), np.ones(n - 2)]), (0, 1, 2), [n, n]).toarray()
    start = time.perf_counter()
    y = sla.solve_triangular(R, (-1 / (n ** 4)) * np.ones(n))
    x = sla.solve_triangular(R.T, y, lower=True)
    end = time.perf_counter()
    return x, end - start

#1(e)
def prefactored_band(n):
    R = np.array(
        [np.append([0, 0], np.ones(n - 2)), np.append([0], -2 * np.ones(n - 1)), np.append(2, np.ones(n - 1))])
    RT = np.array(
        [np.append(2, np.ones(n - 1)), np.append(-2 * np.ones(n - 1), [0]), np.append(np.ones(n - 2), [0, 0])])
    start = time.perf_counter()
    y = sla.solve_banded((0, 2), R, (-1 / (n ** 4)) * np.ones(n))
    x = sla.solve_banded((2, 0), RT, y)
    end = time.perf_counter()
    return x, end - start

#1(f)
def prefactored_sparse(n):
    R = diags(np.array(
        [np.append([2], np.ones(n - 1)),
         -2 * np.ones(n - 1), np.ones(n - 2)]), (0, 1, 2), [n, n], format='csr')
    start = time.perf_counter()
    y = spsolve_triangular(R, (-1 / (n ** 4)) * np.ones(n), lower=False)
    x = spsolve_triangular(scipy.sparse.csr_matrix(R.T), y, lower=True)
    end = time.perf_counter()
    return x, end - start

#1(g)
def cholesky(n):
    A = diags(np.array(
        [np.ones(n - 2), np.append(-4 * np.ones(n - 2), [-2]), np.append(9, np.append(6 * np.ones(n - 3), [5, 1])),
         np.append(-4 * np.ones(n - 2), [-2]), np.ones(n - 2)]), (-2, -1, 0, 1, 2), [n, n]).toarray()
    start = time.perf_counter()
    c, low = sla.cho_factor(A)
    x = sla.cho_solve((c, low), (-1 / (n ** 4)) * np.ones(n))
    end = time.perf_counter()
    return x, end - start

#1(h)
def print_table(n: list):
    row = []
    for item in n:
        row.append([item])
        row[-1].append(str(format(LU(item)[1], '.6f')) + "ms")
        row[-1].append(str(format(LU_band(item)[1], '.6f')) + "ms")
        row[-1].append(str(format(LU_sparse(item)[1], '.6f')) + "ms")
        row[-1].append(str(format(prefactored(item)[1], '.6f')) + "ms")
        row[-1].append(str(format(prefactored_band(item)[1], '.6f')) + "ms")
        row[-1].append(str(format(prefactored_sparse(item)[1], '.6f')) + "ms")
        row[-1].append(str(format(cholesky(item)[1], '.6f')) + "ms")
    print("  n |      LU     |   banded    |  sparse LU  |      R      |  banded R   |   sparse R  |     chol")
    for itm in row:
        print(itm)
def get_true_sol(n):
    """
    returns the true solution of the continuous model on the mesh,
    x_i = i / n , i=1,2,...,n.
    """
    x = np.linspace(1/n,1,n)
    d = (1/24)*(-(1-x)**4 + 4*(1-x) -3)
    return d



def compare_to_true(d):
    """
    produces plot similar to the handout,
    the input is the solution to the n x n banded linear system,
    this is one way to visually check if your code is correct.
    """
    dtrue = get_true_sol(100) #use large enough n to make plot look smooth

    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 14})
    plt.title("Horizontal Cantilevered Bar")
    plt.xlabel("x")
    plt.ylabel("d(x)")

    xtrue = np.linspace(1/100,1,100)
    plt.plot(xtrue,dtrue,'k')

    n = len(d)
    x = np.linspace(0,1,n+1)
    plt.plot(x,np.hstack([0,d]),'--r')

    plt.legend(['exact',str(n)])
    plt.grid()
    plt.show()

def print_relatively_error(min_n, max_n):
    """
    minimum n = 2^(min_n)
    maximum n = 2^(max_n)
    """
    n_lst = []
    lu_sol = []
    prefactored_sol = []
    for i in range(min_n, max_n + 1):
        n = 2 ** i
        n_lst.append(n)
        true_sol= get_true_sol(n)
        lu = LU_band(n)[0]
        pre = prefactored_band(n)[0]
        lu_sol.append(abs((np.linalg.norm(lu, np.inf) - np.linalg.norm(true_sol, np.inf)) / np.linalg.norm(true_sol, np.inf)))
        prefactored_sol.append(abs((np.linalg.norm(pre, np.inf) - np.linalg.norm(true_sol, np.inf)) / np.linalg.norm(true_sol, np.inf)))
    plt.loglog(n_lst, lu_sol, 'r', n_lst, prefactored_sol, 'b--')
    plt.xlabel("n")
    plt.ylabel("relative error")
    plt.legend(("LU-band", "Prefactored-Band"))
    plt.savefig("a2-report.png")
    plt.show()






if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    # d = np.zeros(8)
    # compare_to_true(d)

    print_table([200, 400, 600, 800, 1200, 1400, 1600])
    print_relatively_error(4, 16)
