#CSC 336 Summer 2020 A3Q2 starter code

#Note: you may use the provided code or write your own, it is your choice.

#some general imports
import time
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt


def compute_inverse(A, atol = 1e-11):
    xo = np.divide(A.T, np.dot(np.linalg.norm(A, 1), np.linalg.norm(A, np.inf)))
    x = xo
    err = 1
    while (err > atol):
        x = np.add(x, np.dot(x, np.subtract(np.identity(xo.shape[0]), np.dot(A, x))))
        err = np.linalg.norm(np.subtract(np.identity(xo.shape[0]), np.dot(A, x)))
    return x


if __name__ == "__main__":
    runtime_new = []
    runtime_la = []
    acc_new = []
    acc_la = []
    for i in range(1, 10):
        A = np.random.randint(25, size=(2 ** i, 2 ** i))
        while np.linalg.det(A) == 0:
            A = np.random.randint(10, size=(2**i, 2**i))

        start = time.perf_counter()
        A_inv = compute_inverse(A)
        end = time.perf_counter()
        norm = np.linalg.norm(np.subtract(np.identity(2**i), np.dot(A, A_inv)))
        runtime_new.append(end - start)
        acc_new.append(norm)

        start_la = time.perf_counter()
        A_inv_la = LA.inv(A)
        end_la = time.perf_counter()
        norm_LA = np.linalg.norm(np.subtract(np.identity(2 ** i), np.dot(A, A_inv_la)))
        runtime_la.append(end_la - start_la)
        acc_la.append(norm_LA)

        print("matrix size: " + str(2**i) + "x" + str(2**i) + " , LA.inv runtime: " + str(format(end_la - start_la, '.4e')) + " ; Newton's runtime: " + str(format(end - start, '.4e'))+
               " , LA.inv residual norm: " + str(format(norm_LA, '.4e')) + " ; Newton's residual norm: " + str(format(norm, '.4e')))

    ran = [2**i for i in range(1, 10)]

    plt.plot(ran, runtime_new, 'r', ran, runtime_la, 'g')
    plt.xlabel("matrix size")
    plt.ylabel("time")
    plt.legend(("newton's method", "LA"))
    plt.show()

    plt.plot(ran, acc_new, 'r', ran, acc_la, 'g')
    plt.xlabel("matrix size")
    plt.ylabel("norm of residual")
    plt.legend(("newton's method", "LA"))
    plt.show()

