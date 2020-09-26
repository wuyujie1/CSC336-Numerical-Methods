#CSC 336 HW#3 starter code

import scipy.linalg as sla
import numpy as np
import numpy.linalg as LA

#Q1 assign values to the following variables as
#specified in the handout

A = np.array([[21.0, 67.0, 88.0, 73.0], [76.0, 63.0, 7.0, 20.0], [0.0, 85.0, 56.0, 54.0], [19.3, 43.0, 30.2, 29.4]])
B = np.array([[1, 2, 3, 4], [12, 13, 14, 5], [11, 16, 15, 6], [10, 9, 8, 7]])
C = B.T
b = np.array([141.0, 109.0, 218.0, 93.7])
x = sla.solve(A, b)
r = np.dot(A, x) - b
Ainv = LA.inv(A)
c1 = LA.cond(A, 1)
c1_2 = float(np.dot(LA.norm(A, 1), LA.norm(Ainv, 1)))
cinf = LA.cond(A, np.inf)
A32 = np.array(A, dtype=np.float32)
b32 = np.array(b, dtype=np.float32)
x32 = np.array(sla.solve(A32, b32), dtype=np.float32)
y = np.dot(np.dot(np.dot(LA.inv(B), (2 * A + np.identity(4))),  (LA.inv(C) + A)), b)

#Q2 Hilbert matrix question
#your code here
cond = []
rel_err = []
for i in range(2, 8):
    H = sla.hilbert(i)
    cond.append(LA.cond(H, np.inf))
    xxx = np.ones(i)
    bbb = np.dot(H, xxx)
    cal_x = sla.solve(H, bbb)
    rel_err.append(np.divide(LA.norm(cal_x, np.inf) - LA.norm(xxx, np.inf), LA.norm(xxx, np.inf)))

print("   Results for float64\n")
print(" n |  rel err |  cond(H)\n")
print("------------------------")
for j in range(2, 8):
    print(" " + str(j) + " | " + str(format(rel_err[j - 2], ".2e")) + " | " + str(format(cond[j - 2], ".2e")) + "\n")


#Q3c
#provided code for gaussian elimination (implements algorithm from the GE notes)
def ge(A,b):
    for k in range(A.shape[0]-1):
        for i in range(k+1, A.shape[1]):
            if A[k,k] != 0:
                A[i,k] /= A[k,k]
            else:
                return False
            A[i,k+1:] -= A[i,k] * A[k,k+1:]
            b[i] = b[i] - A[i,k]*b[k]
    return True

def bs(A,b):
    x = np.zeros(b.shape)
    x[:] = b[:]
    for i in range(A.shape[0]-1,-1,-1):
        for j in range(i+1,A.shape[0]):
            x[i] -= A[i,j]*x[j]
        if A[i,i] != 0:
            x[i] /= A[i,i]
        else:
            return None
    return x

def ge_solve(A,b):
    if ge(A,b):
        return bs(A,b)
    else:
        return None #GE failed

def solve(eps):
    """
    return the solution of [ eps 1 ] [x1]   [1 + eps]
                           |       | |  | = |       |
                           [ 1   1 ] [x2]   [   2   ]
    The solution is obtained using GE without pivoting
    and back substitution. (the provided ge_solve above)
    """
    a = np.array([[eps, 1], [1, 1]])
    bb = np.array([1 + eps, 2])
    return ge_solve(a, bb)


#Q3d code here for generating your table of values
e = 0.1
xs = []
relative_err = []
for e in range(1, 13):
    ep = 10 ** (- e)
    xx = solve(ep)
    xs.append(xx)
    relative_err.append(np.divide(LA.norm(np.subtract(xx, np.array([1, 1])), np.inf), LA.norm(np.array([1, 1]), np.inf)))


print("  eps   |  rel err")
print("--------------------")
for i in range(len(xs)):
    print(str(format(10 ** (-i - 1), ".1e")) + " | " + str(format(relative_err[i], ".4e")))
    print("--------------------")


