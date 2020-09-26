#CSC336 Summer 2020 HW6 starter code

#some basic imports
import numpy as np
import numpy.linalg as LA
from scipy.linalg import solve
import matplotlib.pyplot as plt

#Question 1 [autotested]

#Complete the following 5 functions that compute the Jacobians
#of the 5 nonlinear systems in Heath Exercise 5.9

#NOTE: For this question, assume x.shape == (2,),
#      so you can access components of
#      x as x[0] and x[1].

def jac_a(x):
    return np.array([[2 * x[0],2 * x[1]],
                     [2 * x[0], -1] ])

def jac_b(x):
    return np.array([[2 * x[0] + x[1] ** 3,3 * (x[1] ** 2) * x[0]],
                     [6 * x[0] * x[1],-3 * (x[1] ** 2) + 3 * (x[0] ** 2)] ])

def jac_c(x):
    return np.array([[1 - 2 * x[1],1 - 2 * x[0]],
                     [2 * x[0] - 2, 2 * x[1] + 2] ])

def jac_d(x):
    return np.array([[3 * (x[0] ** 2), - 2 * x[1]],
                     [1 + 2 * x[0] * x[1], x[0] ** 2] ])

def jac_e(x):
    return np.array([[2 * np.cos(x[0]) - 5, - np.sin(x[1])],
                     [-4 * np.sin(x[0]), 2 * np.cos(x[1]) - 5] ])

########################################
#Question 2

#NOTE: You may use the provided code below or write your own for Q2,
#it is up to you.

#NOTE: For this question, if you use the provided code, it assumes
#      that x.shape == (3,1),
#      so you need to access components of
#      x as x[0,0], x[1,0], and x[2,0]


#useful for checking convergence behaviour of the fixed point method
from scipy.linalg import eigvals
def spectral_radius(A):
    return np.max(np.abs(eigvals(A)))

#This is essentially the same as estimate_convergence_rate from HW5,
#but for non-scalar x values.
def conv_info(xs):
    """
    Returns approximate values of the convergence rate r,
    constant C, and array of error estimates, for the given
    sequence, xs, of x values.

    Note: xs should be an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.

    This code uses the infinity norm for the norm of the errors.
    """

    errs = []
    nxs = np.diff(np.array(xs),axis=0)

    for row in nxs:
        errs.append(LA.norm(row,np.inf))

    r = np.log(errs[-1]/errs[-2]) / np.log(errs[-2]/errs[-3])
    c = errs[-1]/(errs[-2]**r)

    return r,c,errs

#functions for doing root finding
def fixed_point(g,x0,atol = 1e-14):
    """
    Simple implementation of a fixed point iteration for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    The fixed point iteration is x_{k+1} = g(x_k).

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    x = x0
    xs = [x0]
    err = 1
    while(err > atol):
        x = g(x)
        xs.append(x)
        err = LA.norm(xs[-2]-xs[-1],np.inf)
    return np.array(xs)

def newton(f,J,x0,atol = 1e-14):
    """
    Simple implementation of Newton's method for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    f is the function and J computes the Jacobian.

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    x = x0
    xs = [x0]
    err = 1
    while(err > atol):
        x = x - solve(J(x),f(x))
        xs.append(x)
        err = LA.norm(xs[-2]-xs[-1],np.inf)
    return np.array(xs)

def broyden(f,x0,atol = 1e-14):
    """
    Simple implementation of Broyden's method for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    f is the function to find the root for.

    Initially, the approximate Jacobian is set to the identity matrix.

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    B = np.identity(len(x0))
    x = x0
    xs = [x0]
    err = 1
    fx = f(x)
    while(err > atol):
        s = solve(B,-fx)
        x = x + s
        xs.append(x)
        next_fx = f(x)
        y = next_fx - fx
        #update approximate Jacobian
        B = B + (( y - B @ s) @ s.T) / (s.T @ s)
        fx = next_fx
        err = LA.norm(xs[-2]-xs[-1],np.inf)
    return np.array(xs)

#the function g from the question description,
#(i.e. x = g(x) is the fixed-point iteration)
def g(x):
    cos,sin = np.cos(x),np.sin(x)
    return np.array([[-cos[0][0]/81 + (x[1][0] ** 2) / 9 + sin[2][0] / 3], [sin[0][0] / 3 + cos[2][0] / 3], [-cos[0][0] / 9 + x[1][0] / 3 + sin[2][0] / 6]])

#computes the jacobian of g(x)
def getG(x):
    cos,sin = np.cos(x),np.sin(x)
    return np.array([[sin[0][0]/81, 2 * x[1][0] / 9, cos[2][0]/3], [cos[0][0] / 3, 0, -sin[2][0]/3], [sin[0][0]/9, 1/3, cos[2][0]/6]])

#x = g(x) rewritten in the form f(x), pass this into broyden / newton code
def f(x):
    return x - g(x) #pass #your code here

#computes the jacobian of f(x)
def jac(x):
    return np.subtract(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), getG(x)) #your code here

#useful to verify your jacobian is correct
def check_jac(my_f,my_J,x,verbose=True):
    """
    Returns True if my_J(x) is close to the true jacobian of my_f(x)
    """
    J = my_J(x)
    J_CS = np.zeros([len(x),len(x)])
    for i in range(len(x)):
        y = np.array(x,np.complex)
        y[i] = y[i] + 1e-10j
        J_CS[:,i] = my_f(y)[:,0].imag / 1e-10
    if not np.allclose(J,J_CS):
        if verbose:
            print("Jacobian doesn't match - check your function "
                  "and your Jacobian approximation")
            print("The difference between your Jacobian"
                  " and the complex step Jacobian was\n")
            print(J-J_CS)
        return False
    return True

########################################
#Question 3

from scipy.optimize import fsolve
import time

#define any functions you'll use for Q3 here, but call
#them in the main block
def fun(x):
    return [np.sin(x[0]) + x[1] ** 2 + np.log(x[2]) - 3, 3 * x[0] + 2 ** x[1] - x[2] ** 3, x[0] ** 2 + x[1] ** 2 + x[2] ** 3 - 6]

def naive_search():
    start = time.perf_counter()
    root = []
    i = 0
    while len(root) < 4:
        x = np.random.uniform(-2, 2, 3)
        sol = list(fsolve(fun, x))
        if np.allclose(fun(sol), [0.0, 0.0, 0.0]) and all([not np.allclose(sol, item) for item in root]):
            root.append(sol)
        i += 1
    end = time.perf_counter()
    return root, i, end - start

def hun(x, curr_root):
    lo = 1
    for item in curr_root:
        lo *= np.linalg.norm(x - item)
    return np.divide(fun(x), lo)

def deflation():
    start = time.perf_counter()
    root = []
    i = 0
    while len(root) < 4:
        x = np.random.uniform(-2, 2, 3)
        sol = list(fsolve(hun, x, args=(root)))
        if np.allclose(fun(sol), [0.0, 0.0, 0.0]) and all([not np.allclose(sol, item) for item in root]):
            root.append(sol)
        i += 1
    end = time.perf_counter()
    return root, i, end - start

def modified_hun(x, curr_root):
    ad = 1
    for item in curr_root:
        ad *= 1 + (1 / np.linalg.norm(x - item))
    return np.multiply(fun(x), ad)

def modified_deflation():
    start = time.perf_counter()
    root = []
    i = 0
    while len(root) < 4:
        x = np.random.uniform(-2, 2, 3)
        sol = list(fsolve(modified_hun, x, args=(root)))
        if np.allclose(fun(sol), [0.0, 0.0, 0.0]) and all([not np.allclose(sol, item) for item in root]):
            root.append(sol)
        i += 1
    end = time.perf_counter()
    return root, i, end - start



########################################

if __name__ == '__main__':
    #import doctest
    #doctest.testmod()

    #import any other non-standard modules here
    #and run any code for generating your answers here too.

    #Any code calling functions you defined for Q2:

    #here are some sample bits of code you might have if using the
    #provided code:
    if True:
        print("Jac check: ",check_jac(f,jac,np.ones([3,1])))
        xo = np.array([[3],[4],[1]]) #arbitary starting point chosen
        xs = fixed_point(g,xo)
        r,c,errs = conv_info(xs)
        print(r,c,'fixed point') #make sure to update to format any output neatly
        print(xs)



        xs_newton = newton(f,jac, xo)
        r_newton,c_newton,errs_newton = conv_info(xs_newton)
        print(r_newton, c_newton, 'newton\'s method')
        print(len(xs_newton))

        xs_b = broyden(f, xo)
        r_b,c_b,errs_b = conv_info(xs_b)
        print(r_b, c_b, 'broyden\'s method')
        print(len(xs_b))

        plt.semilogy(errs,'*-b', errs_newton, '*-r', errs_b, '*-g') #the "*-" makes it easier to
                                 #see convergence behaviour in plot
        plt.legend(("fixed point", "newton\'s method", "broyden"))
        plt.show() #make sure to add plot labels if you use this code

    ####################################

    #Any code calling functions you defined for Q3:
    num_sample_naive = []
    num_sample_deflated = []
    num_sample_modified = []
    time_naive = []
    time_deflated = []
    time_modified = []
    rng = [i for i in range(100)]
    for i in range(100):
        a, b, c = naive_search()
        num_sample_naive.append(b)
        time_naive.append(c)

        aa, bb, cc = deflation()
        num_sample_deflated.append(bb)
        time_deflated.append(cc)

        aaa, bbb, ccc = modified_deflation()
        num_sample_modified.append(bbb)
        time_modified.append(ccc)

    plt.plot(rng, num_sample_naive, 'r', rng, num_sample_deflated, 'g', rng, num_sample_modified, 'b')
    plt.xlabel("num run")
    plt.ylabel("num sample")
    plt.legend(("naive search", "deflated search", "modified deflated search"))
    plt.show()

    plt.plot(rng, time_naive, 'r', rng, time_deflated, 'g', rng, time_modified, 'b')
    plt.xlabel("num run")
    plt.ylabel("time")
    plt.legend(("naive search", "deflated search", "modified deflated search"))
    plt.show()

    print(np.mean(num_sample_naive))
    print(np.std(num_sample_naive))
    print(np.mean(time_naive))
    print(np.std(time_naive))
    print("------------------------------------\n")

    print(np.mean(num_sample_deflated))
    print(np.std(num_sample_deflated))
    print(np.mean(time_deflated))
    print(np.std(time_deflated))
    print("------------------------------------\n")

    print(np.mean(num_sample_modified))
    print(np.std(num_sample_modified))
    print(np.mean(time_modified))
    print(np.std(time_modified))