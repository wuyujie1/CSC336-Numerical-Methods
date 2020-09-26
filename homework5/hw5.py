#CSC336 Summer 2020 HW5 starter code

import numpy as np

#question 1
def estimate_convergence_rate(xs):
    """
    Return approximate values of the convergence rate r and
    constant C (see section 5.4 in Heath for definition and
    this week's worksheet for the approach), for the given
    sequence, xs, of x values.

    (You might find np.diff convenient)

    Examples:

    >>> xs = 1+np.array([1,0.5,0.25,0.125,0.0625,0.03125])
    >>> b,c = estimate_convergence_rate(xs)
    >>> close = lambda x,y : np.allclose(x,y,atol=0.1)
    >>> close(b,1)
    True
    >>> close(c,0.5)
    True
    >>> xs = [0, 1.0, 0.7357588823428847, 0.6940422999189153,\
    0.6931475810597714, 0.6931471805600254, 0.6931471805599453]
    >>> b,c = estimate_convergence_rate(xs)
    >>> close(b,2)
    True
    >>> close(c,0.5)
    True
    """
    ek = np.linalg.norm(xs[-2] - xs[-3])
    ekp1 = np.linalg.norm(xs[-1] - xs[-2])
    ekm1 = np.linalg.norm(xs[-3] - xs[-4])

    r = np.log(ekp1) / np.log(ek/ekm1) - np.log(ek) / np.log(ek/ekm1)

    c = ekp1 / (ek ** r)

    return r, c #r,c

#question 2
#Put any functions you define for Q2 here.
#See the worksheet for a similar example if you have
#trouble getting started with Q2.
def g1(x):
    return ((x ** 2) + 2) / 3

def g2(x):
    return np.sqrt(3 * x - 2)

def g3(x):
    return 3 - 2 / x

def g4(x):
    return ((x ** 2) - 2) / (2 * x - 3)

def dg1(x):
    return (2 * x) / 3

def dg2(x):
    return 3 / (2 * np.sqrt(3 * x - 2))

def dg3(x):
    return 2 / (x ** 2)

def dg4(x):
    return (2 * x * (2 * x - 3) - 2 * ((x ** 2) - 2)) / ((2 * x - 3) ** 2)

import matplotlib.pyplot as plt
plt.figure()

gs = [g1, g2, g3, g4]

xs = np.linspace(-4,4)

#plotting the derivatives
dgs = [dg1, dg2, dg3, dg4]
plt.figure()
for dg in dgs:
    ys = dg(xs)
    plt.plot(xs,ys)
plt.plot(xs,np.ones(xs.shape)) #flat line for reference
plt.plot(xs,-np.ones(xs.shape)) #flat line for reference
plt.legend([dg.__name__ for dg in dgs]+['1','-1'])
plt.ylim([-2,2])
plt.grid()
plt.title("Derivatives of g(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#running the fixed point method
atol = 1e-6

def fixed_point(g,xo,atol = 1e-6,maxit = 50): #note I added a maxit for the stop criterion
    x = xo
    err = 1
    xs = [x]
    while (err > atol and err < 1e10 and len(xs) < maxit):
        x = g(x)
        xs.append(x)
        err = np.abs(xs[-1]-xs[-2])
    return xs

xo = [1.5, 1.45, 1.45, 1.85] #can change to check convergence behaviour
for g in gs:
    xs = fixed_point(g,xo[gs.index(g)])
    print("Method " + g.__name__)
    print("i x")
    for i,x in enumerate(xs):
        print(i,x)


#g2
err = 1
x = 1.45
g2x = [1.45]
while (err > atol and err < 1e10 and len(g2x) < 50):
    x = g2(x)
    g2x.append(x)
    err = np.abs(g2x[-1]-g2x[-2])
print(estimate_convergence_rate(g2x))
#g3
err = 1
x = 1.45
g3x = [1.45]
while (err > atol and err < 1e10 and len(g3x) < 50):
    x = g3(x)
    g3x.append(x)
    err = np.abs(g3x[-1]-g3x[-2])
print(estimate_convergence_rate(g3x))
#g4
err = 1
x = 1.85
g4x = [1.85]
while (err > atol and err < 1e10 and len(g4x) < 50):
    x = g4(x)
    g4x.append(x)
    err = np.abs(g4x[-1]-g4x[-2])
print(estimate_convergence_rate(g4x))

#optional question 3 starter code
a = np.pi #value of root
def f(x,m=2):
    return (x - a)**m
def df(x,m=2):
    return m*(x-a)**(m-1)

def newton(f,fp,xo,atol = 1e-10):
    """
    Simple implementation of Newton's method for scalar problems,
    with stopping criterion based on absolute difference between
    values of x on consecutive iterations.

    Returns the sequence of x values.
    """
    x = xo
    xs = [xo]
    err = 1
    while(err > atol):
        x = x - f(x)/fp(x)
        xs.append(x)
        err = np.abs(xs[-2]-xs[-1])
    return xs

def multiplicity_newton(f,fp,xo,atol = 1e-10):
    """
    version of Newton's method that monitors the convergence rate,
    r and/or C, and tries to speedup convergence for multiple roots

    Returns the sequence of x values.
    """
    #modify the Newton's method code below based on your algorithm
    #from Q3b
    x = xo
    xs = [xo]
    err = 1
    while(err > atol):
        x = x - f(x)/fp(x)
        xs.append(x)
        err = np.abs(xs[-2]-xs[-1])
    return xs

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #call any code related to Q2 here

    #optional code if you try Q3c
    if False:
        xo = 0
        print("error orig, error alt, iters orig, iters alt")
        for m in range(2,8):
            xs = multiplicity_newton(lambda x: f(x,m=m),lambda x: df(x,m=m),xo)
            iters_alt = len(xs)
            xalt = xs[-1]
            xs = newton(lambda x: f(x,m=m),lambda x: df(x,m=m),xo)
            iters_orig = len(xs)
            x = xs[-1]
            print(f"{x-a:<10.2e},{xalt-a:<10.2e},{iters_orig:<11d},"
                    f"{iters_alt:<10d}")