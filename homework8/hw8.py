#CSC 336 Summer 2020 HW8 starter code

import numpy as np
import matplotlib.pyplot as plt

########
#Q1 code
########

#the random data
ts = np.linspace(0,10,11)
ys = np.random.uniform(-1,1,ts.shape)

#points to evaluate the spline at
xs = np.linspace(ts[0],ts[-1],201)

#implement this function
def qintrp_coeffs(ts,ys,c1=0):
    """
    return the coefficents for the quadratic interpolant,
    as specified in the Week 12 worksheet. The coefficients
    should be returned as an array with 3 columns and 1 row
    for each subinterval, so the i'th row contains a_i,b_i,c_i.

    ts are the interpolation points and ys contains the
    data values at each interpolation point

    c1 is the value chosen for c1, default is 0
    """
    #your code here to solve for the coefficients
    a = np.array(ys[:-1])
    b = np.array((ys[1:] - a) / (ts[1:] - ts[:-1]))
    c = [c1]
    for i in range(1, a.shape[0]):
        c.append((c[i-1] * (ts[i] - ts[i-1]) + b[i-1] - b[i]) / (ts[i] - ts[i+1]))

    #you may find it useful to form a,b, and c in arrays, then
    #return np.stack([a,b,c]).T
    return np.stack([a,b,c]).T

#provided code to evaluate the quadratic spline
def qintrp(coeffs,h,xs):
    """
    Evaluates and returns the quadratic interpolant determined by the
    coeffs array, as returned by qintrp_coeffs, at the points
    in xs.

    h is the uniform space between the knots.

    assumes that each xs is between 0 and h*len(coeffs)
    """
    y = []
    for x in xs:
        i = int(x // h) #get which subinterval we are in
        if i == len(coeffs): #properly handle last point
            i = i-1
        C = coeffs[i]
        ytmp = C[-1]*(x-(i+1)*h)
        ytmp = (x-i*h)*(ytmp + C[-2])
        ytmp += C[0]
        y.append(ytmp)
    return np.array(y)

#define any additional functions for Q1 here

########################

#define any functions for Q2 and Q3 here
def get_approx(coeff, xss):
    ii = 0
    result = 0
    for item in coeff:
        result += coeff[ii] * xss ** ii
        ii += 1
    return result
    # return coef[0] * xs ** 0 + coef[1] * xs ** 1 + coef[2] * xs ** 2 + coef[3] * xs ** 3

def get_derivative(coefff, xsss):
    ii = 0
    result = 0
    for item in coefff[:-1]:
        result += (ii + 1) * coefff[ii + 1] * xsss ** ii
        ii += 1
    return result
    # return coefff[1] * xsss ** 0 + 2 * coefff[2] * xsss ** 1 + 3 * coefff[3] * xsss ** 2

def get_second_derivative(ccoef, xxs):
    ii = 0
    result = 0
    for item in ccoef[:-2]:
        result += np.math.factorial(ii + 2) * ccoef[ii + 2] * xxs ** ii
        ii += 1
    return result

if __name__ == '__main__':
    # Q1
    # add any code here calling the functions you defined above

    coef = qintrp_coeffs(ts, ys)
    y = qintrp(coef, 1, xs)

    plt.plot(ts, ys, '*k', xs, y)
    plt.legend(("actual value", "quadratic spline"))
    plt.show()

    # b
    plt.plot(ts, ys, '*k', xs, y)

    ys[0] += 0.5
    coef = qintrp_coeffs(ts, ys)
    y = qintrp(coef, 1, xs)
    plt.plot(ts, ys, '*k', xs, y)
    plt.legend(("actual value", "quadratic spline"))
    plt.show()

    # Q2
    t = np.array([[1, -2, 4, -8, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 2, 0, 0, 0, -2, 0],
                  [0, 0, 2, -12, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 6]])
    y = np.array([-27, -1, -1, 0, 0, 0, 0, 0])
    coef = np.linalg.solve(t, y)

    x_val = []
    x_val_2 = []
    approx = []
    derivative = []
    sec_derivative = []
    for i in np.arange(-2, 0, 0.001):
        x_val.append(i)
    approx.extend(get_approx(coef[:4], np.array(x_val)))
    derivative.extend((get_derivative(coef[:4], np.array(x_val))))
    sec_derivative.extend((get_second_derivative(coef[:4], np.array(x_val))))
    for i in np.arange(0, 1, 0.001):
        x_val_2.append(i)
    approx.extend(get_approx(coef[4:], np.array(x_val_2)))
    derivative.extend(get_derivative(coef[4:], np.array(x_val_2)))
    sec_derivative.extend(get_second_derivative(coef[4:], np.array(x_val_2)))
    plt.plot(x_val + x_val_2, approx)
    plt.plot([-2, 0, 1], [-27, -1, 0], '*k')
    plt.legend(("spline", "original data points"))
    plt.show()

    plt.plot(x_val + x_val_2, derivative)
    plt.plot(x_val + x_val_2, sec_derivative)
    plt.plot(x_val + x_val_2, np.zeros(3000), "--g")
    plt.legend(("first derivative", "second derivative"))
    plt.show()

    # q3
    tt = np.array([[1,-2, 4], [1, 0, 0], [1, 1, 1]])
    yy = np.array([-27, -1, 0])
    quadratic_poly_coef = np.linalg.solve(tt, yy)

    t = np.array([[1, -2, 4, -8, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 2, 0, 0, 0, -2, 0],
                  [0, 1, -4, 12, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 2, 6]])

    quadratic_poly = get_approx(quadratic_poly_coef, np.array(x_val + x_val_2))
    quadratic_poly_first_d = get_derivative(quadratic_poly_coef, np.array(x_val + x_val_2))

    y = np.array([-27, -1, -1, 0, 0, 0, quadratic_poly_first_d[0], quadratic_poly_first_d[-1]])
    coef = np.linalg.solve(t, y)

    first_derivative = list(get_derivative(coef[:4], np.array(x_val))) + list(get_derivative(coef[4:], np.array(x_val_2)))

    plt.plot(x_val + x_val_2, first_derivative)
    plt.plot(x_val + x_val_2, quadratic_poly_first_d, "--")
    plt.legend(("first derivative Q3", "quadratic poly first derivative"))
    plt.show()

    approximation = list(get_approx(coef[:4], np.array(x_val))) + list(get_approx(coef[4:], np.array(x_val_2)))

    plt.plot(x_val + x_val_2, approximation)
    plt.plot(x_val + x_val_2, get_approx(quadratic_poly_coef, np.array(x_val + x_val_2)), "--")
    plt.plot([-2, 0, 1], [-27, -1, 0], '*k')
    plt.legend(("clamped cubic spline", "quadratic poly", "original data points"))
    plt.show()

    plt.plot(x_val + x_val_2, approximation)
    plt.plot(x_val + x_val_2, approx)
    plt.plot([-2, 0, 1], [-27, -1, 0], '*k')
    plt.legend(("clamped cubic spline", "natural cubic spline", "original data points"))
    plt.show()

    plt.plot(x_val + x_val_2, first_derivative)
    plt.plot(x_val + x_val_2, derivative)
    plt.legend(("first derivative Q3", "first derivative Q2"))
    plt.show()

    second_derivative = list(get_second_derivative(coef[:4], np.array(x_val))) + list(get_second_derivative(coef[4:], np.array(x_val_2)))
    plt.plot(x_val + x_val_2, second_derivative)
    plt.plot(x_val + x_val_2, sec_derivative)
    plt.legend(("second derivative Q3", "second derivative Q2"))
    plt.show()



