#CSC336 Assignment #1 starter code
import numpy as np

#Q2a
def alt_harmonic(fl=np.float16):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series.

    The floating point type fl is used in the calculations.
    """
    ssum = 0
    n = 1
    while 1:
        old_sum = ssum
        cur_term = fl(np.divide(np.power(-1, n + 1), n))
        ssum = fl(np.add(ssum, cur_term))
        if np.equal(ssum, old_sum):
            return [n, ssum]
        n += 1

#Q2b
#add code here as stated in assignment handout
cal = alt_harmonic()[1]
q2b_rel_error = np.divide(np.subtract(np.log(2), cal), np.log(2))

#Q2c
def alt_harmonic_given_m(m, fl=np.float16):
    """
    Returns the sum of the first m terms of the alternating
    harmonic series. The sum is performed in an appropriate
    order so as to reduce rounding error.

    The floating point type fl is used in the calculations.
    """
    ssum = 0
    n = 1
    while n <= m:
        old_sum = ssum
        cur_term = np.divide(np.power(-1, n + 1), n)
        ssum = np.add(ssum, cur_term)
        n += 1
    return fl(ssum)


#Q3a
def alt_harmonic_v2(fl=np.float32):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series (using
    the formula in Q3a, where terms are paired).

    The floating point type fl is used in the calculations.
    """
    ssum = 0
    n = 1
    while 1:
        old_sum = ssum
        cur_term = fl(np.divide(1, 2 * n * (2 * n - 1)))
        ssum = fl(np.add(ssum, cur_term))
        if np.equal(ssum, old_sum):
            return [n, ssum]
        n += 1
        

#Q3b
#add code here as stated in assignment handout
cal = alt_harmonic_v2()[1]
q3b_rel_error = np.divide(np.subtract(np.log(2), cal), np.log(2))

#Q4b
def hypot(a, b):
    """
    Returns the hypotenuse, given sides a and b.
    """
    c = np.sqrt(a + b + np.sqrt(2) * np.sqrt(a) * np.sqrt(b))\
        * np.sqrt(a + b - np.sqrt(2) * np.sqrt(a) * np.sqrt(b)) #replace with improved algorithm
    return c

#Q4c
q4c_input = np.float16([1000]) #see handout for what value should go here.


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    pass
