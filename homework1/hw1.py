#CSC336 - Homework 1 - starter code

#Q1a
def int2bin(x):
    """
    convert integer x into a binary bit string

    >>> int2bin(0)
    '0'
    >>> int2bin(10)
    '1010'
    """
    return bin(x)[2:]

#Q1b
def frac2bin(x):
    """
    convert x into its fractional binary representation.

    precondition: 0 <= x < 1

    >>> frac2bin(0.75)
    '0.11'
    >>> frac2bin(0.625)
    '0.101'
    >>> frac2bin(0.1)
    '0.0001100110011001100110011001100110011001100110011001101'
    >>> frac2bin(0.0)
    '0.0'
    """
    if x == 0.0:
        return "0.0"
    binrep = "0."
    while x != 0:
        x *= 2
        if x >= 1:
            binrep += '1'
            x -= 1
        else:
            binrep += '0'
    return binrep

#Question 3
#set these to the values you have chosen as your answers to
#this question. Make sure they aren't just zero and they are
#in the interval (0,1).
x1 = 0.00000000000000000000000000000000000000000000000000000000000000001
x2 = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001


#Question 4
import numpy as np

#list of floating point data types in numpy for reference
fls = [np.float16,np.float32,np.float64,np.float128]

#fix the following function so that it is correct
##import math #you can remove this import once you
            #have the correct solution
def eval_with_precision(x,y,fl=np.float64):
    """
    evaluate sin((x/y)**2 + 2**2) + 0.1, ensuring that ALL
    calculations are correctly using the
    floating point type fl

    precondition: y != 0

    >>> x = eval_with_precision(2,10,fl=fls[0])
    >>> type(x) == fls[0]
    True
    >>> x == fls[0](-0.6816)
    True
    """
    a = fl(np.add(fl(np.power(fl(np.divide(fl(x), fl(y))), fl(2))), fl(np.power(fl(2), fl(2)))))
    return fl(np.add(fl(np.sin(a)), np.divide(fl(1),fl(10))))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #you can add any additional testing you do here
