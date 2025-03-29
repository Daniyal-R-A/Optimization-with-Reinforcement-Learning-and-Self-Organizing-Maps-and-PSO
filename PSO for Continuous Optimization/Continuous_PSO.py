import random
import copy
import math
import numpy as np

class Continuous_PSO():
    def __init__(self, func, xlim, ylim):
        self.func = func
        self.xlim = xlim #(a,b): a < x < b 
        self.ylim = ylim #(c,d): d < y < d


if __name__ == '__main__':

    # Define the Rosenbrock function
    def function1(x, y):
        return 100 * (x**2 - y)**2 + (1 - x)**2

    # Define the 2d- Greiwank’s function
    def function2(x, y):
        return 1 + (x**2 / 4000) + (y**2 / 4000) - math.cos(x) * math.cos(y / math.sqrt(2))

    xlim = (-2, 2)
    ylim = (-1, 3)    
    solver1 = Continuous_PSO(function1, xlim, ylim)

    # xlim = (-30, 30)
    # ylim = (-30, 30)    
    # solver2 = Continuous_PSO(function2, xlim, ylim)

__