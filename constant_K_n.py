'''
Compute and plot the constant K_n in theorem 4

'''

import numpy as np
import matplotlib.pyplot as plt
import math

def S_n(n):
    n_ = n-1
    volume_n_minus_1 = (math.pi**(n_/2.))/math.gamma(n_/2. + 1.)
    surface_area_n = 2*math.pi* volume_n_minus_1
    return surface_area_n

def get_constant(n):
    s=1.
    C = math.pi/2.
    if n%2.==0:
        s = 2.
        C = 1.
    for i in np.arange(s, n,2):
        C *= float(i)/(i+1.)
    return C

n = np.arange(1,200)
K_n = []
for n_i in n:
    val = 2.*get_constant(n_i)* S_n(n_i-1)/S_n(n_i)
    val = "{:.5f}".format(val)
    K_n.append(val)
plt.plot(n,K_n)
plt.show()
print(K_n)