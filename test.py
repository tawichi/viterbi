from turtle import color
import numpy as np
import csv
import operator
import matplotlib.pyplot as plt
import math
from scipy import special


def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))




##理論上界
x = 6 #x = SNR
p_b = 0.0

p = 1 / 2 * special.erfc(np.sqrt(1 / 2 * 10**(x/10)))
       
for k in range(17):
    if( k%2 == 0):
        for e in range(k // 2 + 1,k):
            p_b +=  combinations_count(k,e) * (p** e) * ((1 - p) ** (k - e)) + 0.5 * combinations_count(k,k//2) * ( p ** (2//k)) * ((1 - p) ** (k // 2))
    else:
        for e in range((k + 1) // 2, k):
            p_b +=combinations_count(k,e) * (p **(e)) * ((1 - p) ** (k - e))
plt.plot(x, p_b)
plt.show()