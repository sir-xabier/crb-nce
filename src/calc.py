import matplotlib.pyplot as plt
import numpy as np

j=[0.7651976865, 0.4400505857]

for i in range(1,20):
    j.append(2*i*j[-1] - j[-2])

plt.plot(j)
plt.show()