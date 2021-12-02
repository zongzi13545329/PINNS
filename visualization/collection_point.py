"""This the visualization of train data"""

import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt("../examples/train.dat", skiprows=1, unpack=True)

x_1 = data[0]
x_2 = data[1]

fig,ax = plt.subplots()

plt.plot(x_1[:999],x_2[:999], '*', color = 'red', markersize = 5, label = 'Boundary collocation points= 1000')
plt.plot(x_1[1000:],x_2[1000:], 'o', markersize = 0.5, label = 'PDE collocation points = 3020')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Collocation points')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('scaled')
plt.show()

fig.savefig('collocation_points_Helmholtz.png', dpi = 500)
