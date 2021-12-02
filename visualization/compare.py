"""This the visualization of y_pred and y_test"""

import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("../examples/test.dat", skiprows=1, unpack=True)

y_true = data[2]
y_pred = data[3]

#plot 
fig,ax = plt.subplots()
iters = range(len(y_true))
plt.plot(iters, y_true, label = 'y_true')
plt.plot(iters, y_true, label = 'y_pred')
plt.yscale('symlog')
plt.xlabel('iterations')
plt.ylabel('y')
plt.legend()
plt.show()

fig.savefig('compare.png', dpi = 500, bbox_inches='tight')
