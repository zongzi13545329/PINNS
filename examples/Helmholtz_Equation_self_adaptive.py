"""Backend supported:  pytorch
    We follew the setting in https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/tree/main/TensorFlow/Helmholtz%20Equation
"""
import sys
sys.path.append("..")
import deepxde as dde
import numpy as np



def gen_testdata():
    x_1 = np.linspace(-1,1,256)  # 256 points between -1 and 1 [256x1]
    x_2 = np.linspace(1,-1,256)  # 256 points between 1 and -1 [256x1]
    xx, tt = np.meshgrid(x_1,x_2) 
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    a_1 = 1 
    a_2 = 1
    y = np.sin(a_1 * np.pi * xx) * np.sin(a_2 * np.pi * tt) 
    y = y.reshape(-1,1) 
    return X, y


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=10000, num_boundary=400, num_initial=600,  train_distribution="LHS",
)
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

## self-adaptive
model.compile("adam", lr=1.0e-3)
model.train(epochs=10000)
model.compile("L-BFGS")
model.train()

X = geomtime.random_points(100000)
err = 1
while err > 0.005:
    f = model.predict(X, operator=pde)
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    x_id = np.argmax(err_eq)
    print("Adding new point:", X[x_id], "\n")
    data.add_anchors(X[x_id])
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)

model.compile("adam", lr=1e-3)
model.train(epochs=10000, disregard_previous_best=True, callbacks=[early_stopping])
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))