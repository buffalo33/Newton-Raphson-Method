#!/usr/bin/python3
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


# Compute the jacobian's matrix of a function at the point x
def Jacobian(f,x,h):
    m = []
    for j in range(x.shape[0]):
        y = np.copy(x)
        y[j][0] += h
        m.append(np.asarray(np.transpose((f(y) - f(x))/h))[0])
    return np.transpose(np.matrix(m))

if __name__ == '__main__':
    def f(x):
        return x

    def g(x):
        return np.matrix([[x[0, 0]],[x[0, 0]+2*x[1, 0]], [x[0, 0]+2*x[1, 0]+3*x[2, 0]]])

    def h_func(x):
        return np.matrix([[3*x[0,0]*x[1,0]], [x[0,0]*x[0,0]], [4*x[0,0]+x[1,0]*x[0,0]]])

    # Jacobian's matrix for function h
    def H_h_func(x):
        return np.matrix([[3*x[1,0], 3*x[0,0]], [2*x[0,0], 0.], [4+x[1,0], x[0,0]]])

    def rel_gap(f, H_f, x, y, h):
        z = np.matrix([[x],[y]])
        return np.linalg.norm(Jacobian(f, z, h) - H_f(z))/np.linalg.norm(H_f(z))
    
    x = np.matrix([[1.], [2.], [3.]])
    #print("\nH_identity(1,2,3) =\n", Jacobian(f, x, 0.000001), "\nexpected result :\n", np.matrix([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]))
    #print("\nH_g =\n", Jacobian(g, x, 0.000001), "\nexpected result :\n", np.matrix([[1., 0. ,0.], [1., 2., 0.], [1., 2., 3.]]))
    #print("\nH_h =\n", Jacobian(h, x, 0.000001), "\nexpected result :\n", np.matrix([[0., 9., 6.], [2., 0., 0.], [4., 3., 2.]]))


    # make these smaller to increase the resolution
    dx, dy= 0.1, 0.1
    h = 0.000001

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(0, 10 + dy, dy),
                    slice(0, 10 + dx, dx)]
    
    z = np.zeros(x.shape)

    for i in range(0, 100):
        for j in range(0, 100):
            x0 = np.matrix([[x[i,j]], [y[i,j]]])
            z[i,j] = np.linalg.norm(Jacobian(h_func, x0, h) - H_h_func(x0))/np.linalg.norm(H_h_func(x0))

    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

    
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax1 = plt.subplots(nrows=1)

    """
    im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax0)
    ax0.set_title('pcolormesh with levels')
    """
    
    # contours are *point* based plots, so convert our bound into point
    # centers
    cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                      y[:-1, :-1] + dy/2., z[:-1, :-1], levels=levels,
                      cmap='jet')
    fig.colorbar(cf, ax=ax1, label='relative gap')
    ax1.set_title('Relative gap between the theorical Jacobian matrix and the result of our algorithm')
    
    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()

    plt.ylabel("value of y")
    plt.xlabel("value of x")
    
    plt.show()
