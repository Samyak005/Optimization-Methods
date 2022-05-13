import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d

def funcPlot(points):
    plt.style.use('seaborn-poster')
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')

    ax.grid()
    for i in range(len(points)):
        if points[i][2]>0:
            # diff is greater than zero
            ax.plot3D(points[i][0], points[i][1], points[i][2], linestyle='--', marker='o', color='green')
        else:
            ax.plot3D(points[i][0], points[i][1], points[i][2], linestyle='--', marker='o', color='orange')

    ax.set_title('Diff Plot - Orange when less than zero, Green when more than zero')

    # Set axes label
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('f(x* + alpha* dtheta) - f(x*)', labelpad=20)

    plt.show()


def contourPlot(x1min, x1max, x2min,x2max, points):
    df = pd.DataFrame(points, columns=['x', 'y', 'fValue'])


    x1list = np.linspace(x1min, x1max, 100)
    x2list = np.linspace(x2min, x2max, 100)
    X1, X2 = np.meshgrid(x1list, x2list)

    Z = 10*X1*X1 + X2*X2 + 10*X1*X2 + 4*X1 - 10*X2 + 2

    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X1, X2, Z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Contours Plot')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for i,row in df.iterrows():
        if i==0:
            pass
        else:
            plt.annotate('',xy=(row['x'],row['y']),xytext=(df.iloc[i-1]['x'],df.iloc[i-1]['y']),
            arrowprops=dict(facecolor='black',width=1,headwidth=5))
    for i in range(len(points)):
        plt.plot(points[i][0], points[i][1], linestyle='--', marker='o', color='orange')

    plt.show()

def contourPlot_second(x1min, x1max, x2min,x2max, points):
    df = pd.DataFrame(points, columns=['x', 'y', 'fValue'])


    x1list = np.linspace(x1min, x1max, 100)
    x2list = np.linspace(x2min, x2max, 100)
    X1, X2 = np.meshgrid(x1list, x2list)

    Z = 16*X1*X1 + 10*X2*X2 + 8*X1*X2 + 12*X1 - 6*X2 + 2

    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X1, X2, Z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Contours Plot')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for i,row in df.iterrows():
        if i==0:
            pass
        else:
            plt.annotate('',xy=(row['x'],row['y']),xytext=(df.iloc[i-1]['x'],df.iloc[i-1]['y']),
            arrowprops=dict(facecolor='black',width=1,headwidth=5))
    for i in range(len(points)):
        plt.plot(points[i][0], points[i][1], linestyle='--', marker='o', color='orange')

    plt.show()


def get_fx_val(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    fx = 10*x1*x1 + x2*x2 + 10*x1*x2 + 4*x1 - 10*x2 + 2
    return fx


def get_fx_val_second(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    fx = 16*x1*x1 + 10*x2*x2 + 8*x1*x2 + 12*x1 - 6*x2 + 2
    return fx

# Partial derivatives of fx wrt x1 and x2
def get_delfx_val(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    delfx_x1 = 20*x1 + 10*x2 + 4
    delfx_x2 = 2*x2  + 10*x1 - 10
    # print(delfx_x1, delfx_x2)
    # convert value from int to float and return
    return np.array([delfx_x1, delfx_x2]).astype(np.float64)

# Partial derivatives of fx wrt x1 and x2
def get_delfx_val_second(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    delfx_x1 = 32*x1 + 8*x2 + 12
    delfx_x2 = 20*x2  + 8*x1 - 6
    # print(delfx_x1, delfx_x2)
    # convert value from int to float and return
    return np.array([delfx_x1, delfx_x2]).astype(np.float64)

def get_hessian():
    delfx_x1_x2 = 10
    delfx_x1_x1 = 20
    delfx_x2_x1 = 10
    delfx_x2_x2 = 2
    h1 = np.array([[delfx_x1_x1, delfx_x1_x2], [delfx_x2_x1, delfx_x2_x2]]).astype(np.float64)
    h1_inv = np.linalg.inv(h1)
    h1_eigen = np.linalg.eig(h1)
    print("inverse", h1_inv)
    print("eigen", h1_eigen)
    ld1 = np.max(h1_eigen[0])
    print("max of 2 eigen values", ld1)
    return h1_eigen

def get_hessian_second():
    delfx_x1_x2 = 8
    delfx_x1_x1 = 32
    delfx_x2_x1 = 8
    delfx_x2_x2 = 20
    h1 = np.array([[delfx_x1_x1, delfx_x1_x2], [delfx_x2_x1, delfx_x2_x2]]).astype(np.float64)
    h1_inv = np.linalg.inv(h1)
    h1_eigen = np.linalg.eig(h1)
    print("inverse", h1_inv)
    print("eigen", h1_eigen)
    ld1 = np.max(h1_eigen[0])
    print("max of 2 eigen values", ld1)
    return h1_eigen

# main program starting...
val_x_initial = [1.8, -4]
val_x_initial_second = [-0.5, 0.5]

fx_val_initial = get_fx_val(val_x_initial)
fx_val_initial_second = get_fx_val_second(val_x_initial_second)

del_fx_val = get_delfx_val(val_x_initial)
del_fx_val_second = get_delfx_val_second(val_x_initial_second)

print(del_fx_val)
print(get_hessian())
print(del_fx_val_second)
print(get_hessian_second())

values1 = list()
values2 = list()
val_x = val_x_initial
val_x_second = val_x_initial_second
alpha = 0.01
for theta in range(0,360):
    s1 = np.sin(theta* 3.142/180)
    c1 = np.cos(theta* 3.142/180)
    dk = np.array([c1, s1]).astype(np.float64)

    val_x = val_x_initial + alpha * dk
    val_x_second = val_x_initial_second + alpha * dk

    fx_val = get_fx_val(val_x)
    fx_val_second = get_fx_val_second(val_x_second)

    diff1 = fx_val - fx_val_initial
    diff2 = fx_val_second - fx_val_initial_second

    # print(theta, diff1)
    values1.append([val_x[0], val_x[1], diff1])
    values2.append([val_x_second[0], val_x_second[1], diff2])

funcPlot(values1)
contourPlot(-5, 5, -5,5, values1)
# OBSERVATIONS -
# From the contour plot, it is visible that in two directions, 
# the value of the function is decreasing while in other two directions, 
# the value of the function is increasing.
# From the plot of difference around the point [1.8, -4] for varying values of theta
# The difference between the value of [f(x* + alpha * d theta) – f(x*)] is positive 
# as well as negative for certain values of theta.  In two directions, 
# the difference is negative and in other orthogonal two directions, the difference is positive.

# del_fx = [0. 0.]
# Hessian inverse = [[-0.03333333  0.16666667]
#  [ 0.16666667 -0.33333333]]
# eigen values of Hessian =  (array([24.45362405, -2.45362405]), array([[ 0.91350006, -0.40683858],
#        [ 0.40683858,  0.91350006]]))
# max of 2 eigen values 24.45362404707371


funcPlot(values2)
contourPlot_second( -5,5, -5,5, values2)
# contourPlot_second( -5,5, -5,5, values2)


# OBSERVATIONS -
# From the contour plot of the function, 
# it is visible that function is increasing in almost all directions and 
# the minima is at the point [-0.5, 0.5]. 

# From the plot of difference around the point [0,5, 0.5] for varying values of theta
# The difference between the value of [f(x* + alpha * d theta) – f(x*)] is positive 
# for all values of theta. That is function value is minimum at the point [-0.5, 0.5] 

# 
# del_fx =[ 0. 0.]
# Hessian inverse [[ 0.03472222 -0.01388889]
#  [-0.01388889  0.05555556]]
# eigen values of Hessian = (array([36., 16.]), array([[ 0.89442719, -0.4472136 ],
#        [ 0.4472136 ,  0.89442719]]))
# max of 2 eigen values 36.0
