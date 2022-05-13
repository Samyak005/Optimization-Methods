import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d

def testplot(points):
    df = pd.DataFrame(points, columns=['x', 'y', 'fValue'])
    for i,row in df.iterrows():
        if i==0:
            pass
        else:
            plt.annotate('',xy=(row['x'],row['y']),xytext=(df.iloc[i-1]['x'],df.iloc[i-1]['y']),
            arrowprops=dict(facecolor='black',width=1,headwidth=5))

    plt.xlim(-1 , 1.1)
    plt.ylim(-0.2,0.2)
    plt.show()

def funcPlot(points):
    plt.style.use('seaborn-poster')
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')

    ax.grid()
    for i in range(len(points)):
        ax.plot3D(points[i][0], points[i][1], points[i][2], linestyle='--', marker='o', color='orange')

    ax.set_title('3D Function Value Plot')

    # Set axes label
    ax.set_xlabel('x1', labelpad=20)
    ax.set_ylabel('x2', labelpad=20)
    ax.set_zlabel('f(x1,x2)', labelpad=20)

    plt.show()


def contourPlot(x1min, x1max, x2min,x2max, points):
    df = pd.DataFrame(points, columns=['x', 'y', 'fValue'])


    x1list = np.linspace(x1min, x1max, 100)
    x2list = np.linspace(x2min, x2max, 100)
    X1, X2 = np.meshgrid(x1list, x2list)

    A1 = X1 + 3 * X2  - 0.1
    A2 = X1 - 3 * X2  - 0.1
    A3 = - X1 - 0.1
    Z = np.exp(A1) + np.exp(A2) + np.exp(A3) 

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
    fx = 5*x1*x1 + 5*x2*x2 - x1*x2 - 11*x1 + 11*x2 + 11
    return fx

# Partial derivatives of fx wrt x1 and x2
def get_delfx_val(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    delfx_x1 = 10*x1 - x2 - 11
    delfx_x2 = 10*x2 - x1 + 11
    # print(delfx_x1, delfx_x2)
    # convert value from int to float and return
    return np.array([delfx_x1, delfx_x2]).astype(np.float64)


def get_hessian():

    delfx_x1_x2 = -1
    delfx_x1_x1 = 10
    delfx_x2_x1 = -1
    delfx_x2_x2 = 10

    h1 = np.array([[delfx_x1_x1, delfx_x1_x2], [delfx_x2_x1, delfx_x2_x2]]).astype(np.float64)
    h1_inv = np.linalg.inv(h1)
    h1_eigen = np.linalg.eig(h1)
    print("inverse", h1_inv)
    print("eigen", h1_eigen)
    return h1, h1_inv, h1_eigen



def steepest_descent(val_x_initial, flag):
    epsilon = np.array(0.000001).astype(np.float64)  
 
    k = 1  #iteration counter
    kmax = 100

    val_x = val_x_initial
    del_fx_val = get_delfx_val(val_x)  
    norm_delfx_val = np.linalg.norm(del_fx_val) 
    values = list()
    values.append([val_x[0], val_x[1], get_fx_val(val_x)])
    print(k, val_x, get_fx_val(val_x))

    # Hessian is independent of the value of val_x
    h1, h1_inv, h1_eigen  = get_hessian()
    ld1 = np.max(h1_eigen[0])
    print("max of 2 eigen values", ld1)
    while (norm_delfx_val >= epsilon) and (k < kmax):
        s = - del_fx_val
        # Compute the step
        if (flag == "less"):
            l1 = 1/ld1
        else:
            l1 = 3/ld1

        val_x = val_x + l1 * s
        del_fx_val = get_delfx_val(val_x)  
        norm_delfx_val = np.linalg.norm(del_fx_val) 
        values.append([val_x[0], val_x[1], get_fx_val(val_x)])
        k = k+1
        print(k, val_x, get_fx_val(val_x))

    contourPlot(-2.1, 2.1, -2.1, 2.1, values)

# MAIN
# declaring variables and initial data provided

print("Performing Steepest Descent")
val_x_initial = [1, 0.1] 
steepest_descent(val_x_initial, "less")

val_x_initial = [-2, 2] 
steepest_descent(val_x_initial, "less")

val_x_initial = [2, -2] 
steepest_descent(val_x_initial, "less")

val_x_initial = [2, 2] 
steepest_descent(val_x_initial, "less")

val_x_initial = [-2, -2] 
steepest_descent(val_x_initial, "less")

val_x_initial = [1, 0.1] 
steepest_descent(val_x_initial, "more")

val_x_initial = [-2, 2] 
steepest_descent(val_x_initial, "more")

val_x_initial = [2, -2] 
steepest_descent(val_x_initial, "more")

val_x_initial = [2, 2] 
steepest_descent(val_x_initial, "more")

val_x_initial = [-2, -2] 
steepest_descent(val_x_initial, "more")