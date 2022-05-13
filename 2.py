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

def ellipsoid_plot(x1min, x1max, x2min,x2max, points):

    # df = pd.DataFrame(points, columns=['x', 'y', 'fValue', 'hessian'])

    x1list = np.linspace(x1min, x1max, 500)
    x2list = np.linspace(x2min, x2max, 500)
    X1, X2 = np.meshgrid(x1list, x2list)
    print(x1list)

    for i in range(len(points)):
        plt.clf()
        for x1 in x1list:
            for x2 in x2list:
                A3 = np.array([x1 - points[i][0], x2 - points[i][1]])
                
                A2 = points[i][3]
                
                A1 = np.transpose(A3)
                # print(A1.shape)
                # print(A2.shape)
                # print(A3.shape)
                ans = np.matmul(A1, np.matmul(A2, A3))
                if ans<=1:
                    plt.plot(x1, x2, linestyle='--', marker='o', color='orange')
        
        plt.show()

def contourPlot(x1min, x1max, x2min,x2max, points):
    df = pd.DataFrame(points, columns=['x', 'y', 'fValue', 'hessian'])


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

def combined_plot(x1min, x1max, x2min,x2max, points):

    df = pd.DataFrame(points, columns=['x', 'y', 'fValue', 'hessian'])

    x1list = np.linspace(x1min, x1max, 100)
    x2list = np.linspace(x2min, x2max, 100)
    X1, X2 = np.meshgrid(x1list, x2list)
    print(x1list)

    for j in range(len(points)):
        plt.clf()

        #contour plot

        B1 = X1 + 3 * X2  - 0.1
        B2 = X1 - 3 * X2  - 0.1
        B3 = - X1 - 0.1
        Z = np.exp(B1) + np.exp(B2) + np.exp(B3) 

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(X1, X2, Z)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Contours Plot' + str(j))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        #ellipsoid plot
        for x1 in x1list:
            for x2 in x2list:
                A3 = np.array([x1 - points[j][0], x2 - points[j][1]])
                
                A2 = points[j][3]
                
                A1 = np.transpose(A3)
                # print(A1.shape)
                # print(A2.shape)
                # print(A3.shape)
                ans = np.matmul(A1, np.matmul(A2, A3))
                if ans<=1:
                    plt.plot(x1, x2, linestyle='--', marker='o', color='yellow')
        
        # points and arrows 
        for i,row in df.iterrows():
            if i==0:
                pass
            else:
                plt.annotate('',xy=(row['x'],row['y']),xytext=(df.iloc[i-1]['x'],df.iloc[i-1]['y']),
                arrowprops=dict(facecolor='black',width=1,headwidth=5))
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1], linestyle='--', marker='o', color='orange')
        
        plt.plot(points[j][0], points[j][1], linestyle='--', marker='o', color='red')
        plt.show()

def get_fx_val(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    a1 = x1 + 3 * x2  - 0.1
    a2 = x1 - 3 * x2  - 0.1
    a3 = - x1 - 0.1
    fx = math.exp(a1) + math.exp(a2) + math.exp(a3) 
    return fx

# Partial derivatives of fx wrt x1 and x2
def get_delfx_val(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    a1 = x1 + 3 * x2  - 0.1
    a2 = x1 - 3 * x2  - 0.1
    a3 = - x1 - 0.1
    delfx_x1 = math.exp(a1) + math.exp(a2) - math.exp(a3) 
    delfx_x2 = 3* math.exp(a1) -3 * math.exp(a2)
    # print(delfx_x1, delfx_x2)
    # convert value from int to float and return
    return np.array([delfx_x1, delfx_x2]).astype(np.float64)

def get_hessian(val_x):
    x1 = val_x[0]
    x2 = val_x[1]
    a1 = x1 + 3 * x2  - 0.1
    a2 = x1 - 3 * x2  - 0.1
    a3 = - x1 - 0.1
    delfx_x1 = math.exp(a1) + math.exp(a2) - math.exp(a3) 
    delfx_x2 = 3* math.exp(a1) -3 * math.exp(a2)
    delfx_x1_x2 = 3 * math.exp(a1) -3 * math.exp(a2) 
    delfx_x1_x1 = math.exp(a1) + math.exp(a2) + math.exp(a3) 
    delfx_x2_x1 = 3* math.exp(a1) -3 * math.exp(a2)
    delfx_x2_x2 = 9* math.exp(a1) + 9 * math.exp(a2)

    h1 = np.array([[delfx_x1_x1, delfx_x1_x2], [delfx_x2_x1, delfx_x2_x2]]).astype(np.float64)
    h1_inv = np.linalg.inv(h1)
    # print(delfx_x1, delfx_x2)
    # convert value from int to float and return
    return h1, h1_inv

# main program starting...
print("\nPerforming Inexact Line Search with Backtracking")
epsilon = np.array(0.000001).astype(np.float64) 
alpha_init = 1  #assumed
Tau = 0.7  #assumed
beta = 0.1  
val_x_initial = [0.1, 0.1] 
k = 1  #iteration counter
kmax = 100

print('epsilon=',epsilon)
print('alpha_init=',alpha_init)
print('Tau=',Tau)
print('beta=',beta)
print('k=',k)

val_x = val_x_initial
del_fx_val = get_delfx_val(val_x)  
norm_delfx_val = np.linalg.norm(del_fx_val) 
if(norm_delfx_val < epsilon):
    print('Solution found at initial point', val_x)
    exit(1)

alphak = alpha_init
values = list()
h1, h1_inv  = get_hessian(val_x)
values.append([val_x[0], val_x[1], get_fx_val(val_x), h1])
val_dk = -1* np.dot(h1_inv, del_fx_val)

while (norm_delfx_val >= epsilon) and (k < kmax):
    fx_val = get_fx_val(val_x) 
    delta = alpha_init * beta * np.dot(del_fx_val, val_dk)
    armijo_check = fx_val + delta
    val_x_temp = val_x + alphak*val_dk  # x+a*d
    armijo_check_new = get_fx_val(val_x_temp)  # F(x+a*d)

    alphak = alpha_init
    while (armijo_check <= armijo_check_new):
        alphak = alphak*Tau
        # fx_val = get_fx_val(val_x_temp) 
        delta = alphak * beta * np.dot(del_fx_val, val_dk)
        armijo_check = fx_val + delta
        val_x_temp = val_x + alphak*val_dk  # x+a*d
        armijo_check_new = get_fx_val(val_x_temp)  # F(x+a*d)

    val_x = val_x_temp
    del_fx_val = get_delfx_val(val_x)  # delfx(x)
    h1, h1_inv  = get_hessian(val_x)
    val_dk = -1* np.dot(h1_inv, del_fx_val)

    norm_delfx_val = np.linalg.norm(del_fx_val) 
    values.append([val_x[0], val_x[1], get_fx_val(val_x), h1])
    k = k + 1
print('norm_delfx_val', str(norm_delfx_val), 'epsilon', str(epsilon), 'numIterations', str(k))

for i in range(len(values)):
    print(values[i])

funcPlot(values)
# contourPlot(-0.7, 0.1, -0.1, 0.1, values)
# ellipsoid_plot(-1, 1, -1, 1, values)
combined_plot(-1, 1, -0.5, 0.5, values)