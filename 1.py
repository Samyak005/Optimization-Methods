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

#exact line search 
def steepest_descent():
    epsilon = np.array(0.000001).astype(np.float64)  
    val_x_initial = [1, 0.1]  
    k = 1  #iteration counter
    kmax = 100

    val_x = val_x_initial
    del_fx_val = get_delfx_val(val_x)  
    norm_delfx_val = np.linalg.norm(del_fx_val) 
    values = list()
    values.append([val_x[0], val_x[1], get_fx_val(val_x)])
    print(k, val_x, get_fx_val(val_x))

    while (norm_delfx_val >= epsilon) and (k < kmax):
        h1, h1_inv  = get_hessian(val_x)
        s = - del_fx_val
        st = np.transpose(s)
        # Compute the step
        numerator = np.dot(st, s)
        denom1 = np.dot(h1, s)
        denom2 = np.dot(st, denom1)

        l1 = numerator/denom2

        val_x = val_x + l1 * s
        del_fx_val = get_delfx_val(val_x)  
        norm_delfx_val = np.linalg.norm(del_fx_val) 
        values.append([val_x[0], val_x[1], get_fx_val(val_x)])
        k = k+1
        print(k, val_x, get_fx_val(val_x))

    # testplot(values)
    funcPlot(values)
    contourPlot(-0.5, 1, -0.1, 0.1, values)

#goldstein line search 
def armijo_goldstein():
    print("\nPerforming armijo_goldstein Line Search")
    epsilon = np.array(0.000001).astype(np.float64) 
    alpha_init = 1  #assumed
    Tau = 0.7  #assumed
    beta = 0.1  
    val_x_initial = [1, 0] 
    k = 1  #iteration counter
    kmax = 100
    eta = 0.7 #assumed
    beta = 0.3 #assumed

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
    values.append([val_x[0], val_x[1], get_fx_val(val_x)])
    val_dk = -1 * get_delfx_val(val_x)

    while (norm_delfx_val >= epsilon) and (k < kmax):
        fx_val = get_fx_val(val_x) 
        delta = alpha_init * beta * np.dot(del_fx_val, val_dk)
        goldstein_delta = alpha_init * eta * np.dot(del_fx_val, val_dk)
        armijo_check = fx_val + delta
        goldstein_check = fx_val + goldstein_delta
        val_x_temp = val_x + alphak*val_dk  # x+a*d
        armijo_check_new = get_fx_val(val_x_temp)  # F(x+a*d)
        # goldstein_check_new = 

        alphak = alpha_init
        while (armijo_check <= armijo_check_new or armijo_check_new <= goldstein_check):
            alphak = alphak*Tau
            # fx_val = get_fx_val(val_x_temp) 
            delta = alphak * beta * np.dot(del_fx_val, val_dk)
            goldstein_delta = alpha_init * eta * np.dot(del_fx_val, val_dk)
            armijo_check = fx_val + delta
            goldstein_check = fx_val + goldstein_delta
            val_x_temp = val_x + alphak*val_dk  # x+a*d
            armijo_check_new = get_fx_val(val_x_temp)  # F(x+a*d)

        
        val_x = val_x_temp
        del_fx_val = get_delfx_val(val_x)  # delfx(x)
        val_dk = -1 * del_fx_val
        norm_delfx_val = np.linalg.norm(del_fx_val) 
        values.append([val_x[0], val_x[1], get_fx_val(val_x)])
        k = k + 1
    print('norm_delfx_val', str(norm_delfx_val), 'epsilon', str(epsilon), 'numIterations', str(k))

    for i in range(len(values)):
        print(values[i])
        
    funcPlot(values)
    contourPlot(-1.5, 1.5, -0.5, 0.1, values)

# MAIN
# declaring variables and initial data provided

# print("Performing Steepest Descent - Exact Line Search")
# steepest_descent()

print("Performing Steepest Descent - Armijo-Goldstein Line Search")
armijo_goldstein()

print("\nPerforming Inexact Line Search with Backtracking")
epsilon = np.array(0.000001).astype(np.float64) 
alpha_init = 1  #assumed
Tau = 0.7  #assumed
beta = 0.1  
val_x_initial = [1, 0] 
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
values.append([val_x[0], val_x[1], get_fx_val(val_x)])
val_dk = -1 * get_delfx_val(val_x)

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
    val_dk = -1 * del_fx_val
    norm_delfx_val = np.linalg.norm(del_fx_val) 
    values.append([val_x[0], val_x[1], get_fx_val(val_x)])
    k = k + 1
print('norm_delfx_val', str(norm_delfx_val), 'epsilon', str(epsilon), 'numIterations', str(k))

for i in range(len(values)):
    print(values[i])
    
funcPlot(values)
contourPlot(-1.5, 1.5, -0.5, 0.1, values)
