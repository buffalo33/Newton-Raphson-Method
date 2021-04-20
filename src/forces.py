#! /usr/bin/python3
import numpy as np
import jacob as j
import Newton_Raphson as nr
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


###########################################################
                   #Forces#
###########################################################


def elastic_force(k, U):
    return np.array([[-k*x], [-k*y]])

def centrifugal_force(k, x0, y0, U):
    F=np.array([[k*(U[0,0]-x0)], [k*(U[1,0]-y0)]])
    return F

def gravitational_force(k, x0, y0, U):
    
    d=np.array([[U[0,0]-x0], [U[1,0]-y0]])
    n=np.linalg.norm(d)
    if n==0:
        return 0
    fx=-k*d[0,0]/(n**3)
    fy=-k*d[1,0]/(n**3)
    return np.array([[fx], [fy]])


#################################################
                  #Exemple# 
#################################################

f1 = lambda U : gravitational_force(1.0, 0.0, 0.0, U) + gravitational_force(0.01, 1.0, 0.0, U)+centrifugal_force(1.0, 0.01/1.01, 0.0, U)  

V = lambda x, y : np.array([[x], [y]])

J = lambda f, x, h : j.Jacobian(f, x, h)


##################################################
                   #Simple test#
##################################################

def simple_test():
    N=10
    eps=1e-8
    U0 = np.array([[1.5], [0.0]])
    print("----------------- U0------------------")
    print(U0)
    print("----------------- f1(U0)------------------")
    print(f1(U0))
    print("-----------------Jacobian : f1------------------")
    print(J(f1, U0, 1e-8))
    print("-----------------Newton Raphson : f1------------------")
    print(nr.Newton_Raphson(f1, J, U0, N, eps))
    print("-----------------Newton Raphson - backtracking : f1------------------")
    print(nr.Newton_Raphson_back(f1, J, U0, N, eps))

##################################################
                   #Graph of f1 : 3D#
##################################################

def graph_f1():
    ax = Axes3D(plt.figure())
    X = np.arange(-1,1,0.02)
    Y = np.arange(-1,1,0.02)
    X, Y = np.meshgrid(X, Y)
    Z = f1(np.array([[X], [Y]]))[0,0]+f1(np.array([[X],[Y]]))[1,0]
    ax.plot_surface(X, Y, Z)
    plt.show()



####################################################################
                 #Lagrangian points#
####################################################################

def Lagrangian_points():
    print("Please wait")
    N=10
    X=[]
    Y=[]
    x0=-5.0
    for a in range(40):
        y0=-5.0
        for b in range(40):
            if(b==20):
                print('*')
            
            U0=np.array([[x0], [y0]])
            A=nr.Newton_Raphson_back(f1, J, U0, N, 1e-8)
            U=A[0]
            if(A[1]<1e-6):
                X.append(U[0,0])
                Y.append(U[1,0])
            y0=y0+0.25
                
                
        x0=x0+0.25

        
    plt.plot(X, Y)
    plt.plot(0, 0, marker='o', c='yellow')
    plt.text(-0.04, -0.07, 'Sun')
    
    plt.plot(1, 0, marker='o', c='blue')
    plt.text(0.95, -0.07, 'Earth')
    
    plt.plot(0.855421, -0.000623156, marker='o', c='black')
    plt.text(0.75, -0.07, 'L1')
    
    plt.plot(1.15728, 0.00327623, marker='o', c='red')
    plt.text(1.15728, -0.09, 'L2')
    
            
    plt.plot(-0.998237, -0.00429466, marker='o', c='green')
    plt.text(-1, -0.09, 'L3')
    
    plt.plot(0.503092, 0.867106, marker='o', c='brown')
    plt.text(0.55, 0.86, 'L4')
    
    plt.plot(0.503092, -0.867106, marker='o', c='orange')
    plt.text(0.55, -0.88, 'L5')
    
            
    plt.xlabel("abscisses")
    plt.ylabel("ordonnees")
            
    plt.title("Lagrangian points")
            
    plt.show()




def main():
    print("####################  TEST1  #################")
    simple_test()
    print("####################  TEST2  #################")
    Lagrangian_points()
    #graph_f1()

if __name__ == "__main__":
    main()



