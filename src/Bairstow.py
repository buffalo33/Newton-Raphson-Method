import numpy as np
import cmath
import random
from matplotlib.pyplot import *
from Newton_Raphson import *

def R_S(P,BC):  
    A=np.polydiv(P,np.array([1,BC[0],BC[1]]))[1]
    if(len(A) <2):
        return np.array([0,A[0]])
    else:
        return A
    

def partial_derivatives(P,BC):
    Q=np.polydiv(P,np.array([1.0,BC[0],BC[1]]))[0]
    R1=R_S(Q,BC)
    if(len(R1) == 2):
        a=R1[0]
        b=R1[1]
    else:
        a=R1
        b=0
    return np.array([[BC[0]*a-b ,-a],[ BC[1]*a,-b]])
    
 
def quadratic_factor(P,U0):
    f=lambda BC:R_S(P,BC)
    Jf=lambda f,BC,h: partial_derivatives(P,BC)
    N=50
    eps=1e-5
    U = Newton_Raphson(f,Jf,U0,N,eps)[0]
    B=U[0]
    C=U[1]
    fac=np.array([1,B,C])
    return fac


def solver_quadratic_equation(T):
   delta = T[1]**2 - 4*T[0]*T[2]
   if delta > 0:
       racineDeDelta=np.sqrt(delta)
       retour = [(-T[1]-racineDeDelta)/(2*T[0]),(-T[1]+racineDeDelta)/(2*T[0])]
   elif delta < 0:
       racineDeDelta=np.sqrt(-delta)
       retour = [(-T[1]-1j*racineDeDelta)/(2*T[0]),(-T[1]+1j*racineDeDelta)/(2*T[0])]
   else:
       retour = [-T[1]/(2*T[0])] 
   return retour

def bairstow(P):
    if (len(P) < 3): 
        return solver_quadratic_equation(P)
    retour = []
    T = [0]*len(P)
    U0= [-15,15]
    while (len(P) > 3):
        T = quadratic_factor(P,U0)
        print(T)
        retour = retour + solver_quadratic_equation(T)
        P = np.polydiv(P,T)[0]
        
    if(len(P)==3):
        return retour + solver_quadratic_equation(P)
    else:
        return retour + [-P[1]]
    
    



#############################################test_fonction##########################################


def generate_poly(n):
    P=[1]
    for i in range(n):
        P+=[random.randint(0,10)]
    return P

def test_bairstow(n):
    P=[0]*n
    T=generate_poly(n)
    A=sorted([abs(i) for i in np.roots(T)])
    B=sorted([abs(i) for i in bairstow(T)])
    for i in range(n):
        P[i]=A[i]-B[i]
    return P



if __name__ == '__main__':
    #-----------------------------------------------------------------------------#

    #P=np.array([1.0,2,8,0,-1])
    #BC=np.array([0,1])
    #print(np.polydiv(P,np.array([1,BC[0],BC[1]]))[0])
    #print(np.polydiv(P,np.array([1,BC[0],BC[1]]))[1])

    
   
    #print(RS(P,np.array([-1.0,1.0])))
    #print(partial_derivatives(P,BC))
    #P=np.array([1.0,0,1])
    
    #U0=[1,3]

    #print(quadratic_factor(P,U0))
    #print(bairstow(P))

    #print(test_bairstow(8))
    #print(test_bairstow(6))
    

    

    ############################# the graph #######################################
    n=7
    T=[0]*(n)
    for i in range(2,n+2):
        
        #print(test_bairstow(i))
        T[i-2]=max(test_bairstow(i))

    x=[i-1 for i in range (2,n+2)]
    #print(T)
    plt.plot(x, T)
    plt.xlabel('Degree')
    plt.ylabel('Max of difference')
    plt.title('Relative difference in different  degrees')
    plt.show()




