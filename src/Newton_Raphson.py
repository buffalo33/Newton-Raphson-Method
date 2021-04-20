#!/usr/bin/python3

import numpy as np

#Application of the Newton-Raphson method
def Newton_Raphson(f, J, U0, N,eps):
    U = U0
    norm_fU  = np.linalg.norm(f(U))
    TAB_VAL = [norm_fU]
    i = 0
    h = 1e-5
    #print("Norme == ",np.linalg.norm(f(U)))
    while (norm_fU > eps) and (i <= N):
        #print(np.linalg.norm(f(U)))
        res = np.linalg.lstsq(J(f,U,h),-f(U),-1)  # -1 is a conventional security value to avoid error
        #V = np.matrix(res[0])
        V =res[0]
        #res = np.linalg.lstsq(J(f,U,h),-f(U),-1)  # -1 is a conventional security value to avoid error
        V = np.linalg.lstsq(J(f,U,h),-f(U),-1)[0]
        U = U + V
        norm_fU = np.linalg.norm(f(U))
        TAB_VAL.append(norm_fU)
        i += 1
        #print("Norme == ",norm_fU)
    return(U,norm_fU,f(U),i,TAB_VAL)

#Application of the Newton-Raphson method with backtracking
def Newton_Raphson_back(f, J, U0, N,eps):
    U = U0
    TAB_VAL = [np.linalg.norm(f(U))]
    i = 0
    h = 1e-5
    #print("Norme == ",np.linalg.norm(f(U)))
    while (np.linalg.norm(f(U)) > eps) and (i <= N):
        res = np.linalg.lstsq(J(f,U,h),-f(U),-1) # -1 is a conventional security value to avoid error
        V = res[0]
        #print("V == ",V)
        alpha = 0.75
        exp = 1
        #print(np.linalg.norm(f(U + V)))

        while (np.linalg.norm(f(U + V)) > np.linalg.norm(f(U))) and (exp < N):
           # print("***Entree dans boucle while***")

            #print("Norme == ",np.linalg.norm(f(U + V)))
            #print("V == ",V)
            V = (alpha**exp) * V
            exp += 1
            #print(np.linalg.norm(f(U + V)) >= np.linalg.norm(f(U)))
        #print("Sortie boucle while")
        U = U + V
        #print("U + V == ",U )
        TAB_VAL.append(np.linalg.norm(f(U)))
        i += 1
        #print("Norme == ",np.linalg.norm(f(U)))
    return(U,np.linalg.norm(f(U)),f(U),i,TAB_VAL)


#Newton Raphson method optmized by avoiding usless evaluation of f(U) or N(f(u))
def Newton_Raphson_back_opt(f, J, U0, N,eps):
    U = U0
    norm_fU = np.linalg.norm(f(U))
    TAB_VAL = [norm_fU]
    i = 0
    h = 1e-5
    #print("Norme == ",norm_fU)
    while (norm_fU > eps) and (i <= N):
        #res = np.linalg.lstsq(J(f,U,h),-f(U),-1) # -1 is a conventional security value to avoid error
        V = np.linalg.lstsq(J(f,U,h),-f(U),-1)[0]
        #print("V == ",V)
        alpha = 0.75
        exp = 1
        #print(np.linalg.norm(f(U + V)))
        norm_fU_V = np.linalg.norm(f(U + V))
        while (norm_fU_V >= norm_fU) and (exp < N):
            #print("***Enter in while loop***")
            #print("Norme == ",norm_fU_V)
            #print("V == ",V)
            V = (alpha**exp) * V
            norm_fU_V = np.linalg.norm(f(U + V))
            exp += 1
            #print(np.linalg.norm(f(U + V)) >= np.linalg.norm(f(U)))
        #print("Exit of while loop")
        U = U + V
        norm_fU = np.linalg.norm(f(U))
        TAB_VAL.append(norm_fU)
        i += 1
        #print("Norme == ",norm_fU)
    return(U,norm_fU,f(U),i,TAB_VAL)


import matplotlib.pyplot as plt
import jacob as j
import time as t


#-------------------------------Set of several test functions---------------------------------
def test_linear(X):
    Y = np.copy(X)
    Y[0][0] += 2
    return(Y) 

def test_poly_2(X):
    Y = np.copy(X)
    Y[0][0] = X[0][0]**2 + 3*X[0][0] - 4
    return(Y)

def test_dim_2(X): # Cette fonction  a-t-elle une solution ????
    Y = np.copy(X)
    Y[0][0] = (X[0][0] + X[1][0])/(X[0][0]**2 + 1)
    Y[1][0] = (X[0][0]**2 + 2)*X[1][0]
    return Y

#-----------------------------------------------------------------------------------------------


if __name__ == '__main__':

    print("--------------------------------Newton-Raphson tests--------------------------------")

    #Ensembles de parametres pour dimension 1
    U0 = np.matrix([[500.]])
    eps0 = 1e-12
    N0 = 100

    #Ensemble de parametres pour dimension 2
    U1 = np.array([[14], [50.23]])
    N1 = 50
    eps1 = 1e-12

    
    #res_linear = Newton_Raphson(test_linear,j.Jacobian,U0,50,1e-4)
    #print(res_linear)
    
    res_poly_2 = Newton_Raphson(test_poly_2,j.Jacobian,U0,N0,eps0)
    #print(res_poly_2)
    plt.plot(res_poly_2[-1],label='Without backtracking',color='r')
    plt.legend()
    plt.ylabel("||f(U)||")
    plt.xlabel("Number of Newton-Raphson method iterations")
    plt.yscale('log')
    plt.show()

    print("--------------------test simple :------------")
    sol = Newton_Raphson(test_dim_2,j.Jacobian,U1,N1,eps1)
    #print(sol)
    print("--------------------test avec back tracking :------------")
    U1 = np.matrix([[4],[45]])
    sol_back =  Newton_Raphson_back(test_dim_2,j.Jacobian,U1,N1,eps1)
    #print(sol_back)
    
    plt.plot(sol[-1],label='Without backtracking',color='r')
    plt.plot(sol_back[-1],label='With backtracking',color='b')
    plt.legend()
    plt.ylabel("||f(U)||")
    plt.xlabel("Number of Newton-Raphson method iterations")
    plt.yscale('log')
    plt.show()

    U1 = np.matrix([[600.],[45.]])
    sol_back =  Newton_Raphson_back(test_dim_2,j.Jacobian,U1,N1,eps1)
    plt.plot(sol_back[-1],label='With backtracking',color='b')
    plt.legend()
    plt.ylabel("||f(U)||")
    plt.xlabel("Number of Newton-Raphson method iterations")
    plt.yscale('log')
    plt.show()

    print("-----------------------------------------------------------------------------")
    
    #U1 = np.matrix([[4],[45]])
    #res_dim_2 = Newton_Raphson(test_dim_2,j.Jacobian,U1,N1,eps1)
    #print(res_dim_2)
    
    #print("-------------------------------------------------------------------------------")
    
    #print("------------------------------Newton-Raphson backtracking tests----------------")
    
    #U0 = np.matrix([[500.]])
    
    #res_linear_back = Newton_Raphson_back(test_linear,j.Jacobian,U0,50,1e-4)
    #print(res_linear_back)
    
    #res_poly_2_back = Newton_Raphson_back(test_poly_2,j.Jacobian,U0,50,1e-7)
    #print(res_poly_2_back)
    
    #U1 = np.matrix([[600.],[45.]])
    #res_dim_2_back = Newton_Raphson_back(test_dim_2,j.Jacobian,U1,100,1e-7)
    #print(res_dim_2_back)
    
    #print("---------------------------------------------------------------------------------")


    #print("--------------------Comparaison sans et avec optimisation :------------")
    #start = t.time()
    #sol_back =  Newton_Raphson_back(test_dim_2,j.Jacobian,U1,N1,eps1)
    #time1 = t.time() - start
    #print(sol_back)
    


    #start = t.time()
    #sol_back_opt =  Newton_Raphson_back_opt(test_dim_2,j.Jacobian,U1,N1,eps1)
    #time2 = t.time() - start
    #print(sol_back_opt)
    

    #print("Execution duration of Newton_Raphson method without optimization == ",time1)    
    #print("Execution duration of Newton_Raphson method with optimization == ",time2)
    
    #print("---------------------------------------------------------------------------------")
