#! /usr/bin/python3

from math import *
import numpy as np
import matplotlib.pyplot as plt
import jacob as j
import Newton_Raphson as nr




#fonctions preliminaires : 


def elec_energy(x):
    N=x.shape[0]
    E=0
    for i in range(N):
        E+=log(abs(x[i]+1))+log(abs(x[i]-1))
        for j in range(N):
            if (j!=i):
                E+= 0.5*log(abs(x[i]-x[j]))
    return E

#test de elec_energy
x=np.matrix([[-0.25],[0.5],[-0.75],[0.2],[0.3]])
print("Test de elec_energy : " ,elec_energy(x))

def gradient_elec_energy(x):
    N=x.shape[0]
    Energy=np.zeros((N,1))
    for i in range(N):
        Energy[i][0]=(1/(x[i][0]-1)) + (1/(x[i][0]+1))
        for j in range(N):
            if j!=i and x[i][0]-x[j][0]!=0:
                    Energy[i][0]=Energy[i][0] + (1/(x[i][0]-x[j][0]))
    return Energy


#test de gradient_elec_energy
print("Test de gradient_elec_energy : \n")
print(gradient_elec_energy(x))
print("\n")
y=np.matrix([[0.25],[-0.3],[0.75]])
print(gradient_elec_energy(y))
print("\n")

# Question 1 :
def gradient_jacob(x):
    N=x.shape[0]
    Jacob=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j==i:
                Jacob[i][j]=-1/(x[j][0]+1)**2 - 1/(x[j][0]-1)**2
                for l in range(N):
                    if l!=i:
                        Jacob[i][j]+=-1/(x[j][0]-x[l][0])**2
            else:
                Jacob[j][i]=1/(x[j][0]-x[i][0])**2
    return Jacob
print("La Jacobienne du gradient de l'energie :\n") 
print(gradient_jacob(y))




# Test for "1" electrostatic charge
length=500
pos=np.linspace(-0.99,0.99,length)
Energy=np.zeros((length,1))
for k in range(length):
    Energy[k]=elec_energy(np.asarray([pos[k]]))

plt.title("Energy & Equilibrium of one charge")
plt.xlabel("Charge's position")
plt.ylabel("Electrostatic Energy")
plt.plot(pos,Energy)
plt.show()

# Question 2

def Newton_Raphson_back_modif(f, J, U0, N, eps):
    f_image = f(U0)
    U = U0
    global norme_f
    global iterations
    norme_f=[]
    iterations=[]
    
    f_norm = np.linalg.norm(f_image)
    iter_count = 0
    norme_f.append(f_norm)
    iterations.append(iter_count)
    while iter_count < N and eps < abs(f_norm) :
        V = np.linalg.solve(J(U),-f_image)
        U = V + U
        iter_count +=1
        f_image = f(U)
        f_norm = np.linalg.norm(f_image)
        iterations.append(iter_count)
        norme_f.append(f_norm)
    if eps < abs(f_norm):
        iter_count = NULL
        
    return U, iter_count

U0=np.zeros((5,1))
U0[0]=-0.05640298
U0[1]=0.23504586
U0[2]=-0.10058913
U0[3]=-0.58297730
U0[4]=0.96542272



W0=np.zeros((10,1))
W0[0]=-0.05640298
W0[1]=0.23504586
W0[2]=-0.10058913
W0[3]=0.58297730
W0[4]=0.96542272
W0[5]=0.382977308
W0[6]=-0.24235892
W0[7]=-0.56233178
W0[8]=0.95422312
W0[9]=0.482977308
J=gradient_jacob
f=gradient_elec_energy

print("Test for Newton_method_back_modif to solve the equation")
Newton_Raphson_back_modif(f, J, U0, N=50, eps=1e-5)
plt.plot(iterations,norme_f,label="5 charges")
Newton_Raphson_back_modif(f, J, W0, N=50, eps=1e-5)
plt.plot(iterations,norme_f,label="10 charges")
plt.title("Electrostatic equilibrium")
plt.xlabel("iterations")
plt.ylabel("Norme(f)")
plt.legend()
plt.show()
