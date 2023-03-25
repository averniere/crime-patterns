import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import time
import copy

'''Parameters initialization'''

# Vectors of parameters
case_chosen = input("choose a set of parameters between a,b,c,d :")
param_3_a = [1, 50, 1/100, 1/2, 0.019, 0.56, 1/15, 0.2]
param_3_b = [1, 20, 1/100, 1/2, 0.002, 5.6, 1/15, 0.2]
param_3_c = [1, 20, 1/100, 1/2, 0.019, 0.56, 1/15, 0.03]
param_3_d = [1, 20, 1/100, 1/2, 0.002, 5.6, 1/15, 0.03]
param = globals()['param_3_%s' % case_chosen]

# Distance between each houses
l = param[0]

# Size of the grid
L = param[1]

# Discete time unite
dt = param[2]

# Time
t = 0 
# Final time
end = param[3]

# Generation rate
gamma = param[4]
theta = param[5]
w = param[6]
eta = param[7]

"""Attractiveness"""
A_0 = np.zeros((L, L))+1/30
B_avg = gamma*theta/w
B_0 = np.zeros((L, L))+B_avg
B = B_0
A = A_0+B
A_avg = A_0+B_avg

"""List of all the sites s=(i,j)"""
Sites = []
for i in range(L):
    for j in range(L):
        Sites.append((i,j))

"""Definition of a function giving the list of the neighbors for a given site"""
def distance(s, n):
    return np.sqrt((s[0]-n[0])**2+(s[1]-n[1])**2)

def neighbors(s):
    Neighbors = []
    for n in Sites:
        if distance(s, n) == 1.0:
            Neighbors.append(n)
        if len(Neighbors) == 4:
            return np.array(Neighbors)
    return np.array(Neighbors)

Ngbr = [[neighbors ((i, j)) for j in range (L)]for i in range (L)]

'''Matrix giving the number of neighboring sites'''
Z = np.zeros((L, L))
for s in Sites:
    i, j = s[0],s[1]
    Z[i][j] = len(Ngbr[i][j])

'''Matrix of the number of criminals in the site'''
N_crim = (gamma*dt)/(1-np.exp(-A_avg*dt))
n_avg = (gamma*dt)/(1-np.exp(-A_avg[0][0]*dt))

'''Creation of a matrix that indicates the number of criminals at each site'''
N = np.zeros((L,L))
for i in range(L):
    for j in range(L):
        p = rd.random()
        if p <= n_avg :
            N[i][j] = 1
"""total number of criminals"""
n_criminals = int(sum(sum(N)))
print("criminels",n_criminals)

# Probability to move from site s to site n
def q (s, n):
    n_x, n_y = n[0], n[1]
    s_x, s_y = s[0], s[1]
    return A[n_x][n_y]/sum([A[S[0]][S[1]] for S in Ngbr[s_x][s_y]])


P = 1-np.exp(-A*dt)
h = 0
C = [] # List of the number of criminals on the grid at each iteration
T = []
Added = [] # List of the number of criminals added to the grid at each iteration
# Criminal loop
while t <= end:
    E=np.zeros((L,L))
    N_t = copy.deepcopy(N)
    added = 0
    for s in Sites:
        i,j = s[0],s[1]
        if N[i][j] > 0:
            nb_criminals_s = int(N[i][j])
            for criminal in range(nb_criminals_s):
                r = rd.random()
                # Burgle
                if r<P[i][j]:
                    # Remove burglar from grid
                    N_t[i][j] = N[i][j]-1 
                    # Increment E
                    E[i][j]+=1 
                # Move to another site  
                else:
                    s_neighbors = Ngbr[i][j]
                    prob_neighbors = [q(s,n) for n in s_neighbors]
                    res=[k for k in range (len(s_neighbors))]
                    n=rd.choice(res,p=prob_neighbors)
                    s_chosen=s_neighbors[n]
                    N_t[s_chosen[0]][s_chosen[1]]+=1
                    N_t[i][j]= N[i][j]-1
        c = rd.random()
        if c < gamma :
            N_t[i][j]+=1 
            added += 1
    #Computation of the discrete spatial Laplacian operator applied to B
    delta_B = np.zeros((L,L))
    for s in Sites:
        i,j= s[0],s[1]
        delta_B[i][j]=(sum([B[S[0]][S[1]] for S in Ngbr[i][j]])-Z[i][j]*B[i][j])
    #Computation of B(t+dt) via Eq.2.6
    B=(B+eta*(delta_B)/Z)*(1-w*dt)+theta*E
    #Update of the variables
    A = A_0+B
    P = 1-np.exp(-A*dt)
    N = N_t
    C.append(int(sum(sum(N))))
    T.append(t)
    Added.append(added)
    t=t+dt
    if t/end > 0.1*h :
        h+=1
        print(int((t/end)*100),"% reached !")

    #if int(t/dt)%10 == 0:
        print("n_criminals =", sum(sum(N)))
        plt.imshow(N,interpolation='none',cmap = 'jet', origin = "lower")
        plt.colorbar()
        plt.title("N = number of criminals")
        plt.show()

        plt.imshow(P,interpolation='none',cmap = 'jet', origin = "lower",vmin=0,vmax = 1)
        plt.colorbar()
        plt.title("P = Probability")
        plt.pause(0.001)
        plt.show()
        
        print(B)
        plt.imshow(delta_B,interpolation='none',cmap = 'jet', origin = "lower")
        plt.colorbar()
        plt.title("delta_B")
        plt.show()

        plt.imshow(B,interpolation='none',cmap = 'jet', origin = "lower")
        plt.colorbar()
        plt.title("B")
        plt.show()

        plt.imshow(E,interpolation='none',cmap = 'jet', origin = "lower")
        plt.colorbar()
        plt.title("E")
        plt.pause(0.001)
        plt.show()
        
XB = np.array([k for k in range(L)])
YB = np.array([k for k in range(L)])

U = A[XB][YB]

plt.imshow(U,interpolation='none',cmap = 'jet', origin = "lower")
plt.colorbar()
plt.title("Attractiveness")
plt.pause(0.001)
plt.show()

plt.plot(T,C)
plt.grid()
plt.pause(0.001)
plt.show()
plt.plot(T,Added)
plt.grid()
plt.pause(0.001)
plt.show()

