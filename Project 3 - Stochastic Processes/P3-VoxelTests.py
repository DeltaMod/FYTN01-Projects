from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

plotIter = False
plotInitEnd = True

def init_matrix(N=20):
    """ Use this to setup the size of the matrices and the number of walkers. """
    return(np.random.choice([0,1], size=(N,N,N), p=[0.9, 0.1]))

def update_matrix_simple(mat):
    mat_1 = np.zeros(mat.shape)
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            for z in range(mat.shape[2]):
                if mat[x,y,z] : #if it is a walker
                    direct = np.random.randint(6) #pick a random direction
                    mat_1[x,y,z] = 0 
                    if direct == 0 and x != 0 :
                        mat_1[x-1, y, z] = 1
                    elif direct == 1 and y != 0 :
                        mat_1[x, y-1, z] = 1
                    elif direct == 2 and z != 0 :
                        mat_1[x, y, z-1] = 1
                    elif direct == 3 and x != mat.shape[0] - 1:
                        mat_1[x+1, y, z] = 1
                    elif direct == 4 and y != mat.shape[1] - 1:
                        mat_1[x, y+1, z] = 1
                    elif direct == 5 and z != mat.shape[2] - 1:
                        mat_1[x, y, z+1] = 1
                    else : #it is at some edge
                        mat_1[x,y,z] = 1 #it stays there
    return(mat_1)

def update_matrix_anihilation(mat):
    mat_1 = np.zeros(mat.shape)
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            for z in range(mat.shape[2]):
                if mat[x,y,z] : #if it is a walker
                    direct = np.random.randint(6) #pick a random direction
                    mat_1[x,y,z] = 0 
                    if direct == 0 and x != 0:
                        mat_1[x-1, y, z] = 1 if not mat[x-1, y, z] else 0
                    elif direct == 1 and y != 0 :
                        mat_1[x, y-1, z] = 1 if not mat[x, y-1, z] else 0
                    elif direct == 2 and z != 0 :
                        mat_1[x, y, z-1] = 1 if not mat[x, y, z-1] else 0
                    elif direct == 3 and x != mat.shape[0] - 1:
                        mat_1[x+1, y, z] = 1 if not mat[x+1, y, z] else 0
                    elif direct == 4 and y != mat.shape[1] - 1:
                        mat_1[x, y+1, z] = 1 if not mat[x, y+1, z] else 0
                    elif direct == 5 and z != mat.shape[2] - 1:
                        mat_1[x, y, z+1] = 1 if not mat[x, y, z+1] else 0
                    else : #it is at some edge
                        mat_1[x,y,z] = 1 #it stays there
    return(mat_1)

#update_matrix = update_matrix_simple
update_matrix = update_matrix_anihilation

def plot_mat(mat, title=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(mat, edgecolor="k")
    if title :
        plt.title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def nb_of_walkrs(mat):
    """Returns number of walkers in matrix mat."""
    nb = 0
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            for z in range(mat.shape[2]):
                if mat[x,y,z] : nb+=1
    return(nb)

def iterate(A_0=init_matrix(), n_iteration = 10):
    """Returns a list of matrices, one for each iteration."""
    A = [A_0]
    for n in range(n_iteration):
        A_1 = update_matrix(A[n])
        #print(str(nb_of_walkrs(A_1))+ ' walkers at iteration '+str(n)) #unnecessary with plot_nb_walkrs
        A.append(A_1)
    return(A)


def plot_nb_walkrs(A):
    """ Takes a list of matrices """
    nb = []
    for mat in A :
        nb.append(nb_of_walkrs(mat))
    plt.figure()
    plt.plot(nb, marker='+', label='Measured data')
    t = np.arange(3, len(nb)+3, dtype=float)
    ro = t**(-1)*np.log(t)
    plt.plot(np.mean(nb/ro)*ro, label='Analytic model', color='k')
    plt.title('Evolution of the number of walkers')
    plt.xlabel('Number of iterations')
    plt.ylabel('Number of walkers')
    plt.legend()
    plt.show()

#%%
res = iterate(init_matrix(), 100)
if plotInitEnd: plot_mat(res[0], 'Initial state')
if plotInitEnd: plot_mat(res[-1], 'End state')
if plotIter : 
    for n in range(len(res)) :
        plot_mat(res[n], 'Iteration n'+str(n))

#%%
plot_nb_walkrs(res)
