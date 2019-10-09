from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

plotIter = False

def init_matrix(N=20):
    return(np.random.choice([0,1], size=(N,N,N), p=[0.9, 0.1]))

def update_matrix_simple(mat):
    #TODO : add anihilation
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
    plt.show()

def nb_of_walkrs(mat):
    nb = 0
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            for z in range(mat.shape[2]):
                if mat[x,y,z] : nb+=1
    return(nb)

def iterate(A_0=init_matrix(), n_iteration = 10):
    plot_mat(A_0, 'Init')
    A = [A_0]
    for n in range(n_iteration):
        A_1 = update_matrix(A[n])
        if plotIter : plot_mat(A_1, 'Iteration n'+str(n))
        print(str(nb_of_walkrs(A_1))+ ' walkers at iteration '+str(n))
        A.append(A_1)
    plot_mat(A[-1], 'End')
    return(A)


def plot_nb_walkrs(A):
    nb = []
    for mat in A :
        nb.append(nb_of_walkrs(mat))
    plt.figure()
    plt.plot(nb, marker='+')
    plt.title('Evolution of the number of walkers')
    plt.show()


plot_nb_walkrs(iterate(init_matrix(), 100))
